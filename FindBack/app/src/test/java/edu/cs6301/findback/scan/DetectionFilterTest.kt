package edu.cs6301.findback.scan

import edu.cs6301.findback.map.Pose4
import edu.cs6301.findback.map.Vec3
import edu.cs6301.findback.scan.ScanFixtures.boxAt
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

/** Plan sections 3 and 5: which detections of a keyframe are kept, and the rejection ranges. */
class DetectionFilterTest {
  /** Camera at the origin of a map that equals the world, looking along -Z at a wall [depthM] away. */
  private fun geom(depthM: Double = 1.0) =
    ScanFixtures.geometry(ScanFixtures.pose(Vec3.ZERO), ScanFixtures.flatDepth(depthM), Pose4.IDENTITY)

  private fun det(label: String, score: Float, u: Double = 960.0, v: Double = 540.0, side: Double = 100.0) =
    SceneDetection(boxAt(u, v, side), label, score)

  private fun one(d: SceneDetection, depthM: Double = 1.0) = DetectionFilter.apply(listOf(d), geom(depthM)).single()

  @Test
  fun aKeptDetectionGetsItsDepthPositionAndSize() {
    val r = one(det("Cup", 0.8f, u = 1237.5, v = 262.5, side = 138.75), depthM = 2.0)
    assertNull(r.reason)
    assertTrue(r.kept)
    assertEquals("cup", r.detection.label) // labels are compared in lower case
    assertEquals(2.0, r.depthM!!, 1e-12)
    // (u − cx)/fx = 0.2 and (cy − v)/fy = 0.2 at 2 m: 0.4 m right, 0.4 m up, 2 m ahead.
    assertEquals(0.4, r.mapPoint!!.x, 1e-9)
    assertEquals(0.4, r.mapPoint!!.y, 1e-9)
    assertEquals(-2.0, r.mapPoint!!.z, 1e-9)
    assertEquals(Math.sqrt(0.16 + 0.16 + 4.0), r.distanceM!!, 1e-9)
    assertEquals(0.2, r.sizeM!!, 1e-6) // 138.75 px × 2 m / 1387.5
  }

  @Test
  fun rejectionRangesOfTheDepth() {
    assertEquals(DetectionFilter.DEPTH_RANGE, one(det("cup", 0.8f), depthM = 0.15).reason)
    assertNull(one(det("cup", 0.8f), depthM = 0.25).reason)
    assertNull(one(det("cup", 0.8f, side = 60.0), depthM = 3.9).reason)
    assertEquals(DetectionFilter.DEPTH_RANGE, one(det("cup", 0.8f, side = 60.0), depthM = 4.5).reason)
    // No depth estimate under the box.
    val r = one(det("cup", 0.8f), depthM = 0.0)
    assertEquals(DetectionFilter.NO_DEPTH, r.reason)
    assertNull(r.mapPoint)
  }

  @Test
  fun thingsWiderThanSixtyCentimetresAreDropped() {
    // 800 px at 1 m is 0.58 m; 900 px is 0.65 m.
    assertNull(one(det("suitcase", 0.8f, side = 800.0)).reason)
    assertEquals(DetectionFilter.TOO_LARGE, one(det("suitcase", 0.8f, side = 900.0)).reason)
    // The longer side counts, whichever way the sensor image is turned.
    val tall = SceneDetection(Box(910f, 90f, 1010f, 990f), "umbrella", 0.8f)
    assertEquals(DetectionFilter.TOO_LARGE, one(tall).reason)
    val wide = SceneDetection(Box(510f, 490f, 1410f, 590f), "umbrella", 0.8f)
    assertEquals(DetectionFilter.TOO_LARGE, one(wide).reason)
  }

  @Test
  fun scoreLabelAndStopList() {
    assertEquals(DetectionFilter.LOW_SCORE, one(det("cup", 0.34f)).reason)
    assertNull(one(det("cup", 0.35f)).reason)
    assertEquals(DetectionFilter.NO_LABEL, one(det("???", 0.9f)).reason)
    assertEquals(DetectionFilter.NO_LABEL, one(det("  ", 0.9f)).reason)
    for (label in listOf("couch", "bed", "dining table", "toilet", "tv", "refrigerator", "oven", "sink", "chair", "bench", "potted plant", "Dining Table")) {
      assertEquals(label, DetectionFilter.STOP_LIST, one(det(label, 0.9f, side = 60.0)).reason)
    }
    assertEquals(11, ScanConfig.STOP_LABELS.size)
  }

  @Test
  fun aPersonIsNeverKeptAndTakesTheThingsOnItAlong() {
    val person = SceneDetection(Box(600f, 200f, 1000f, 900f), "person", 0.9f)
    val phoneInHand = det("cell phone", 0.8f, u = 800.0, v = 500.0, side = 80.0) // inside the person box
    val cupNearby = det("cup", 0.8f, u = 1300.0, v = 500.0, side = 80.0)
    val halfOn = SceneDetection(Box(940f, 400f, 1040f, 500f), "book", 0.8f) // 60% inside
    val edge = SceneDetection(Box(960f, 600f, 1060f, 700f), "remote", 0.8f) // 40% inside
    val reports = DetectionFilter.apply(listOf(person, phoneInHand, cupNearby, halfOn, edge), geom())
    assertEquals(listOf(DetectionFilter.PERSON, DetectionFilter.ON_PERSON, null, DetectionFilter.ON_PERSON, null), reports.map { it.reason })
    assertTrue(DetectionFilter.hasPerson(reports))
    assertNull(reports[0].mapPoint) // a person is never placed in the map

    // A person below the keep threshold is still a person: nothing on it is kept either.
    val faint = DetectionFilter.apply(listOf(person.copy(score = 0.22f), phoneInHand), geom())
    assertEquals(listOf(DetectionFilter.PERSON, DetectionFilter.ON_PERSON), faint.map { it.reason })
    assertTrue(DetectionFilter.hasPerson(faint))
    assertEquals(DetectionFilter.PERSON, one(det("Person", 0.99f)).reason)
    assertFalse(DetectionFilter.hasPerson(DetectionFilter.apply(listOf(cupNearby), geom())))
  }

  @Test
  fun sameLabelAndStrongOverlapIsOneDetection() {
    val a = SceneDetection(Box(900f, 500f, 1000f, 600f), "cup", 0.7f)
    val twin = SceneDetection(Box(910f, 500f, 1010f, 600f), "cup", 0.9f) // IoU 0.82
    val otherLabel = SceneDetection(Box(905f, 500f, 1005f, 600f), "bowl", 0.6f) // overlaps, another label
    val apart = SceneDetection(Box(950f, 500f, 1050f, 600f), "cup", 0.5f) // IoU with `twin` 0.43
    val reports = DetectionFilter.apply(listOf(a, twin, otherLabel, apart), geom())
    assertEquals(listOf(DetectionFilter.DUPLICATE, null, null, null), reports.map { it.reason })
    // Exactly at the limit is not "over 0.6": 100 px boxes shifted by 25 px have IoU 0.6.
    val b1 = SceneDetection(Box(0f, 500f, 100f, 600f), "cup", 0.7f)
    val b2 = SceneDetection(Box(25f, 500f, 125f, 600f), "cup", 0.6f)
    assertEquals(0.6f, b1.box.iou(b2.box), 1e-6f)
    assertEquals(listOf<String?>(null, null), DetectionFilter.apply(listOf(b1, b2), geom()).map { it.reason })
  }

  @Test
  fun atMostTenDetectionsAreKeptTheBestByScore() {
    val many = (0 until 13).map { i -> det("item$i", 0.40f + 0.03f * i, u = 200.0 + 120.0 * i, v = 540.0, side = 60.0) }
    val reports = DetectionFilter.apply(many, geom())
    assertEquals(10, reports.count { it.kept })
    // The three lowest scores are the ones cut.
    assertEquals(listOf(0, 1, 2), reports.withIndex().filter { it.value.reason == DetectionFilter.OVER_CAP }.map { it.index })
    // Detections that are dropped for another reason do not use up the ten.
    val withJunk = many + listOf(det("person", 0.99f, u = 1800.0, v = 100.0, side = 40.0), det("chair", 0.99f, u = 1800.0, v = 900.0, side = 40.0))
    assertEquals(10, DetectionFilter.apply(withJunk, geom()).count { it.kept })
  }
}
