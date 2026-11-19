package edu.cs6301.findback.scan

import edu.cs6301.findback.map.Pose4
import edu.cs6301.findback.map.Vec3
import edu.cs6301.findback.map.distance
import edu.cs6301.findback.memory.EmbeddingFile
import edu.cs6301.findback.scan.ScanFixtures.embedding
import edu.cs6301.findback.scan.ScanFixtures.observe
import edu.cs6301.findback.scan.ScanFixtures.poseLookingAt
import kotlin.math.cos
import kotlin.math.sin
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertSame
import org.junit.Assert.assertTrue
import org.junit.Test

/** v0.8 study logging (docs/PLAN_v0.8_study_logging.md): the person rule, view retention, poses and lines. */
class StudyLogTest {
  // ------------------------------------------------------------------ person rule (plan rule 3)

  private val person = SceneDetection(Box(1311.5f, 211.25f, 1517.75f, 877.125f), "person", 0.6f)

  @Test
  fun aViewThatTouchesThePaddedPersonBoxIsNeverCut() {
    val persons = PersonRule.fromDetections(listOf(person))
    // Padded by 10 % of its size on every side: 20.625 px left and right, 66.5875 px up and down.
    val padded = persons.single()
    assertEquals(1311.5f - 20.625f, padded.left, 1e-3f)
    assertEquals(877.125f + 66.5875f, padded.bottom, 1e-3f)
    // Left of the person box itself but inside its padding: no view.
    val near = Box(1200f, 300f, 1300f, 400f)
    assertEquals(0f, near.intersectionArea(person.box), 0f)
    assertFalse(PersonRule.allowsView(near, persons))
    assertEquals(9.125 / 100.0, PersonRule.overlap(near, persons), 1e-4) // 9.125 of its 100 px width
    // Clear of the padding: a view, overlap 0.
    val clear = Box(1100f, 300f, 1290f, 400f)
    assertTrue(PersonRule.allowsView(clear, persons))
    assertEquals(0.0, PersonRule.overlap(clear, persons), 0.0)
    // Touching edges have no area in common: allowed.
    val touching = Box(1100f, 300f, padded.left, 400f)
    assertTrue(PersonRule.allowsView(touching, persons))
    // Inside the person: overlap 1.
    assertEquals(1.0, PersonRule.overlap(Box(1400f, 400f, 1450f, 500f), persons), 0.0)
  }

  @Test
  fun onlyPersonBoxesAtOrAboveTheFloorBlockAView() {
    val inside = Box(1400f, 400f, 1450f, 500f)
    val weak = person.copy(score = 0.19f)
    val floor = person.copy(score = StudyConfig.PERSON_MIN_SCORE)
    val spaced = person.copy(label = " Person ")
    val cup = person.copy(label = "cup")
    assertTrue(PersonRule.allowsView(inside, PersonRule.fromDetections(listOf(weak, cup))))
    assertFalse(PersonRule.allowsView(inside, PersonRule.fromDetections(listOf(floor))))
    assertFalse(PersonRule.allowsView(inside, PersonRule.fromDetections(listOf(spaced))))
    assertTrue(PersonRule.allowsView(inside, emptyList()))
    assertEquals(0.20f, StudyConfig.PERSON_MIN_SCORE, 0f)
  }

  @Test
  fun aPersonIsNeverAnObservationAndItsBoxIsNeverWritten() {
    // A cup next to a person in one keyframe, through the real filter.
    val cupAt = Vec3(0.4, 0.9, -1.1)
    val pose = poseLookingAt(cupAt)
    val cupObs = observe(cupAt, pose, "cup", embedding(0))!!
    val geom = ScanFixtures.geometry(pose, ScanFixtures.flatDepth(cupObs.depthM))
    val reports = DetectionFilter.apply(listOf(person, SceneDetection(cupObs.box, "cup", 0.8f)), geom)
    assertEquals(DetectionFilter.PERSON, reports[0].reason) // the person is dropped, never kept
    assertTrue(reports[1].kept)
    val persons = PersonRule.boxes(reports)
    assertEquals(1, persons.size)
    val rect = StudyImage.viewRect(cupObs.box, ScanFixtures.W, ScanFixtures.H)
    assertTrue("the cup is far from the person: its view is allowed", PersonRule.allowsView(rect, persons))

    val line = line(cupObs, personInFrame = true, overlap = PersonRule.overlap(rect, persons)).toString()
    for (n in listOf("1311.5", "211.25", "1517.75", "877.125")) assertFalse("person box value $n written", line.contains(n))
    val keys = HashSet<String>()
    collectKeys(JSONObject(line), keys)
    assertEquals(setOf("personInFrame", "personOverlap"), keys.filter { it.lowercase().contains("person") }.toSet())
    assertEquals(true, JSONObject(line).getBoolean("personInFrame"))
  }

  private fun collectKeys(o: Any, out: MutableSet<String>) {
    when (o) {
      is JSONObject -> for (k in o.keys()) {
        out.add(k)
        collectKeys(o.get(k), out)
      }
      is JSONArray -> for (i in 0 until o.length()) collectKeys(o.get(i), out)
    }
  }

  // ------------------------------------------------------------------ image arithmetic

  @Test
  fun theViewRectangleAndItsPixelSizeAreThoseOfTheDevice() {
    val b = Box(100f, 200f, 300f, 350f)
    val r = StudyImage.viewRect(b, 1920, 1080)
    assertEquals(Box(84f, 188f, 316f, 362f), r) // 8 % of 200 and of 150
    assertEquals(232 to 174, StudyImage.uprightCropSize(r, 1920, 1080, 0))
    assertEquals(174 to 232, StudyImage.uprightCropSize(r, 1920, 1080, 90))
    // At the image edge the padded box is clamped.
    val edge = StudyImage.viewRect(Box(1850f, -5f, 1925f, 60f), 1920, 1080)
    assertEquals(0f, edge.top, 0f)
    assertEquals(1920f, edge.right, 0f)
    assertEquals(1920 - edge.left.toInt() to 65, StudyImage.uprightCropSize(edge, 1920, 1080, 180))
  }

  // ------------------------------------------------------------------ camera pose in the map frame

  @Test
  fun theMapPoseTakesCameraPointsToMapPoints() {
    val tWM = Pose4(1.2, -0.4, 3.1, 0.6)
    val pose = ScanFixtures.pose(Vec3(0.5, 1.4, -2.0), yawDeg = 37.0, pitchDeg = -20.0, rollDeg = 80.0)
    val m = StudyGeometry.mapPose(pose, tWM)
    for (p in listOf(Vec3(0.0, 0.0, -1.0), Vec3(0.3, -0.2, -0.7), Vec3(1.0, 2.0, 3.0))) {
      val expected = tWM.inverse().transformPoint(pose.toWorld(p))
      assertTrue(distance(expected, m.toWorld(p)) < 1e-9)
    }
    val q = m.quaternion()
    assertEquals(1.0, q.sumOf { it * it }, 1e-12) // still a unit quaternion
  }

  @Test
  fun yawAndPitchFollowThePoseCsvConvention() {
    val tWM = Pose4(0.0, 0.0, 0.0, 0.0)
    // Looking along map -Z: yaw 0. Turned left by 90° (looking along -X): yaw +90.
    val ahead = StudyGeometry.forwardMap(ScanFixtures.pose(Vec3.ZERO), tWM)
    assertEquals(0.0, StudyGeometry.yawDeg(ahead)!!, 1e-9)
    val left = StudyGeometry.forwardMap(ScanFixtures.pose(Vec3.ZERO, yawDeg = 90.0), tWM)
    assertEquals(90.0, StudyGeometry.yawDeg(left)!!, 1e-9)
    val up = StudyGeometry.forwardMap(ScanFixtures.pose(Vec3.ZERO, pitchDeg = 30.0), tWM)
    assertEquals(30.0, StudyGeometry.pitchDeg(up), 1e-9)
    assertNull(StudyGeometry.yawDeg(Vec3(0.0, -1.0, 0.0)))
    // A map frame turned against the world turns the heading back.
    val turned = StudyGeometry.forwardMap(ScanFixtures.pose(Vec3.ZERO, yawDeg = 90.0), Pose4(0.0, 0.0, 0.0, Math.toRadians(90.0)))
    assertEquals(0.0, StudyGeometry.yawDeg(turned)!!, 1e-9)
  }

  // ------------------------------------------------------------------ view retention (plan rule 2)

  private val item = Vec3(1.0, 0.8, -2.0)

  /** A camera 1 m from the item at heading [deg] (horizontal circle). */
  private fun cand(id: String, deg: Double, quality: Double = 1.0, at: Vec3 = item): ViewCandidate {
    val a = Math.toRadians(deg)
    return ViewCandidate(id, at, Vec3(at.x + sin(a), at.y, at.z + cos(a)), quality)
  }

  @Test
  fun aFullGroupDropsTheNearDuplicateDirection() {
    val s = ViewSelector()
    val g = s.assign(item)
    for ((id, deg) in listOf("a" to 0.0, "b" to 5.0, "c" to 60.0, "d" to 120.0)) {
      val p = s.propose(g, cand(id, deg))
      assertTrue("a group with room keeps every view", p.keep)
      assertNull(p.evict)
      s.commit(p)
    }
    assertEquals(4, s.retained)
    // A new direction replaces one of the two near duplicates: {a, c, d, e} has the widest spread.
    val e = s.propose(g, cand("e", 180.0))
    assertTrue(e.keep)
    assertEquals("b", e.evict!!.obsId)
    s.commit(e)
    assertEquals(listOf("a", "c", "d", "e"), s.viewsOf(g).map { it.obsId })
    assertEquals(4, s.retained)
    // A direction 2° from one already kept adds nothing.
    val f = s.propose(g, cand("f", 62.0))
    assertFalse(f.keep)
    assertEquals(StudyConfig.VIEW_NOT_COMPLEMENTARY, f.reason)
    assertEquals(setOf("a", "c", "d", "e"), s.retainedIds())
  }

  @Test
  fun withinOneSpreadStepQualityDecidesAndTiesKeepWhatIsKept() {
    val kept = listOf(cand("a", 0.0), cand("b", 65.0), cand("c", 130.0), cand("d", 195.0))
    // 66° instead of 65°: the spread stays in the same 10° step (64° or 65°), so quality decides.
    val sharp = cand("e", 66.0, quality = 100.0)
    assertSame(kept[1], ViewRetention.leaveOut(kept, sharp))
    val blurry = cand("e", 66.0, quality = 0.5)
    assertSame(blurry, ViewRetention.leaveOut(kept, blurry))
    val equal = cand("e", 66.0, quality = 1.0)
    assertSame(equal, ViewRetention.leaveOut(kept, equal)) // a tie keeps the view already on disk
    // A group with room takes anything.
    assertNull(ViewRetention.leaveOut(kept.take(3), equal))
    assertEquals(65.0, ViewRetention.spreadDeg(kept), 1e-9)
    assertEquals(6, ViewRetention.spreadBin(kept))
  }

  @Test
  fun groupsArePlacesWithFixedAnchors() {
    val s = ViewSelector()
    val g0 = s.assign(item)
    assertEquals(g0, s.assign(Vec3(item.x + 0.25, item.y, item.z))) // within 0.30 m
    val g1 = s.assign(Vec3(item.x + 0.35, item.y, item.z)) // farther from the (unmoved) anchor
    assertTrue(g1 != g0)
    // Between the two anchors (0.30 and 0.05 m away): the nearer one.
    assertEquals(g1, s.assign(Vec3(item.x + 0.30, item.y, item.z)))
    assertEquals(2, s.groupCount)
  }

  @Test
  fun theScanCapStopsNewViewsButNotReplacements() {
    val s = ViewSelector(perGroup = 2, perScan = 2)
    val g = s.assign(item)
    s.commit(s.propose(g, cand("a", 0.0)))
    s.commit(s.propose(g, cand("b", 3.0)))
    assertEquals(2, s.retained)
    val other = s.assign(Vec3(item.x + 1.0, item.y, item.z))
    val capped = s.propose(other, cand("c", 0.0, at = Vec3(item.x + 1.0, item.y, item.z)))
    assertFalse(capped.keep)
    assertEquals(StudyConfig.VIEW_SCAN_CAP, capped.reason)
    // A better direction in the full group still replaces a view: the count stays at the cap.
    val wide = s.propose(g, cand("d", 90.0))
    assertTrue(wide.keep)
    s.commit(wide)
    assertEquals(2, s.retained)
    assertEquals(2, s.retainedIds().size)
  }

  @Test
  fun aProposalChangesNothingUntilCommitted() {
    val s = ViewSelector()
    val g = s.assign(item)
    val p = s.propose(g, cand("a", 0.0))
    s.propose(g, cand("a", 0.0))
    assertEquals(0, s.retained)
    assertTrue(s.retainedIds().isEmpty())
    s.commit(p)
    assertEquals(setOf("a"), s.retainedIds())
    s.commit(ViewProposal(cand("x", 10.0), g, false, null, StudyConfig.VIEW_SCAN_CAP)) // not kept: ignored
    assertEquals(1, s.retained)
  }

  // ------------------------------------------------------------------ lines

  private fun line(o: Observation, personInFrame: Boolean = false, overlap: Double = 0.0, result: ObservationResult? = null): JSONObject {
    val pose = poseLookingAt(o.mapPoint)
    val c =
      ObsJson.Common(
        obsId = StudyConfig.obsId("scan0001", 12, 0), kind = StudyConfig.KIND_SCAN, scan = "scan0001", session = "sess0001",
        mapId = "map00001", room = "room1", keyframe = 12, det = 0, tMs = 1_000L, studyTag = "exp=pilot",
        imageWidth = ScanFixtures.W, imageHeight = ScanFixtures.H, rotation = 90, pose = pose, mapToWorld = ScanFixtures.T_WM,
        intrinsics = ScanFixtures.K, embedding = o.embedding, embeddingModel = "mobilenet_v3_small",
        personInFrame = personInFrame, personOverlap = overlap,
      )
    val rect = StudyImage.viewRect(o.box, ScanFixtures.W, ScanFixtures.H)
    val view = ObsJson.ViewFields(3, rect, StudyConfig.viewPath("scan0001", c.obsId), 120, 140, StudyConfig.VIEW_KEPT, null)
    return ObsJson.scanObservation(c, o, 4, view, result)
  }

  @Test
  fun anObservationLineHasThePlannedFields() {
    val cupAt = Vec3(0.4, 0.9, -1.1)
    val o = observe(cupAt, poseLookingAt(cupAt), "cup", embedding(2, 0.6))!!
    val r =
      ObservationResult(
        o, ScanSession.KIND_KNOWN, "e0000001", 0.81f, 0.04, true,
        listOf(MergeCandidate("e0000001", 0.04, 0.81f, true, true, false), MergeCandidate("e0000002", 0.6, 0.2f, false, false, true)),
      )
    val j = line(o, result = r)
    assertEquals(1, j.getInt("v"))
    assertEquals("obs", j.getString("type"))
    assertEquals("scan0001-12-0", j.getString("obs"))
    assertEquals("scan", j.getString("kind"))
    for (k in listOf(
      "scan", "session", "mapId", "room", "kf", "det", "rawDet", "t", "studyTag", "label", "score", "box", "image", "rotation",
      "depthM", "sizeM", "distanceM", "mapPoint", "worldPoint", "quality", "camera", "mapToWorld", "intrinsics", "emb", "embDim",
      "embModel", "personInFrame", "personOverlap", "viewGroup", "viewRect", "view", "viewPx", "viewReason", "viewReplaces", "decision",
    )) assertTrue("missing $k", j.has(k))
    assertArrayEquals(o.embedding, EmbeddingFile.decode(j.getString("emb")), 0f)
    assertEquals(8, j.getInt("embDim"))
    assertEquals("views/scan0001/scan0001-12-0.jpg", j.getString("view"))
    assertEquals(120, j.getJSONArray("viewPx").getInt(0))
    val cam = j.getJSONObject("camera")
    assertEquals(4, cam.getJSONObject("world").getJSONArray("q").length())
    assertEquals(3, cam.getJSONObject("map").getJSONArray("forward").length())
    val d = j.getJSONObject("decision")
    assertEquals("baseline", d.getString("matcher"))
    assertEquals("known", d.getString("kind"))
    assertEquals("e0000001", d.getString("entry"))
    assertTrue(d.getBoolean("byLabel"))
    assertEquals(2, d.getJSONArray("candidates").length())
    assertTrue(d.getJSONArray("candidates").getJSONObject(1).getBoolean("tentative"))
    assertEquals(0.6, d.getJSONArray("candidates").getJSONObject(1).getDouble("distM"), 0.0)
    // A new entry has no match: no similarity, distance or label flag.
    val fresh = line(o, result = ObservationResult(o, ScanSession.KIND_NEW, "e0000003", null, null)).getJSONObject("decision")
    assertTrue(fresh.isNull("sim"))
    assertTrue(fresh.isNull("byLabel"))
    assertEquals(0, fresh.getJSONArray("candidates").length())

    val ev = ObsJson.viewEvicted("scan0001", ViewCandidate("scan0001-3-1", cupAt, Vec3.ZERO, 1.0), "views/scan0001/scan0001-3-1.jpg", "scan0001-12-0", 3, 2_000L)
    assertEquals("view_evicted", ev.getString("type"))
    assertEquals("scan0001-3-1", ev.getString("obs"))
    assertEquals("scan0001-12-0", ev.getString("by"))
  }
}
