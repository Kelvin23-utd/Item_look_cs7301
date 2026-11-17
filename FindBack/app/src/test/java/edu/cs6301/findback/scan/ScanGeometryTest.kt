package edu.cs6301.findback.scan

import edu.cs6301.findback.map.Pose4
import edu.cs6301.findback.map.Vec3
import edu.cs6301.findback.scan.ScanFixtures.K
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

/** Plan section 5: pixel + depth to map point and back, the depth mapping, the rejection ranges. */
class ScanGeometryTest {
  private fun assertVec(expected: Vec3, actual: Vec3, tol: Double) {
    assertEquals("x", expected.x, actual.x, tol)
    assertEquals("y", expected.y, actual.y, tol)
    assertEquals("z", expected.z, actual.z, tol)
  }

  @Test
  fun cameraFrameFollowsTheArcoreConvention() {
    // Camera.getPose(): +X right and +Y up relative to the image readout, -Z forward.
    val right = Pinhole.unproject(K.cx + 100, K.cy, 2.0, K)
    assertTrue(right.x > 0)
    assertEquals(0.0, right.y, 1e-12)
    val above = Pinhole.unproject(K.cx, K.cy - 100, 2.0, K) // a smaller v is higher in the image
    assertTrue(above.y > 0)
    assertEquals(0.0, above.x, 1e-12)
    // The depth is the distance to the camera plane, not along the ray.
    assertEquals(-2.0, right.z, 1e-12)
    assertEquals(-2.0, Pinhole.unproject(0.0, 0.0, 2.0, K).z, 1e-12)
    // The plan's formula, written out.
    val p = Pinhole.unproject(1500.0, 200.0, 1.5, K)
    assertVec(Vec3(1.5 * (1500.0 - 960.0) / 1387.5, 1.5 * (540.0 - 200.0) / 1387.5, -1.5), p, 1e-12)
  }

  @Test
  fun projectInvertsUnproject() {
    for ((u, v, d) in listOf(Triple(960.0, 540.0, 1.0), Triple(12.5, 1070.0, 0.35), Triple(1900.0, 3.0, 3.9))) {
      val pr = Pinhole.project(Pinhole.unproject(u, v, d, K), K)!!
      assertEquals(u, pr.u, 1e-9)
      assertEquals(v, pr.v, 1e-9)
      assertEquals(d, pr.depthM, 1e-12)
    }
    assertNull(Pinhole.project(Vec3(0.1, 0.1, 0.5), K)) // behind the camera
    assertNull(Pinhole.project(Vec3(0.1, 0.1, 0.0), K)) // in the camera plane
  }

  @Test
  fun cameraPoseRotatesLikeAPose4AndInverts() {
    // A pure yaw must agree with Pose4 (the map transform uses the same sense of rotation).
    val yaw = Math.toRadians(37.0)
    val q = Pose4(0.5, 1.0, -2.0, yaw).quaternion()
    val cam = CameraPose.of(floatArrayOf(0.5f, 1.0f, -2.0f), q)
    val v = Vec3(0.3, -0.2, 1.7)
    assertVec(Pose4(0.5, 1.0, -2.0, yaw).transformPoint(v), cam.toWorld(v), 1e-6)

    val p = ScanFixtures.pose(Vec3(1.0, 1.4, -0.5), yawDeg = 130.0, pitchDeg = -25.0, rollDeg = 90.0)
    assertVec(v, p.toCamera(p.toWorld(v)), 1e-12)
    // No rotation: the camera looks along world -Z.
    assertVec(Vec3(0.0, 0.0, -1.0), ScanFixtures.pose(Vec3.ZERO).toWorld(Vec3(0.0, 0.0, -1.0)), 1e-12)
    // Yaw +90° turns the view from -Z to -X; pitch +90° makes it look straight up.
    assertVec(Vec3(-1.0, 0.0, 0.0), ScanFixtures.pose(Vec3.ZERO, yawDeg = 90.0).toWorld(Vec3(0.0, 0.0, -1.0)), 1e-12)
    assertVec(Vec3(0.0, 1.0, 0.0), ScanFixtures.pose(Vec3.ZERO, pitchDeg = 90.0).toWorld(Vec3(0.0, 0.0, -1.0)), 1e-12)
    assertEquals(40.0, ScanFixtures.pose(Vec3.ZERO, yawDeg = 10.0).angleTo(ScanFixtures.pose(Vec3.ZERO, yawDeg = 50.0)), 1e-9)
  }

  @Test
  fun aMapPointProjectsAndUnprojectsToItselfForSeveralCameraPoses() {
    val item = Vec3(0.4, 0.9, -1.1) // map frame
    val world = ScanFixtures.T_WM.transformPoint(item)
    val poses =
      listOf(
        ScanFixtures.pose(Vec3(world.x, world.y, world.z + 1.0)),
        ScanFixtures.pose(Vec3(world.x + 0.3, world.y + 0.2, world.z + 1.5), yawDeg = 8.0, pitchDeg = -6.0),
        // Phone held upright: the sensor image is rolled 90° against the world.
        ScanFixtures.pose(Vec3(world.x - 0.2, world.y + 0.1, world.z + 0.8), yawDeg = -5.0, pitchDeg = 4.0, rollDeg = 90.0),
        ScanFixtures.pose(Vec3(world.x + 1.2, world.y + 0.5, world.z + 0.1), yawDeg = 80.0, pitchDeg = -20.0, rollDeg = -90.0),
        ScanFixtures.pose(Vec3(world.x, world.y + 1.0, world.z + 0.4), pitchDeg = -65.0, rollDeg = 33.0),
      )
    for ((i, pose) in poses.withIndex()) {
      val geom = ScanFixtures.geometry(pose, ScanFixtures.flatDepth(1.0))
      val pr = geom.projectMap(item)
      assertNotNull("pose $i sees the item", pr)
      assertTrue("pose $i has the item in the image", geom.inImage(pr!!, 0.0))
      assertVec(item, geom.mapPoint(pr.u, pr.v, pr.depthM), 1e-9)
      assertVec(world, geom.worldPoint(pr.u, pr.v, pr.depthM), 1e-9)
      val again = geom.projectWorld(world)!!
      assertEquals(pr.u, again.u, 1e-9)
      assertEquals(pr.v, again.v, 1e-9)
    }
    // Where the camera stands, in the map frame.
    assertVec(ScanFixtures.T_WM.inverse().transformPoint(poses[1].position), ScanFixtures.geometry(poses[1], ScanFixtures.flatDepth(1.0)).cameraMap, 1e-12)
  }

  @Test
  fun aPointBesideTheCameraIsNotInTheImage() {
    val geom = ScanFixtures.geometry(ScanFixtures.pose(Vec3.ZERO), ScanFixtures.flatDepth(1.0), Pose4.IDENTITY)
    assertNull(geom.projectMap(Vec3(0.0, 0.0, 1.0))) // behind
    val far = geom.projectMap(Vec3(5.0, 0.0, -1.0))!! // in front, far to the right
    assertTrue(!geom.inImage(far, 0.0))
    val edge = geom.projectMap(Vec3(0.66, 0.0, -1.0))!! // u ≈ 1876: inside, but in the 8% margin
    assertTrue(geom.inImage(edge, 0.0))
    assertTrue(!geom.inImage(edge, ScanConfig.VIEW_MARGIN))
  }

  @Test
  fun intrinsicsScaleWithTheImage() {
    val half = K.scaledTo(960, 540)
    assertEquals(K.fx / 2, half.fx, 1e-12)
    assertEquals(K.cy / 2, half.cy, 1e-12)
    assertTrue(K.scaledTo(1920, 1080) === K)
    // The same ray through the same scene point.
    val full = Pinhole.unproject(1500.0, 300.0, 2.0, K)
    assertVec(full, Pinhole.unproject(750.0, 150.0, 2.0, half), 1e-12)
  }

  @Test
  fun affineFromThreePoints() {
    // Texture coordinates of a depth image that crops the CPU image: 10% off the top and bottom.
    val truth = Affine2(160.0 / 1920, 0.0, 0.0, 0.0, 90.0 / 864, -108.0 * 90.0 / 864)
    val src = doubleArrayOf(960.0, 540.0, 1440.0, 540.0, 960.0, 810.0)
    val dst = DoubleArray(6)
    for (i in 0 until 3) {
      dst[2 * i] = truth.mapX(src[2 * i], src[2 * i + 1])
      dst[2 * i + 1] = truth.mapY(src[2 * i], src[2 * i + 1])
    }
    val fit = Affine2.fromThreePoints(src, dst)!!
    for ((x, y) in listOf(0.0 to 0.0, 1919.0 to 1079.0, 480.0 to 270.0)) {
      assertEquals(truth.mapX(x, y), fit.mapX(x, y), 1e-9)
      assertEquals(truth.mapY(x, y), fit.mapY(x, y), 1e-9)
    }
    // A rotated mapping (depth image turned a quarter against the CPU image) is fitted as well.
    val turned = Affine2(0.0, -0.1, 108.0, 0.1, 0.0, 0.0)
    for (i in 0 until 3) {
      dst[2 * i] = turned.mapX(src[2 * i], src[2 * i + 1])
      dst[2 * i + 1] = turned.mapY(src[2 * i], src[2 * i + 1])
    }
    val fit2 = Affine2.fromThreePoints(src, dst)!!
    assertEquals(turned.mapX(100.0, 900.0), fit2.mapX(100.0, 900.0), 1e-9)
    assertEquals(turned.mapY(100.0, 900.0), fit2.mapY(100.0, 900.0), 1e-9)
    // Three points on a line do not define a map.
    assertNull(Affine2.fromThreePoints(doubleArrayOf(0.0, 0.0, 1.0, 1.0, 2.0, 2.0), dst))
  }

  @Test
  fun depthMappingReadsThePixelUnderAnImagePoint() {
    // Depth pixel (x, y) holds x + 100·y millimetres, so every read names its pixel.
    val depth = DepthMap(ScanFixtures.DW, ScanFixtures.DH, ShortArray(ScanFixtures.DW * ScanFixtures.DH) { i -> ((i % 160) + 100 * (i / 160) + 1000).toShort() }, ScanFixtures.FULL)
    assertEquals(1000 + 80 + 100 * 45, depth.mmAt(80, 45))
    // CPU pixel (966, 546) is the centre of depth pixel (80, 45); a 1×1 patch returns exactly it.
    assertEquals((1000 + 80 + 100 * 45) / 1000.0, depth.medianPatch(966.0, 546.0, 1)!!, 1e-12)
    // CPU pixels 12 apart are one depth pixel apart (1920 / 160).
    assertEquals((1000 + 81 + 100 * 45) / 1000.0, depth.medianPatch(978.0, 546.0, 1)!!, 1e-12)
    assertEquals((1000 + 80 + 100 * 46) / 1000.0, depth.medianPatch(966.0, 558.0, 1)!!, 1e-12)
    // The 5×5 patch is symmetric about its centre, so its median is the centre value.
    assertEquals((1000 + 80 + 100 * 45) / 1000.0, depth.medianPatch(966.0, 546.0)!!, 1e-12)
    // Outside the depth image there is no depth.
    assertNull(depth.medianPatch(-5.0, 540.0))
    assertNull(depth.medianPatch(960.0, 1081.0))
    assertEquals(1.0, depth.validShare(), 1e-12)
  }

  @Test
  fun depthValuesAreUnsignedAndZeroIsInvalid() {
    // 40 000 mm does not fit a signed short: it must read back as 40 m, not as a negative number.
    val far = DepthMap(2, 1, shortArrayOf(40000.toShort(), 0), Affine2.IDENTITY)
    assertEquals(40000, far.mmAt(0, 0))
    assertEquals(40.0, far.medianPatch(0.5, 0.5, 1)!!, 1e-12)
    assertNull(far.medianPatch(1.5, 0.5, 1))
    assertEquals(0.5, far.validShare(), 1e-12)
  }

  @Test
  fun boxDepthIsTheMedianOfTheValidPixelsInTheCentralPart() {
    // A 240 px box centred at (960, 540): its central 40% is 96 px, 8 depth pixels a side.
    val box = ScanFixtures.boxAt(960.0, 540.0, 240.0)
    val central = box.central(ScanConfig.BOX_CENTRAL_FRACTION)
    assertEquals(96f, central.width, 1e-3f)
    // The object (1.2 m) fills the central part; the background (3 m) is outside it.
    val depth = ScanFixtures.depthOf { x, y -> if (x in 76..83 && y in 41..48) 1.2 else 3.0 }
    assertEquals(1.2, depth.medianInBox(central)!!, 1e-12)
    // A few holes and a few far outliers inside do not move the median.
    val noisy =
      ScanFixtures.depthOf { x, y ->
        when {
          x !in 76..83 || y !in 41..48 -> 3.0
          x == 76 -> 0.0
          y == 41 && x < 80 -> 3.0
          else -> 1.2
        }
      }
    assertEquals(1.2, noisy.medianInBox(central)!!, 1e-12)
    // No valid pixel in the region: no depth.
    assertNull(ScanFixtures.depthOf { x, y -> if (x in 76..83 && y in 41..48) 0.0 else 3.0 }.medianInBox(central))
    // A region narrower than one depth pixel uses the pixel under its middle.
    assertEquals(1.2, depth.medianInBox(ScanFixtures.boxAt(960.0, 540.0, 4.0))!!, 1e-12)
    // A box in the part of the CPU image that the depth image does not cover has no depth.
    val cropped = Affine2(160.0 / 1920, 0.0, 0.0, 0.0, 90.0 / 864, -108.0 * 90.0 / 864)
    assertNull(ScanFixtures.flatDepth(1.0, cropped).medianInBox(ScanFixtures.boxAt(960.0, 40.0, 60.0)))
    assertNotNull(ScanFixtures.flatDepth(1.0, cropped).medianInBox(ScanFixtures.boxAt(960.0, 540.0, 60.0)))
  }

  @Test
  fun boxHelpers() {
    val a = Box(0f, 0f, 100f, 100f)
    val b = Box(50f, 0f, 150f, 100f)
    assertEquals(5000f, a.intersectionArea(b), 1e-3f)
    assertEquals(1f / 3f, a.iou(b), 1e-6f)
    assertEquals(0f, a.iou(Box(200f, 200f, 300f, 300f)), 0f)
    assertEquals(1f, a.iou(a), 1e-6f)
    assertEquals(Box(-8f, -8f, 108f, 108f), a.padded(0.08f))
  }
}
