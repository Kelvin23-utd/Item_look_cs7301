package edu.cs6301.findback.map

import java.util.Random
import kotlin.math.PI
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class FixEstimatorTest {
  private val truth = Pose4(1.0, 0.8, -2.0, 0.6)

  private fun noisy(rnd: Random, posM: Double, yawDeg: Double, base: Pose4 = truth) =
    Pose4(
      base.x + rnd.nextGaussian() * posM,
      base.y + rnd.nextGaussian() * posM,
      base.z + rnd.nextGaussian() * posM,
      base.yaw + Math.toRadians(rnd.nextGaussian() * yawDeg),
    )

  @Test
  fun convergesOnSteadySamples() {
    val w = ImageFixWindow()
    val rnd = Random(1)
    var t = 0L
    repeat(19) {
      w.add(FixSample(t, noisy(rnd, 0.002, 0.2), 0.6))
      t += 33
    }
    assertNull("19 samples are not enough", w.estimate(t, allowLoose = true))
    w.add(FixSample(t, noisy(rnd, 0.002, 0.2), 0.6))
    val e = w.estimate(t, allowLoose = true)
    assertNotNull(e)
    e!!
    assertEquals("good", e.quality)
    assertTrue(e.samples >= ImageFixWindow.AGREE_SAMPLES)
    assertTrue(distance(e.pose.translation, truth.translation) < 0.003)
    assertTrue(Math.toDegrees(Pose4.angleDiff(e.pose.yaw, truth.yaw)) < 0.3)
    assertEquals(0.6, e.cameraDistanceM, 1e-9)
  }

  @Test
  fun outliersDoNotBlockConvergence() {
    val w = ImageFixWindow()
    val rnd = Random(2)
    var t = 0L
    repeat(30) { i ->
      val p = if (i % 6 == 0) noisy(rnd, 0.2, 10.0) else noisy(rnd, 0.002, 0.2)
      w.add(FixSample(t, p, 0.5))
      t += 33
    }
    val e = w.estimate(t, allowLoose = false)
    assertNotNull(e)
    assertTrue(distance(e!!.pose.translation, truth.translation) < 0.005)
  }

  @Test
  fun noisySamplesGiveALooseFixOnlyAfterTheTimeout() {
    val w = ImageFixWindow()
    val rnd = Random(3)
    var t = 0L
    while (t < 2900) {
      w.add(FixSample(t, noisy(rnd, 0.05, 5.0), 0.7))
      t += 33
    }
    assertNull(w.estimate(t, allowLoose = true))
    while (t < 3200) {
      w.add(FixSample(t, noisy(rnd, 0.05, 5.0), 0.7))
      t += 33
    }
    assertNull("refreshes never accept loose fixes", w.estimate(t, allowLoose = false))
    val e = w.estimate(t, allowLoose = true)
    assertNotNull(e)
    assertEquals("loose", e!!.quality)
    assertTrue(distance(e.pose.translation, truth.translation) < 0.05)
  }

  @Test
  fun gatedTimeIgnoresGaps() {
    val w = ImageFixWindow()
    w.add(FixSample(0, truth, 0.5))
    w.add(FixSample(100, truth, 0.5))
    w.add(FixSample(5000, truth, 0.5)) // picture was out of view for ~5 s
    w.add(FixSample(5100, truth, 0.5))
    assertEquals(200L, w.gatedMs)
    w.reset()
    assertEquals(0L, w.gatedMs)
    assertEquals(0, w.size)
  }

  @Test
  fun oldSamplesAgeOut() {
    val w = ImageFixWindow()
    var t = 0L
    repeat(25) {
      w.add(FixSample(t, truth, 0.5))
      t += 33
    }
    // Two seconds later, without new samples, the window no longer counts as recent agreement.
    assertNull(w.estimate(t + 2000, allowLoose = false))
  }

  @Test
  fun circularMeanAcrossPlusMinusPi() {
    val poses = listOf(Pose4(0.0, 0.0, 0.0, PI - 0.01), Pose4(0.0, 0.0, 0.0, -PI + 0.01))
    val m = PoseStats.mean(poses)
    assertEquals(0.0, Pose4.angleDiff(m.yaw, PI), 1e-9)
    val med = PoseStats.median(poses + Pose4(0.0, 0.0, 0.0, PI))
    assertEquals(0.0, Pose4.angleDiff(med.yaw, PI), 1e-9)
  }

  @Test
  fun spotCollectorAveragesTenFrames() {
    val c = SpotCollector()
    val rnd = Random(4)
    repeat(SpotCollector.SPOT_FRAMES - 1) { c.add(noisy(rnd, 0.003, 0.5)) }
    assertTrue(!c.done)
    c.add(noisy(rnd, 0.003, 0.5))
    assertTrue(c.done)
    c.add(Pose4(100.0, 0.0, 0.0, 0.0)) // ignored once done
    val e = c.estimate()!!
    assertEquals(SpotCollector.SPOT_FRAMES, e.samples)
    assertTrue(distance(e.pose.translation, truth.translation) < 0.005)
  }

  @Test
  fun spotHeadingAxisIsHorizontalAtAnyRoll() {
    // Phone facing the horizon and rolled by any angle: display +X and +Y rotate in a vertical plane.
    for (deg in 0..360 step 5) {
      val a = Math.toRadians(deg.toDouble())
      val x = Vec3(Math.cos(a), Math.sin(a), 0.0)
      val y = Vec3(-Math.sin(a), Math.cos(a), 0.0)
      val chosen = if (SpotCollector.chooseAxis(x, y) == "Y") y else x
      assertTrue("roll $deg", chosen.horizontalLength() >= 0.707)
    }
    // Upright uses +X (as before); on its side uses +Y.
    assertEquals("X", SpotCollector.chooseAxis(Vec3(1.0, 0.0, 0.0), Vec3(0.0, 1.0, 0.0)))
    assertEquals("Y", SpotCollector.chooseAxis(Vec3(0.0, 1.0, 0.0), Vec3(-1.0, 0.0, 0.0)))
  }
}
