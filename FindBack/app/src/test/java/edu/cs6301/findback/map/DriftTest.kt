package edu.cs6301.findback.map

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class DriftTest {
  @Test
  fun pathLengthIgnoresJitterAndJumps() {
    val d = DriftTracker()
    assertNull(d.proxies(0))
    d.onFix(1000, "fp1", FixType.IMAGE)
    // Standing still with 1 cm jitter for a while adds nothing.
    repeat(300) { i -> d.onPosition(Vec3(0.01 * (i % 2), 1.4, 0.0)) }
    assertEquals(0.0, d.proxies(2000)!!.metersSinceFix, 1e-9)
    // Walk 3 m in 2 cm frames.
    for (i in 0..150) d.onPosition(Vec3(0.02 * i, 1.4, 0.0))
    assertEquals(3.0, d.proxies(3000)!!.metersSinceFix, 0.06)
    // A 2 m relocalisation jump is not walking.
    d.onPosition(Vec3(5.0, 1.4, 0.0))
    assertEquals(3.0, d.proxies(3000)!!.metersSinceFix, 0.06)
  }

  @Test
  fun fixResetsAndReturnsThePreviousProxies() {
    val d = DriftTracker()
    assertNull(d.onFix(0, "a", FixType.IMAGE))
    for (i in 0..50) d.onPosition(Vec3(0.1 * i, 0.0, 0.0))
    d.onTrackingLoss()
    val prev = d.onFix(9000, "b", FixType.SPOT)!!
    assertEquals("a", prev.fixPointId)
    assertEquals(FixType.IMAGE, prev.fixType)
    assertEquals(5.0, prev.metersSinceFix, 1e-6)
    assertEquals(1, prev.trackingLossesSinceFix)
    assertEquals(9000L, prev.msSinceFix)
    val now = d.proxies(9500)!!
    assertEquals("b", now.fixPointId)
    assertEquals(0.0, now.metersSinceFix, 1e-9)
    assertEquals(0, now.trackingLossesSinceFix)
    assertEquals(500L, now.msSinceFix)
    d.clear()
    assertNull(d.proxies(10000))
  }

  @Test
  fun trackingLossCountsAfterOneSecond() {
    val t = TrackingLossDetector()
    // Start-up before the first tracking frame is not a loss.
    assertNull(t.update(false, 0, "NONE"))
    assertNull(t.update(false, 5000, "NONE"))
    assertNull(t.update(true, 5100, "NONE"))
    // A short pause is ignored.
    assertNull(t.update(false, 6000, "EXCESSIVE_MOTION"))
    assertNull(t.update(false, 6500, "EXCESSIVE_MOTION"))
    assertNull(t.update(true, 6600, "NONE"))
    // A 1.5 s pause is one loss, then one resume with its duration.
    assertNull(t.update(false, 7000, "INSUFFICIENT_FEATURES"))
    val lost = t.update(false, 8000, "INSUFFICIENT_FEATURES")
    assertTrue(lost is TrackingLossDetector.Event.Lost)
    assertEquals("INSUFFICIENT_FEATURES", (lost as TrackingLossDetector.Event.Lost).reason)
    assertTrue(t.inLoss)
    assertNull(t.update(false, 8300, "INSUFFICIENT_FEATURES"))
    val resumed = t.update(true, 8500, "NONE")
    assertTrue(resumed is TrackingLossDetector.Event.Resumed)
    assertEquals(1500L, (resumed as TrackingLossDetector.Event.Resumed).durationMs)
  }
}
