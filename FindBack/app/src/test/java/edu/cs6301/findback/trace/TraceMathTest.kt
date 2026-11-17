package edu.cs6301.findback.trace

import kotlin.math.cos
import kotlin.math.sin
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

/** v0.7 plan section 6, item 1: rate limiters, depth statistics, speed and turn rate. */
class TraceMathTest {
  private val ms = 1_000_000L

  /** How many of [n] events [stepNs] apart a gate keeps. */
  private fun kept(gate: RateGate, n: Int, stepNs: Long, startNs: Long = 0L): Int = (0 until n).count { gate.accept(startNs + it * stepNs) }

  @Test
  fun gateThinsFramesToItsRate() {
    // 30 fps frames for 10 s: the 10 Hz pose gate keeps every third frame (100 rows).
    assertEquals(100, kept(RateGate.hz(10.0), 300, 33_333_333L))
    // 5 Hz guidance: every sixth frame.
    assertEquals(50, kept(RateGate.hz(5.0), 300, 33_333_333L))
    // 2 Hz depth: two a second. Frames 0 to 9.967 s: the schedule is 0, 0.5, ..., 10 s, and the
    // frame at 9.967 s is within the 10% slack of the 10 s sample, so 21.
    assertEquals(21, kept(RateGate.hz(2.0), 300, 33_333_333L))
    assertEquals(20, kept(RateGate.hz(2.0), 298, 33_333_333L)) // frames up to 9.9 s
    // At 60 fps the pose gate still gives 10 Hz.
    assertEquals(100, kept(RateGate.hz(10.0), 600, 16_666_667L))
  }

  @Test
  fun gateThinsFastSensorStreams() {
    // A 200 Hz accelerometer stream for 1 s, asked for 50 Hz: 50 rows.
    assertEquals(50, kept(RateGate(20 * ms), 200, 5 * ms))
    // A stream a little faster than the rate (19.5 ms apart, 1.93 s) is thinned to 50 Hz, not
    // to every other event: 97 rows.
    assertEquals(97, kept(RateGate(20 * ms), 100, 19_500_000L))
    // A slower stream (45 Hz) is kept whole.
    assertEquals(100, kept(RateGate(20 * ms), 100, 22_222_222L))
    // Light "on change, at most 5 Hz": 10 s of changes 10 ms apart keep one per 200 ms (the
    // first at 0, then 0.18, 0.38, ..., 9.98 s: 51).
    assertEquals(51, kept(RateGate(200 * ms), 1000, 10 * ms))
  }

  @Test
  fun gateDoesNotBurstAfterAPause() {
    val g = RateGate(100 * ms)
    assertEquals(10, kept(g, 30, 33_333_333L)) // 1 s at 30 fps
    // 5 s without frames, then 1 s more: 10 rows again, not a catch-up burst.
    assertEquals(10, kept(g, 30, 33_333_333L, startNs = 6_000 * ms))
  }

  @Test
  fun gateRestartsWhenTheClockStepsBackAndOnReset() {
    val g = RateGate(100 * ms)
    assertTrue(g.accept(1_000 * ms))
    assertFalse(g.accept(1_050 * ms))
    assertTrue(g.accept(500 * ms)) // clock stepped back: accepted, and the gate restarts there
    assertFalse(g.accept(550 * ms))
    g.reset()
    assertTrue(g.accept(560 * ms))
  }

  @Test
  fun speedAndTurnRateFromTwoPoses() {
    // 0.3 m sideways and 0.4 m forward in 0.5 s: 1.0 m/s.
    assertEquals(1.0, Motion.speed(0.0, 1.4, 0.0, 0.3, 1.4, -0.4, 500 * ms), 1e-12)
    // A 45° yaw turn (about +Y) in 0.5 s: 90°/s.
    val h = Math.toRadians(45.0) / 2
    val turn = Motion.turnRate(0.0, 0.0, 0.0, 1.0, 0.0, sin(h), 0.0, cos(h), 500 * ms)
    assertEquals(90.0, turn, 1e-9)
    // q and -q are the same orientation: no turn.
    assertEquals(0.0, Motion.angleDeg(0.0, sin(h), 0.0, cos(h), 0.0, -sin(h), 0.0, -cos(h)), 1e-6)
    // A 180° turn, the largest angle.
    assertEquals(180.0, Motion.angleDeg(0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0), 1e-9)
    // No time between the samples: unknown.
    assertTrue(Motion.speed(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0L).isNaN())
    assertTrue(Motion.turnRate(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, -1L).isNaN())
  }

  /** A [w]×[h] depth image in millimetres from [f]. */
  private fun image(w: Int, h: Int, f: (Int, Int) -> Int): Pair<IntArray, MmSource> {
    val a = IntArray(w * h) { f(it % w, it / w) }
    return a to MmSource { x, y -> a[y * w + x] }
  }

  @Test
  fun depthStatisticsOnASyntheticImage() {
    // 160×120: the left half has no depth, the right half is a ramp of 1000..4950 mm by column;
    // the centre 5×5 is 1500 mm except one hole.
    val w = 160
    val h = 120
    val (_, src) =
      image(w, h) { x, y ->
        when {
          x in 78..82 && y in 58..62 -> if (x == 80 && y == 60) 0 else 1500
          x < 80 -> 0
          else -> 1000 + (x - 80) * 50
        }
      }
    val d = DepthStats(minMm = 200, maxMm = 4000)
    d.compute(w, h, 4, src)
    // Every 4th pixel: 40 × 30 samples; columns 80, 84, ..., 156 (20 of 40) are valid.
    assertEquals(1200, d.samples)
    // Sampled columns ≥ 80 hit the centre patch only at (80, 60): the hole. So exactly half valid.
    val validCols = (80 until 160 step 4).toList()
    val sampledRows = (0 until 120 step 4).toList()
    val holes = if (80 in validCols && 60 in sampledRows) 1 else 0
    val patch = sampledRows.count { it in 58..62 } * validCols.count { it in 78..82 } - holes
    assertEquals((validCols.size * sampledRows.size - holes).toDouble() / 1200, d.validShare, 1e-12)
    // Values: 1000 + (x-80)·50 for x = 80..156, i.e. 1000..4800; usable ≤ 4000 is x ≤ 140.
    val values = ArrayList<Int>()
    for (y in sampledRows) for (x in validCols) {
      val v = if (x in 78..82 && y in 58..62) (if (x == 80 && y == 60) 0 else 1500) else 1000 + (x - 80) * 50
      if (v > 0) values.add(v)
    }
    assertEquals(values.size, validCols.size * sampledRows.size - holes)
    assertEquals(0, patch) // the patch rows sampled (60) only meet column 80, the hole
    assertEquals(values.count { it in 200..4000 }.toDouble() / 1200, d.usableShare, 1e-12)
    val sorted = values.sorted()
    fun pct(p: Double): Double {
      val pos = p / 100 * (sorted.size - 1)
      val lo = pos.toInt()
      val hi = minOf(lo + 1, sorted.size - 1)
      return (sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo)) / 1000.0
    }
    assertEquals(pct(50.0), d.medianM, 1e-9)
    assertEquals(pct(10.0), d.p10M, 1e-9)
    assertEquals(pct(90.0), d.p90M, 1e-9)
    // Centre: full resolution, 24 valid pixels of 1500 mm around the hole.
    assertEquals(1.5, d.centreM, 1e-12)
  }

  @Test
  fun depthStatisticsWithoutAnyDepth() {
    val (_, src) = image(16, 12) { _, _ -> 0 }
    val d = DepthStats()
    d.compute(16, 12, 4, src)
    assertEquals(12, d.samples)
    assertEquals(0.0, d.validShare, 0.0)
    assertEquals(0.0, d.usableShare, 0.0)
    assertTrue(d.medianM.isNaN())
    assertTrue(d.p10M.isNaN())
    assertTrue(d.p90M.isNaN())
    assertTrue(d.centreM.isNaN())
  }

  @Test
  fun depthStatisticsCountOnlyTheScanRangeAsUsable() {
    // Quarter each: 100 mm (too near), 1000 mm, 3000 mm, 6000 mm (too far).
    val (_, src) = image(8, 8) { x, _ -> intArrayOf(100, 1000, 3000, 6000)[x / 2] }
    val d = DepthStats(minMm = 200, maxMm = 4000)
    d.compute(8, 8, 1, src)
    assertEquals(1.0, d.validShare, 0.0)
    assertEquals(0.5, d.usableShare, 0.0)
    assertEquals(2.0, d.medianM, 1e-12) // between 1000 and 3000
  }

  @Test
  fun frameStatisticsOverAWindow() {
    val s = FrameStats(capacity = 10)
    for (i in 1..20) s.add(i.toDouble(), busy = i % 4 == 0, traceMs = 0.1)
    assertEquals(20, s.frames)
    assertEquals(10.5, s.meanMs, 1e-12)
    assertEquals(20.0, s.maxMs, 0.0)
    assertEquals(0.25, s.busyShare, 1e-12)
    assertEquals(0.1, s.traceMeanMs, 1e-12)
    // The percentile uses the first `capacity` frame times only: 1..10.
    assertEquals(9.55, s.p95Ms(), 1e-6)
    s.reset()
    assertEquals(0, s.frames)
    assertTrue(s.meanMs.isNaN())
    assertTrue(s.p95Ms().isNaN())
  }
}
