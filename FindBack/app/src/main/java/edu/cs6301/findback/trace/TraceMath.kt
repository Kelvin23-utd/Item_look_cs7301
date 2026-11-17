package edu.cs6301.findback.trace

import kotlin.math.abs
import kotlin.math.acos
import kotlin.math.min
import kotlin.math.sqrt

/*
 * v0.7 detailed trace: the pure parts of sampling (rate limits, motion, frame statistics, depth
 * statistics). No Android or ARCore imports; unit-tested in TraceMathTest.
 */

/**
 * Keeps one sample per [periodNs] on average, by the sample's own clock. Samples are due on a
 * fixed schedule (one period apart) and a sample up to [slack] of a period early counts as due,
 * so a stream a little faster than the period, or with jitter, is thinned to the period itself
 * (30 fps frames are 33 ms apart: a 10 Hz gate keeps every third one, a 2 Hz gate exactly two a
 * second). After a gap of more than a period (a pause) the schedule restarts at the next sample,
 * without a burst. A clock that steps back by more than two periods restarts the gate.
 */
class RateGate(val periodNs: Long, slack: Double = 0.1) {
  private val tolNs = (periodNs * slack).toLong()
  private var nextNs = 0L
  private var started = false

  /** True when [tNs] is due; the gate then counts it as taken. */
  fun accept(tNs: Long): Boolean {
    if (!started || tNs < nextNs - 2 * periodNs) {
      started = true
      nextNs = tNs + periodNs
      return true
    }
    if (tNs < nextNs - tolNs) return false
    nextNs = if (tNs - nextNs >= periodNs) tNs + periodNs else nextNs + periodNs
    return true
  }

  fun reset() {
    started = false
  }

  companion object {
    fun hz(rate: Double): RateGate = RateGate((1e9 / rate).toLong())
  }
}

/** Speed and turn rate between two camera poses (section 4.1). */
object Motion {
  /** Straight-line speed in m/s between two positions [dtNs] apart; NaN when dt is not positive. */
  fun speed(x0: Double, y0: Double, z0: Double, x1: Double, y1: Double, z1: Double, dtNs: Long): Double {
    if (dtNs <= 0) return Double.NaN
    val dx = x1 - x0
    val dy = y1 - y0
    val dz = z1 - z0
    return sqrt(dx * dx + dy * dy + dz * dz) / (dtNs / 1e9)
  }

  /**
   * Angle of the rotation between two unit quaternions (x, y, z, w), in degrees from 0 to 180.
   * q and -q are the same rotation, hence the absolute value of the dot product.
   */
  fun angleDeg(ax: Double, ay: Double, az: Double, aw: Double, bx: Double, by: Double, bz: Double, bw: Double): Double {
    val na = sqrt(ax * ax + ay * ay + az * az + aw * aw)
    val nb = sqrt(bx * bx + by * by + bz * bz + bw * bw)
    if (na == 0.0 || nb == 0.0) return Double.NaN
    val dot = abs(ax * bx + ay * by + az * bz + aw * bw) / (na * nb)
    return Math.toDegrees(2.0 * acos(min(1.0, dot)))
  }

  /** Turn rate in degrees per second between two orientations [dtNs] apart; NaN when dt is not positive. */
  fun turnRate(ax: Double, ay: Double, az: Double, aw: Double, bx: Double, by: Double, bz: Double, bw: Double, dtNs: Long): Double {
    if (dtNs <= 0) return Double.NaN
    return angleDeg(ax, ay, az, aw, bx, by, bz, bw) / (dtNs / 1e9)
  }
}

/** Percentile of the first [n] values of a sorted array, interpolated linearly (as numpy's default). */
internal fun percentileSorted(a: FloatArray, n: Int, p: Double): Double {
  if (n <= 0) return Double.NaN
  if (n == 1) return a[0].toDouble()
  val pos = p / 100.0 * (n - 1)
  val lo = pos.toInt().coerceIn(0, n - 1)
  val hi = min(lo + 1, n - 1)
  val f = pos - lo
  return a[lo] + (a[hi] - a[lo]) * f
}

internal fun percentileSorted(a: IntArray, n: Int, p: Double): Double {
  if (n <= 0) return Double.NaN
  if (n == 1) return a[0].toDouble()
  val pos = p / 100.0 * (n - 1)
  val lo = pos.toInt().coerceIn(0, n - 1)
  val hi = min(lo + 1, n - 1)
  val f = pos - lo
  return a[lo] + (a[hi] - a[lo]) * f
}

/**
 * GL frame times over one `perf` window (section 3): count, mean, 95th percentile and maximum of
 * the time inside the frame callback, the share of frames with a capture being analysed, and the
 * time the trace itself took on the GL thread. Holds up to [capacity] frame times for the
 * percentile (10 s at 60 fps is 600); frames past that still count in the mean and the maximum.
 */
class FrameStats(private val capacity: Int = 2048) {
  private val ms = FloatArray(capacity)
  private val sorted = FloatArray(capacity)
  private var stored = 0

  var frames = 0
    private set
  private var sumMs = 0.0
  var maxMs = 0.0
    private set
  private var busyFrames = 0
  private var traceSumMs = 0.0
  var traceMaxMs = 0.0
    private set

  fun add(frameMs: Double, busy: Boolean, traceMs: Double) {
    frames++
    sumMs += frameMs
    if (frameMs > maxMs) maxMs = frameMs
    if (busy) busyFrames++
    traceSumMs += traceMs
    if (traceMs > traceMaxMs) traceMaxMs = traceMs
    if (stored < capacity) ms[stored++] = frameMs.toFloat()
  }

  val meanMs: Double
    get() = if (frames == 0) Double.NaN else sumMs / frames

  val busyShare: Double
    get() = if (frames == 0) Double.NaN else busyFrames.toDouble() / frames

  val traceMeanMs: Double
    get() = if (frames == 0) Double.NaN else traceSumMs / frames

  fun p95Ms(): Double {
    if (stored == 0) return Double.NaN
    System.arraycopy(ms, 0, sorted, 0, stored)
    java.util.Arrays.sort(sorted, 0, stored)
    return percentileSorted(sorted, stored, 95.0)
  }

  fun reset() {
    frames = 0
    stored = 0
    sumMs = 0.0
    maxMs = 0.0
    busyFrames = 0
    traceSumMs = 0.0
    traceMaxMs = 0.0
  }
}

/** Millimetres at depth pixel (x, y), 0 for no estimate. A fun interface, so no boxing per pixel. */
fun interface MmSource {
  fun mm(x: Int, y: Int): Int
}

/**
 * Statistics of one 16-bit depth image (section 4.3), computed on every [step]th pixel in both
 * directions: the share of pixels with depth, the share within the scan's range, the median, 10th
 * and 90th percentile of the valid depths (metres), and the median of the central 5×5 pixels (all
 * of them, not subsampled). Results are NaN when there is nothing to compute them from. One
 * instance per thread; the buffers are reused.
 */
class DepthStats(
  private val minMm: Int = 200,
  private val maxMm: Int = 4000,
  maxSamples: Int = 8192,
) {
  private var values = IntArray(maxSamples)
  private val centreValues = IntArray(CENTRE * CENTRE)

  var samples = 0
    private set
  var validShare = Double.NaN
    private set
  var usableShare = Double.NaN
    private set
  var medianM = Double.NaN
    private set
  var p10M = Double.NaN
    private set
  var p90M = Double.NaN
    private set
  var centreM = Double.NaN
    private set

  fun compute(width: Int, height: Int, step: Int, src: MmSource) {
    val s = step.coerceAtLeast(1)
    val nx = (width + s - 1) / s
    val ny = (height + s - 1) / s
    val total = nx * ny
    if (total > values.size) values = IntArray(total)
    var valid = 0
    var usable = 0
    var y = 0
    while (y < height) {
      var x = 0
      while (x < width) {
        val v = src.mm(x, y)
        if (v > 0) {
          values[valid++] = v
          if (v in minMm..maxMm) usable++
        }
        x += s
      }
      y += s
    }
    samples = total
    if (total == 0) {
      validShare = Double.NaN
      usableShare = Double.NaN
    } else {
      validShare = valid.toDouble() / total
      usableShare = usable.toDouble() / total
    }
    if (valid > 0) {
      java.util.Arrays.sort(values, 0, valid)
      medianM = percentileSorted(values, valid, 50.0) / 1000.0
      p10M = percentileSorted(values, valid, 10.0) / 1000.0
      p90M = percentileSorted(values, valid, 90.0) / 1000.0
    } else {
      medianM = Double.NaN
      p10M = Double.NaN
      p90M = Double.NaN
    }
    // Central 5×5 at full resolution, clipped to the image.
    var n = 0
    val cx = width / 2
    val cy = height / 2
    for (yy in cy - CENTRE / 2..cy + CENTRE / 2) {
      if (yy < 0 || yy >= height) continue
      for (xx in cx - CENTRE / 2..cx + CENTRE / 2) {
        if (xx < 0 || xx >= width) continue
        val v = src.mm(xx, yy)
        if (v > 0) centreValues[n++] = v
      }
    }
    centreM =
      if (n == 0) Double.NaN
      else {
        java.util.Arrays.sort(centreValues, 0, n)
        percentileSorted(centreValues, n, 50.0) / 1000.0
      }
  }

  companion object {
    const val CENTRE = 5
  }
}
