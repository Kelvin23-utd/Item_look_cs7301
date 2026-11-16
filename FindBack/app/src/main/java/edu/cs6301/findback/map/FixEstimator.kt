package edu.cs6301.findback.map

import kotlin.math.abs
import kotlin.math.atan2
import kotlin.math.cos
import kotlin.math.max
import kotlin.math.sin

/** One gated observation of a fix point in the world frame. */
data class FixSample(val tMs: Long, val pose: Pose4, val cameraDistanceM: Double)

/** An accepted fix estimate in the world frame, with the statistics logged in `map_fix`. */
data class FixEstimate(
  val pose: Pose4,
  val samples: Int,
  /** Largest distance of a used sample from [pose] (metres). */
  val posSpreadM: Double,
  /** Largest yaw difference of a used sample from [pose] (degrees). */
  val yawSpreadDeg: Double,
  val cameraDistanceM: Double,
  /** "good" (converged) or "loose" (median after the timeout). SPOT locks are "good". */
  val quality: String,
)

object PoseStats {
  /** Mean position and circular-mean yaw. */
  fun mean(poses: List<Pose4>): Pose4 {
    require(poses.isNotEmpty())
    var x = 0.0
    var y = 0.0
    var z = 0.0
    var s = 0.0
    var c = 0.0
    for (p in poses) {
      x += p.x
      y += p.y
      z += p.z
      s += sin(p.yaw)
      c += cos(p.yaw)
    }
    val n = poses.size
    return Pose4(x / n, y / n, z / n, atan2(s, c))
  }

  /** Component-wise median position; yaw is the median of differences from the circular mean. */
  fun median(poses: List<Pose4>): Pose4 {
    require(poses.isNotEmpty())
    val ref = mean(poses).yaw
    return Pose4(
      medianOf(poses.map { it.x }),
      medianOf(poses.map { it.y }),
      medianOf(poses.map { it.z }),
      Pose4.wrapAngle(ref + medianOf(poses.map { Pose4.angleDiff(it.yaw, ref) })),
    )
  }

  fun maxPositionDeviation(poses: List<Pose4>, c: Pose4): Double =
    poses.maxOfOrNull { distance(it.translation, c.translation) } ?: 0.0

  fun maxYawDeviationDeg(poses: List<Pose4>, c: Pose4): Double =
    poses.maxOfOrNull { Math.toDegrees(abs(Pose4.angleDiff(it.yaw, c.yaw))) } ?: 0.0

  private fun medianOf(v: List<Double>): Double {
    val s = v.sorted()
    val m = s.size / 2
    return if (s.size % 2 == 1) s[m] else (s[m - 1] + s[m]) / 2
  }
}

/**
 * Sliding window of gated picture observations. Accepts when at least [AGREE_SAMPLES] samples from
 * the last [AGREE_WINDOW_MS] lie within [AGREE_POS_M] and [AGREE_YAW_DEG] of their median, using the
 * mean position and circular-mean yaw of those samples. After [LOOSE_AFTER_MS] of gated tracking
 * without convergence it can accept the window median as a "loose" fix.
 */
class ImageFixWindow {
  companion object {
    /** Sliding window length (samples, one per gated frame). */
    const val WINDOW_SAMPLES = 30
    /** Samples that must agree... */
    const val AGREE_SAMPLES = 20
    /** ...among those from this recent period... */
    const val AGREE_WINDOW_MS = 1500L
    /** ...within this distance of their median... */
    const val AGREE_POS_M = 0.02
    /** ...and this yaw difference. */
    const val AGREE_YAW_DEG = 2.0
    /** Gated tracking time after which the median is accepted as a "loose" fix. */
    const val LOOSE_AFTER_MS = 3000L
    /** Fewest samples a loose fix may use. */
    const val LOOSE_MIN_SAMPLES = 10
    /** Consecutive samples further apart than this do not add to the gated time (gate was off). */
    const val MAX_SAMPLE_GAP_MS = 250L
  }

  private val samples = ArrayDeque<FixSample>()
  private var lastSampleMs: Long? = null

  /** Accumulated time with the gate open since the last [reset]. */
  var gatedMs = 0L
    private set

  /** Image axis chosen for this window ("X" or "Z"), fixed until [reset]. */
  var axis: String? = null

  val size: Int
    get() = samples.size

  fun reset() {
    samples.clear()
    lastSampleMs = null
    gatedMs = 0L
    axis = null
  }

  fun add(s: FixSample) {
    lastSampleMs?.let { last ->
      val gap = s.tMs - last
      if (gap in 0..MAX_SAMPLE_GAP_MS) gatedMs += gap
    }
    lastSampleMs = s.tMs
    samples.addLast(s)
    while (samples.size > WINDOW_SAMPLES) samples.removeFirst()
  }

  /** Converged estimate, or (when [allowLoose]) a loose one after the timeout, else null. */
  fun estimate(nowMs: Long, allowLoose: Boolean): FixEstimate? {
    val recent = samples.filter { nowMs - it.tMs <= AGREE_WINDOW_MS }
    if (recent.size >= AGREE_SAMPLES) {
      val med = PoseStats.median(recent.map { it.pose })
      val agree =
        recent.filter {
          distance(it.pose.translation, med.translation) <= AGREE_POS_M &&
            Math.toDegrees(abs(Pose4.angleDiff(it.pose.yaw, med.yaw))) <= AGREE_YAW_DEG
        }
      if (agree.size >= AGREE_SAMPLES) return summarize(agree, PoseStats.mean(agree.map { it.pose }), "good")
    }
    if (allowLoose && gatedMs >= LOOSE_AFTER_MS && samples.size >= LOOSE_MIN_SAMPLES) {
      val all = samples.toList()
      return summarize(all, PoseStats.median(all.map { it.pose }), "loose")
    }
    return null
  }

  private fun summarize(used: List<FixSample>, pose: Pose4, quality: String): FixEstimate {
    val poses = used.map { it.pose }
    return FixEstimate(
      pose = pose,
      samples = used.size,
      posSpreadM = PoseStats.maxPositionDeviation(poses, pose),
      yawSpreadDeg = PoseStats.maxYawDeviationDeg(poses, pose),
      cameraDistanceM = used.map { it.cameraDistanceM }.average(),
      quality = quality,
    )
  }
}

/** Averages the gated camera pose over [SPOT_FRAMES] frames after "Lock". */
class SpotCollector(private val needed: Int = SPOT_FRAMES) {
  companion object {
    /** Frames averaged per spot lock or spot capture. */
    const val SPOT_FRAMES = 10
    /** On a later lock, the spot's stored heading axis must keep at least this horizontal length. */
    const val SPOT_MIN_AXIS = 0.5

    /**
     * Heading axis for a new spot: display +X when it is at least as horizontal as +Y, else +Y.
     * The two are perpendicular, so the chosen one always has a horizontal length of 0.707 or more.
     */
    fun chooseAxis(xAxis: Vec3, yAxis: Vec3): String =
      if (xAxis.horizontalLength() >= yAxis.horizontalLength()) "X" else "Y"
    /** Give up a lock that has not collected [SPOT_FRAMES] gated frames within this time. */
    const val SPOT_TIMEOUT_MS = 3000L
  }

  private val poses = ArrayList<Pose4>()
  var rejectedRolled = 0
    private set
  var rejectedTracking = 0
    private set

  val done: Boolean
    get() = poses.size >= needed

  fun add(p: Pose4) {
    if (!done) poses.add(p)
  }

  fun rejectRolled() {
    rejectedRolled++
  }

  fun rejectTracking() {
    rejectedTracking++
  }

  fun estimate(): FixEstimate? {
    if (poses.isEmpty()) return null
    val m = PoseStats.mean(poses)
    return FixEstimate(
      pose = m,
      samples = poses.size,
      posSpreadM = PoseStats.maxPositionDeviation(poses, m),
      yawSpreadDeg = PoseStats.maxYawDeviationDeg(poses, m),
      cameraDistanceM = 0.0,
      quality = "good",
    )
  }

  /** Why a lock failed, for the toast. */
  fun failureHint(): String =
    if (rejectedRolled >= max(1, rejectedTracking)) "Hold the phone the way it was when the spot was saved (upright or on its side), then Lock again."
    else "Tracking is unstable: move the phone slowly, then Lock again."
}
