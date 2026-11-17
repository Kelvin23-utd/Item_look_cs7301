package edu.cs6301.findback.scan

import edu.cs6301.findback.map.Vec3
import edu.cs6301.findback.map.distance

/**
 * Plan section 4: when to take a keyframe. Fed with the camera pose on every tracking frame of the
 * GL thread (the only thread that uses it). Pure Kotlin, unit-tested.
 *
 * A keyframe is due when no other is being processed, at least
 * [ScanConfig.KEYFRAME_MIN_INTERVAL_MS] have passed since the last one, the camera is not turning
 * faster than [ScanConfig.MAX_TURN_RATE_DEG_S], and
 *  - scan mode: it is the scan's first keyframe, or the camera has moved
 *    [ScanConfig.KEYFRAME_MOVE_M] or turned [ScanConfig.KEYFRAME_TURN_DEG] since the last one;
 *  - passive mode: [ScanConfig.PASSIVE_INTERVAL_MS] have passed since the last one.
 */
class KeyframePacer {
  enum class Mode { SCAN, PASSIVE }

  private var lastFramePose: CameraPose? = null
  private var lastFrameMs = 0L
  private var lastKeyframePose: CameraPose? = null
  private var lastKeyframeMs: Long? = null
  private var retryAfterMs = 0L

  /** Turn rate between the last two frames, degrees per second (0 until two frames were seen). */
  var turnRateDegS = 0.0
    private set

  /**
   * Call once per tracking frame with the camera pose. [scanning]: a scan is running; otherwise a
   * passive keyframe is considered when [passiveAllowed]. [busy]: a keyframe is being processed, or
   * something with priority is pending. Returns the kind of keyframe to take now, or null. The
   * caller answers with [taken] or [failed].
   */
  fun onFrame(nowMs: Long, pose: CameraPose, scanning: Boolean, passiveAllowed: Boolean, busy: Boolean): Mode? {
    val prev = lastFramePose
    val dt = nowMs - lastFrameMs
    // A long gap between frames (tracking was lost, or the app was paused) says nothing about speed.
    turnRateDegS = if (prev != null && dt in 1..500) prev.angleTo(pose) * 1000.0 / dt else 0.0
    lastFramePose = pose
    lastFrameMs = nowMs

    if (busy || nowMs < retryAfterMs) return null
    if (!scanning && !passiveAllowed) return null
    val lastMs = lastKeyframeMs
    if (lastMs != null && nowMs - lastMs < ScanConfig.KEYFRAME_MIN_INTERVAL_MS) return null
    if (turnRateDegS > ScanConfig.MAX_TURN_RATE_DEG_S) return null
    if (scanning) {
      val last = lastKeyframePose ?: return Mode.SCAN
      val moved = distance(last.position, pose.position) >= ScanConfig.KEYFRAME_MOVE_M
      val turned = last.angleTo(pose) >= ScanConfig.KEYFRAME_TURN_DEG
      return if (moved || turned) Mode.SCAN else null
    }
    return if (lastMs == null || nowMs - lastMs >= ScanConfig.PASSIVE_INTERVAL_MS) Mode.PASSIVE else null
  }

  /** A keyframe was captured at [pose]. */
  fun taken(nowMs: Long, pose: CameraPose) {
    lastKeyframeMs = nowMs
    lastKeyframePose = pose
  }

  /** The capture did not happen (no depth yet, nothing in view): ask again a little later. */
  fun failed(nowMs: Long) {
    retryAfterMs = nowMs + ScanConfig.CAPTURE_RETRY_MS
  }

  /** A scan starts: its first keyframe is taken at once, wherever the last passive one was. */
  fun scanStarted() {
    lastKeyframePose = null
    retryAfterMs = 0L
  }

  /** Camera position of the last keyframe (for tests and logs). */
  fun lastKeyframePosition(): Vec3? = lastKeyframePose?.position
}
