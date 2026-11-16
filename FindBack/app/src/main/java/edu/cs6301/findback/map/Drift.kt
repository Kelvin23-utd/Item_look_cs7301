package edu.cs6301.findback.map

/** Drift proxies, all known before any fresh capture. Reset at every applied fix. */
data class DriftProxies(
  val fixPointId: String,
  val fixType: FixType,
  /** Camera path length since the fix (world frame, metres). */
  val metersSinceFix: Double,
  /** Tracking pauses of [TrackingLossDetector.LOSS_MIN_MS] or longer since the fix. */
  val trackingLossesSinceFix: Int,
  val msSinceFix: Long,
) {
  fun toMap(): Map<String, Any?> =
    mapOf(
      "fixPoint" to fixPointId,
      "fixType" to fixType.name,
      "metersSinceFix" to metersSinceFix,
      "trackingLossesSinceFix" to trackingLossesSinceFix,
      "msSinceFix" to msSinceFix,
    )

  companion object {
    /** The same keys with null values, for events logged while not located. */
    fun emptyMap(): Map<String, Any?> =
      mapOf(
        "fixPoint" to null,
        "fixType" to null,
        "metersSinceFix" to null,
        "trackingLossesSinceFix" to null,
        "msSinceFix" to null,
      )
  }
}

/** Accumulates the drift proxies between fixes. GL thread only. */
class DriftTracker {
  companion object {
    /** Path length grows in steps of at least this, so standing still adds no jitter. */
    const val PATH_STEP_M = 0.05
    /** A single step longer than this is a relocalisation jump, not walking. */
    const val MAX_STEP_M = 0.5
  }

  private var fixAtMs: Long? = null
  private var fixPointId: String? = null
  private var fixType: FixType? = null
  private var meters = 0.0
  private var losses = 0
  private var lastPoint: Vec3? = null

  /** Start a new fix; returns the proxies of the previous fix (null if none this session). */
  fun onFix(nowMs: Long, fixPointId: String, fixType: FixType): DriftProxies? {
    val prev = proxies(nowMs)
    fixAtMs = nowMs
    this.fixPointId = fixPointId
    this.fixType = fixType
    meters = 0.0
    losses = 0
    return prev
  }

  /** Location lost (anchor stopped or new map): no proxies until the next fix. */
  fun clear() {
    fixAtMs = null
    fixPointId = null
    fixType = null
  }

  /** Camera position (world frame) on a tracking frame. */
  fun onPosition(p: Vec3) {
    val last = lastPoint
    if (last == null) {
      lastPoint = p
      return
    }
    val d = distance(last, p)
    if (d >= PATH_STEP_M) {
      if (d <= MAX_STEP_M) meters += d
      lastPoint = p
    }
  }

  /** A counted tracking loss (paused for 1 s or longer). */
  fun onTrackingLoss() {
    losses++
    lastPoint = null
  }

  /** Any non-tracking frame: do not count the jump when tracking resumes. */
  fun breakPath() {
    lastPoint = null
  }

  fun proxies(nowMs: Long): DriftProxies? {
    val t = fixAtMs ?: return null
    return DriftProxies(fixPointId!!, fixType!!, meters, losses, nowMs - t)
  }
}

/**
 * Turns per-frame tracking states into tracking-loss events. A pause counts as a loss once it lasts
 * [LOSS_MIN_MS]; the start-up phase before the first tracking frame is not a loss.
 */
class TrackingLossDetector(private val minMs: Long = LOSS_MIN_MS) {
  companion object {
    /** A tracking pause at least this long counts as a loss (drift proxy, chain break). */
    const val LOSS_MIN_MS = 1000L
  }

  sealed class Event {
    data class Lost(val reason: String, val afterMs: Long) : Event()

    data class Resumed(val reason: String, val durationMs: Long) : Event()
  }

  private var everTracked = false
  private var pausedSinceMs: Long? = null
  private var reason = "NONE"
  private var counted = false

  /** True while a counted loss is in progress. */
  val inLoss: Boolean
    get() = counted

  fun update(tracking: Boolean, nowMs: Long, failureReason: String): Event? {
    if (tracking) {
      everTracked = true
      val since = pausedSinceMs ?: return null
      pausedSinceMs = null
      val ev = if (counted) Event.Resumed(reason, nowMs - since) else null
      counted = false
      return ev
    }
    if (!everTracked) return null
    val since = pausedSinceMs
    if (since == null) {
      pausedSinceMs = nowMs
      reason = failureReason
      counted = false
      return null
    }
    if (reason == "NONE" && failureReason != "NONE") reason = failureReason
    if (!counted && nowMs - since >= minMs) {
      counted = true
      return Event.Lost(reason, nowMs - since)
    }
    return null
  }
}
