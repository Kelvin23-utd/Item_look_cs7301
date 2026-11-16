package edu.cs6301.findback.policy

/** Where a remembered position came from in the current session. */
enum class PositionSource {
  /** Same-session ARCore anchor. */
  SESSION,
  /**
   * Stored map-frame position, placed through the located home map (v0.3). Replaces v0.2's
   * REFERENCE (offset from the reference image); old logs still contain "REFERENCE".
   */
  MAP,
  /** No usable spatial cue: show the historical photo only. */
  NONE,
}

/**
 * Signals available *before* any fresh capture. The trigger must not depend on a fresh image,
 * otherwise "selective" would cost as much as "always verify".
 */
data class PolicyInputs(
  val ageMinutes: Double,
  /** Highest similarity of the target's stored embedding to any other record (distractor risk). */
  val maxSimilarityToOthers: Float,
  val positionSource: PositionSource,
  /** Reference image currently under full tracking (not just last-known pose). Logged only. */
  val referenceFullTracking: Boolean,
  /** Camera path length since the last map fix (metres); null when not located. */
  val metersSinceFix: Double? = null,
  /** Tracking pauses of 1 s or longer since the last map fix; null when not located. */
  val trackingLossesSinceFix: Int? = null,
  val msSinceFix: Long? = null,
  /** "IMAGE" or "SPOT": how the map was last fixed; null when not located. */
  val fixType: String? = null,
)

data class PolicyDecision(val verify: Boolean, val reasons: List<String>) {
  fun summary() = (if (verify) "VERIFY" else "SKIP") + ": " + reasons.joinToString("; ")
}

/**
 * Initial heuristic for selective verification, as stated in the proposal. Thresholds are
 * placeholders to be chosen on the development session and frozen before the test session.
 */
object VerificationPolicy {
  const val AGE_MINUTES = 30.0
  const val DISTRACTOR_SIMILARITY = 0.80f
  /** A MAP position is drift-prone after the camera has walked this far since the last fix. */
  const val DRIFT_METERS = 20.0

  /** Fresh-capture similarity at or above this counts as "seen here now". Calibrate on dev data. */
  const val MATCH_THRESHOLD = 0.75f

  fun decide(i: PolicyInputs): PolicyDecision {
    if (i.positionSource == PositionSource.NONE) {
      return PolicyDecision(false, listOf("no spatial cue; historical photo only"))
    }
    val reasons = mutableListOf<String>()
    if (i.ageMinutes > AGE_MINUTES) reasons += "age %.0f min > %.0f".format(i.ageMinutes, AGE_MINUTES)
    if (i.maxSimilarityToOthers > DISTRACTOR_SIMILARITY) {
      reasons += "similar record %.2f > %.2f".format(i.maxSimilarityToOthers, DISTRACTOR_SIMILARITY)
    }
    // v0.3: replaces v0.2's "reference not fully tracked" trigger.
    if (i.positionSource == PositionSource.MAP) {
      val m = i.metersSinceFix ?: 0.0
      val losses = i.trackingLossesSinceFix ?: 0
      if (m > DRIFT_METERS) reasons += "%.0f m walked since map fix > %.0f".format(m, DRIFT_METERS)
      if (losses > 0) reasons += "$losses tracking loss(es) since map fix"
    }
    return if (reasons.isEmpty()) PolicyDecision(false, listOf("recent, distinct, tracking ok"))
    else PolicyDecision(true, reasons)
  }
}
