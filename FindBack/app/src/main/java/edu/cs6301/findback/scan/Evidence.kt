package edu.cs6301.findback.scan

import edu.cs6301.findback.map.Vec3
import edu.cs6301.findback.map.distance
import kotlin.math.max

/** What one keyframe can say about the place where an entry should be. */
enum class Visibility(val code: String) {
  /** The place is not in this image (outside, too near, too far or behind the camera). */
  OUT_OF_VIEW("out"),
  /** In the image, but the depth there is clearly nearer: something is in front. Says nothing. */
  HIDDEN("hidden"),
  /** In the image, but there is no depth estimate there. Says nothing. */
  NO_DEPTH("no_depth"),
  /** In the image and not covered: the camera is looking at the place itself. */
  OBSERVABLE("observable"),
}

/** One entry's place as seen from one keyframe. */
class ViewCheck(
  val visibility: Visibility,
  /** Pixel of the entry's position in the CPU image. */
  val u: Double,
  val v: Double,
  /** The entry's distance along the optical axis (metres). */
  val zM: Double,
  /** Straight distance from the camera to the entry (metres). */
  val distanceM: Double,
  /** Depth measured at that pixel (metres), or null. */
  val measuredM: Double?,
)

/** What a scan, or a passive window, has gathered about one entry. */
class Tally {
  /** Keyframes in which a detection matched the entry. */
  var matchedKeyframes = 0
  /** Keyframes in which the entry's place was in view and no detection matched it. */
  var inViewKeyframes = 0
  var hiddenKeyframes = 0
  var noDepthKeyframes = 0
  /** Keyframes in which the place was observable. */
  var observableKeyframes = 0
  /** Observable keyframes in which the appearance check ran. Only these count towards `missed`. */
  var checkedKeyframes = 0
  /** Best appearance similarity of the checks, or null when none ran. */
  var bestSimilarity: Float? = null
  var appearanceSeen = false
  /** When the entry was last seen in this scan or window (epoch ms). */
  var lastSeenMs: Long? = null

  val seen: Boolean
    get() = matchedKeyframes > 0 || appearanceSeen
}

/**
 * Plan section 7: visibility-aware evidence. Absence only counts when the place was actually
 * visible and looked at. Pure Kotlin, unit-tested.
 */
object EvidenceRules {
  /** Something nearer than the entry by more than this is in front of it. */
  fun hiddenMargin(zM: Double): Double = max(ScanConfig.HIDDEN_MIN_M, ScanConfig.HIDDEN_FRACTION * zM)

  /** How the place at map position [position] appears in the keyframe [geom]. */
  fun classify(position: Vec3, geom: KeyframeGeometry): ViewCheck {
    val p = geom.projectMap(position) ?: return ViewCheck(Visibility.OUT_OF_VIEW, 0.0, 0.0, 0.0, 0.0, null)
    val dist = distance(geom.mapToWorld.transformPoint(position), geom.cameraPose.position)
    val inView =
      geom.inImage(p, ScanConfig.VIEW_MARGIN) && p.depthM >= ScanConfig.VIEW_MIN_M && p.depthM <= ScanConfig.VIEW_MAX_M
    if (!inView) return ViewCheck(Visibility.OUT_OF_VIEW, p.u, p.v, p.depthM, dist, null)
    val m = geom.depth.medianPatch(p.u, p.v) ?: return ViewCheck(Visibility.NO_DEPTH, p.u, p.v, p.depthM, dist, null)
    val vis = if (m < p.depthM - hiddenMargin(p.depthM)) Visibility.HIDDEN else Visibility.OBSERVABLE
    return ViewCheck(vis, p.u, p.v, p.depthM, dist, m)
  }

  /**
   * The one outcome of a scan or passive window for an entry. [eligible] is false for an entry in
   * another container than the one being scanned: it is never looked for.
   */
  fun outcome(tally: Tally?, eligible: Boolean): String =
    when {
      !eligible || tally == null -> ScanConfig.OUTCOME_NOT_OBSERVED
      tally.seen -> ScanConfig.OUTCOME_SEEN
      tally.checkedKeyframes >= ScanConfig.MISSED_MIN_KEYFRAMES -> ScanConfig.OUTCOME_MISSED
      else -> ScanConfig.OUTCOME_NOT_OBSERVED
    }

  /**
   * Which of the observable entries get this keyframe's appearance checks: at most [cap], those
   * checked least recently first (never checked before all), nearer ones first among equals.
   * [lastChecked] maps an entry id to the running number of its last check.
   */
  fun <T> pickForAppearance(
    candidates: List<T>,
    idOf: (T) -> String,
    distanceOf: (T) -> Double,
    lastChecked: Map<String, Int>,
    cap: Int = ScanConfig.APPEARANCE_CHECKS_PER_KEYFRAME,
  ): List<T> =
    candidates.sortedWith(compareBy<T>({ lastChecked[idOf(it)] ?: -1 }, { distanceOf(it) })).take(cap)
}
