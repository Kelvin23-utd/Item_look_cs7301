package edu.cs6301.findback.scan

import edu.cs6301.findback.map.Vec3
import edu.cs6301.findback.map.distance
import java.util.Locale
import kotlin.math.max

/** One raw detector output: a rectangle in CPU (sensor) image pixels, a label and a score. */
data class SceneDetection(val box: Box, val label: String, val score: Float)

/**
 * A raw detection after the keyframe filter: kept ([reason] null) with its place in the map, or
 * dropped for [reason]. Every raw detection gets a report, so the saved keyframe lists them all.
 */
class DetectionReport(val detection: SceneDetection) {
  /** Null when kept, else one of the reasons in [DetectionFilter]. */
  var reason: String? = null
  /** Median depth of the central part of the box (metres along the optical axis). */
  var depthM: Double? = null
  var worldPoint: Vec3? = null
  var mapPoint: Vec3? = null
  /** Straight distance from the camera to the point (what verification rescales crops with). */
  var distanceM: Double? = null
  /** Physical length of the longer box side at [depthM]. */
  var sizeM: Double? = null

  val kept: Boolean
    get() = reason == null
}

/**
 * Plan section 3 and 5: which detections of a keyframe are kept, and where they are in the map.
 * Pure Kotlin, unit-tested.
 */
object DetectionFilter {
  const val PERSON = "person"
  const val LOW_SCORE = "low_score"
  const val NO_LABEL = "no_label"
  const val STOP_LIST = "stop_list"
  const val ON_PERSON = "on_person"
  const val DUPLICATE = "duplicate"
  const val NO_DEPTH = "no_depth"
  const val DEPTH_RANGE = "depth_range"
  const val TOO_LARGE = "too_large"
  const val OVER_CAP = "over_cap"
  const val EMPTY_BOX = "empty_box"

  fun normalise(label: String): String = label.trim().lowercase(Locale.ROOT)

  /** True when any raw detection is a person: such a keyframe's image never reaches the disk. */
  fun hasPerson(reports: List<DetectionReport>): Boolean = reports.any { it.reason == PERSON }

  /**
   * One report per raw detection, in the order given. Order of the rules: person, score, label,
   * stop list, on a person, duplicate, depth, size, and last the cap on kept detections.
   */
  fun apply(raw: List<SceneDetection>, geom: KeyframeGeometry): List<DetectionReport> {
    val reports = raw.map { DetectionReport(it.copy(label = normalise(it.label))) }
    val persons = reports.filter { it.detection.label == ScanConfig.PERSON_LABEL }.map { it.detection.box }
    for (r in reports) {
      val d = r.detection
      r.reason =
        when {
          d.label == ScanConfig.PERSON_LABEL -> PERSON
          d.score < ScanConfig.MIN_SCORE -> LOW_SCORE
          d.label.isEmpty() || d.label == "???" -> NO_LABEL
          d.label in ScanConfig.STOP_LABELS -> STOP_LIST
          d.box.area <= 0f -> EMPTY_BOX
          persons.any { p -> d.box.intersectionArea(p) / d.box.area > ScanConfig.ON_PERSON_OVERLAP } -> ON_PERSON
          else -> null
        }
    }
    // Within one keyframe, the same label with strongly overlapping boxes is one detection.
    val winners = ArrayList<DetectionReport>()
    for (r in reports.filter { it.kept }.sortedByDescending { it.detection.score }) {
      val twin =
        winners.any { w ->
          w.detection.label == r.detection.label && w.detection.box.iou(r.detection.box) > ScanConfig.DUPLICATE_IOU
        }
      if (twin) r.reason = DUPLICATE else winners.add(r)
    }
    for (r in winners) {
      val b = r.detection.box
      val d = geom.depth.medianInBox(b.central(ScanConfig.BOX_CENTRAL_FRACTION))
      if (d == null) {
        r.reason = NO_DEPTH
        continue
      }
      r.depthM = d
      if (d < ScanConfig.DEPTH_MIN_M || d > ScanConfig.DEPTH_MAX_M) {
        r.reason = DEPTH_RANGE
        continue
      }
      val world = geom.worldPoint(b.centerX.toDouble(), b.centerY.toDouble(), d)
      r.worldPoint = world
      r.mapPoint = geom.worldToMapPoint(world)
      r.distanceM = distance(world, geom.cameraPose.position)
      // The image is in sensor orientation (its width is the height of an upright phone's view),
      // so the size is the longer side, as tap registration measures it.
      val size = max(b.width, b.height) * d / geom.intrinsics.fx
      r.sizeM = size
      if (size > ScanConfig.MAX_OBJECT_M) r.reason = TOO_LARGE
    }
    // `winners` is in descending score order.
    winners.filter { it.kept }.drop(ScanConfig.MAX_PER_KEYFRAME).forEach { it.reason = OVER_CAP }
    return reports
  }
}

/**
 * A kept detection with its embedding: what merging works on.
 */
class Observation(
  val label: String,
  val score: Float,
  val box: Box,
  val mapPoint: Vec3,
  val worldPoint: Vec3,
  val depthM: Double,
  val distanceM: Double,
  val sizeM: Double,
  val embedding: FloatArray,
  /** Size of the crop the embedding was made from (the padded box, clamped to the image), pixels. */
  val cropWidthPx: Float,
  val cropHeightPx: Float,
  /** The crop image itself (a Bitmap on the device); kept only while it is an entry's best crop. */
  val crop: Any? = null,
  /** v0.8: the study log's observation id (`<scanId>-<kf>-<j>`), or null when study logging is off. */
  val obsId: String? = null,
) {
  /** Crop quality: box area × score. The best one becomes the entry's photo and embedding. */
  val quality: Float
    get() = box.area * score
}
