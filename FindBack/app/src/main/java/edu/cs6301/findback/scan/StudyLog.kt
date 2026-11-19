package edu.cs6301.findback.scan

import edu.cs6301.findback.map.Pose4
import edu.cs6301.findback.map.Vec3
import edu.cs6301.findback.map.distance
import edu.cs6301.findback.memory.EmbeddingFile
import kotlin.math.acos
import kotlin.math.asin
import kotlin.math.atan2
import kotlin.math.cos
import kotlin.math.floor
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.sin
import org.json.JSONArray
import org.json.JSONObject

/*
 * v0.8 study logging (docs/PLAN_v0.8_study_logging.md): raw observations kept apart from what the
 * baseline matcher decided, a few complementary full-resolution views per place, and the rules that
 * keep people out of them. Pure Kotlin (no Android or ARCore imports), unit-tested. The device side
 * is `StudyRecorder`.
 */

/** Every number and name of study logging, in one place. */
object StudyConfig {
  /** `v` of every line of `obs/` and of the new entry fields. */
  const val SCHEMA = 1

  /** An observation joins the nearest retention group whose anchor is at most this far (map frame). */
  const val GROUP_RADIUS_M = 0.30
  /** Views kept per retention group per scan. */
  const val VIEWS_PER_GROUP = 4
  /** Views kept (on disk at the end) per scan. */
  const val VIEWS_PER_SCAN = 400
  /** Angular spread is compared in steps of this many degrees; within a step, quality decides. */
  const val SPREAD_BIN_DEG = 10.0

  /** Person boxes that block a view: label `person` with at least this score (the detector's own floor). */
  const val PERSON_MIN_SCORE = ScanConfig.RAW_SCORE_MIN
  /** Person boxes are grown by this fraction of their size on every side before the test. */
  const val PERSON_PAD = 0.10f

  /** `decision.candidates`: this many nearest entries the matcher considered. */
  const val DECISION_CANDIDATES = 3

  const val OBS_DIR = "obs"
  const val VIEWS_DIR = "views"
  const val EXTRA_OBS_FILE = "obs/extra_views.jsonl"
  const val EXTRA_VIEWS_DIR = "views/extra"

  const val TYPE_OBS = "obs"
  const val TYPE_VIEW_EVICTED = "view_evicted"
  const val TYPE_SCAN_END = "scan_end"

  const val KIND_SCAN = "scan"
  const val KIND_EXTRA_VIEW = "extra_view"

  const val MATCHER_BASELINE = "baseline"

  // `viewReason`: why an observation has a view file, or why not.
  const val VIEW_KEPT = "kept"
  const val VIEW_PERSON = "person"
  const val VIEW_NOT_COMPLEMENTARY = "not_complementary"
  const val VIEW_SCAN_CAP = "scan_cap"
  const val VIEW_NO_CROP = "no_crop"
  const val VIEW_WRITE_ERROR = "write_error"
  /** Extra views only: the person detector could not run, so no crop is written. */
  const val VIEW_PERSON_CHECK_FAILED = "person_check_failed"

  // History `link` and confirmed `source`.
  const val LINK_CREATED = "created"
  const val LINK_BASELINE_UNCONFIRMED = "baseline_unconfirmed"
  const val SOURCE_CREATED = "created"
  const val SOURCE_USER_VERIFY = "user_verify"

  fun obsFile(scanId: String) = "$OBS_DIR/$scanId.jsonl"

  fun viewPath(scanId: String, obsId: String) = "$VIEWS_DIR/$scanId/$obsId.jpg"

  fun extraViewPath(obsId: String) = "$EXTRA_VIEWS_DIR/$obsId.jpg"

  /** `<scanId>-<kf>-<j>`: keyframe index and the detection's index among that keyframe's kept ones. */
  fun obsId(scanId: String, keyframe: Int, j: Int) = "$scanId-$keyframe-$j"

  /** `<queryId>-x<attempt>` for the capture of one verification attempt. */
  fun extraObsId(queryId: String, attempt: Int) = "$queryId-x$attempt"

  fun configJson(): JSONObject =
    JSONObject().apply {
      put("schema", SCHEMA)
      put("groupRadiusM", GROUP_RADIUS_M)
      put("viewsPerGroup", VIEWS_PER_GROUP)
      put("viewsPerScan", VIEWS_PER_SCAN)
      put("spreadBinDeg", SPREAD_BIN_DEG)
      put("personMinScore", PERSON_MIN_SCORE.toDouble())
      put("personPad", PERSON_PAD.toDouble())
      put("cropPad", ScanConfig.CROP_PAD.toDouble())
      put("jpegQuality", ScanConfig.CROP_JPEG_QUALITY)
      put("decisionCandidates", DECISION_CANDIDATES)
    }
}

/**
 * Rule 3 of the plan: a view is never cut when its rectangle touches a person. Person boxes are the
 * detections labelled `person` with a score of at least [StudyConfig.PERSON_MIN_SCORE], grown by
 * [StudyConfig.PERSON_PAD] on every side. Person boxes themselves are never written anywhere.
 */
object PersonRule {
  /** The padded person boxes of one frame's reports (labels already normalised by the filter). */
  fun boxes(reports: List<DetectionReport>): List<Box> = fromDetections(reports.map { it.detection })

  /** The same for raw detections (labels are normalised here). */
  fun fromDetections(dets: List<SceneDetection>): List<Box> =
    dets
      .filter { DetectionFilter.normalise(it.label) == ScanConfig.PERSON_LABEL && it.score >= StudyConfig.PERSON_MIN_SCORE }
      .map { it.box.padded(StudyConfig.PERSON_PAD) }

  /** True when [rect] has no area in common with any of [persons]. */
  fun allowsView(rect: Box, persons: List<Box>): Boolean = persons.none { rect.intersectionArea(it) > 0f }

  /** Largest share of [rect]'s area inside one padded person box: 0 when none touches it. */
  fun overlap(rect: Box, persons: List<Box>): Double {
    val a = rect.area
    if (a <= 0f) return if (allowsView(rect, persons)) 0.0 else 1.0
    var best = 0.0
    for (p in persons) best = max(best, (rect.intersectionArea(p) / a).toDouble())
    return best
  }
}

/** Image arithmetic the device does with `RectF` and `Bitmap`, restated here so it is testable. */
object StudyImage {
  /** `CameraFrame.clamp`: the rectangle limited to an image of [w] × [h]. */
  fun clamp(b: Box, w: Int, h: Int): Box =
    Box(
      b.left.coerceIn(0f, w - 1f),
      b.top.coerceIn(0f, h - 1f),
      b.right.coerceIn(1f, w.toFloat()),
      b.bottom.coerceIn(1f, h.toFloat()),
    )

  /** The crop rectangle of a detection box: padded by [ScanConfig.CROP_PAD], then clamped. */
  fun viewRect(box: Box, w: Int, h: Int): Box = clamp(box.padded(ScanConfig.CROP_PAD), w, h)

  /**
   * Pixel size (width, height) of `CameraFrame.cropUpright(rect)`: the clamped rectangle truncated
   * to whole pixels, at least 1, then turned upright (width and height swap for 90° and 270°).
   */
  fun uprightCropSize(rect: Box, w: Int, h: Int, rotation: Int): Pair<Int, Int> {
    val c = clamp(rect, w, h)
    val x = c.left.toInt()
    val y = c.top.toInt()
    val cw = max(1, c.width.toInt().coerceAtMost(w - x))
    val ch = max(1, c.height.toInt().coerceAtMost(h - y))
    return if (rotation % 180 == 0) cw to ch else ch to cw
  }
}

/** Camera poses as the study lines write them. */
object StudyGeometry {
  /** Hamilton product of two quaternions (x, y, z, w). */
  fun mul(a: DoubleArray, b: DoubleArray): DoubleArray =
    doubleArrayOf(
      a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1],
      a[3] * b[1] - a[0] * b[2] + a[1] * b[3] + a[2] * b[0],
      a[3] * b[2] + a[0] * b[1] - a[1] * b[0] + a[2] * b[3],
      a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2],
    )

  /** The camera's orientation in the map frame (x, y, z, w): the world orientation turned by −yaw of T_WM. */
  fun mapQuaternion(pose: CameraPose, mapToWorld: Pose4): DoubleArray {
    val h = mapToWorld.yaw / 2
    val worldToMap = doubleArrayOf(0.0, -sin(h), 0.0, cos(h))
    return mul(worldToMap, doubleArrayOf(pose.qx, pose.qy, pose.qz, pose.qw))
  }

  /** The camera pose in the map frame: position and orientation. */
  fun mapPose(pose: CameraPose, mapToWorld: Pose4): CameraPose {
    val q = mapQuaternion(pose, mapToWorld)
    return CameraPose(mapToWorld.inverse().transformPoint(pose.position), q[0], q[1], q[2], q[3])
  }

  /** Unit viewing direction (the camera's −Z axis) in the map frame. */
  fun forwardMap(pose: CameraPose, mapToWorld: Pose4): Vec3 =
    mapPose(pose, mapToWorld).let { it.toWorld(Vec3(0.0, 0.0, -1.0)) - it.position }

  /** Heading of [forward] in degrees, as `pose.csv` `myaw`: atan2(−x, −z). Null when looking straight up or down. */
  fun yawDeg(forward: Vec3): Double? =
    if (hypot(forward.x, forward.z) < 1e-6) null else Math.toDegrees(atan2(-forward.x, -forward.z))

  /** Elevation of [forward] in degrees, + up. */
  fun pitchDeg(forward: Vec3): Double {
    val n = forward.length()
    return if (n < 1e-12) 0.0 else Math.toDegrees(asin((forward.y / n).coerceIn(-1.0, 1.0)))
  }

  /** Unit vector from [from] to [to], or zero when they coincide. */
  fun direction(from: Vec3, to: Vec3): Vec3 {
    val d = to - from
    val n = d.length()
    return if (n < 1e-12) Vec3.ZERO else d * (1.0 / n)
  }

  /** Angle between two unit vectors, degrees (0 to 180). */
  fun angleDeg(a: Vec3, b: Vec3): Double =
    Math.toDegrees(acos((a.x * b.x + a.y * b.y + a.z * b.z).coerceIn(-1.0, 1.0)))
}

/** One observation offered to view retention: where the item was and where the camera stood (map frame). */
class ViewCandidate(val obsId: String, val mapPoint: Vec3, val cameraMap: Vec3, val quality: Double) {
  /** Item-to-camera unit vector in the map frame. */
  val direction: Vec3 = StudyGeometry.direction(mapPoint, cameraMap)
}

/**
 * Which views of a full group to keep. Pure: no state, unit-tested.
 *
 * The score of a set of views is (floor(spread / [StudyConfig.SPREAD_BIN_DEG]), Σ quality), compared
 * in that order, larger better; spread is the smallest angle between the item-to-camera directions
 * of any two views of the set (180° for a single view).
 */
object ViewRetention {
  fun spreadDeg(views: List<ViewCandidate>): Double {
    var best = 180.0
    for (i in views.indices) for (k in i + 1 until views.size) {
      best = minOf(best, StudyGeometry.angleDeg(views[i].direction, views[k].direction))
    }
    return best
  }

  fun spreadBin(views: List<ViewCandidate>, binDeg: Double = StudyConfig.SPREAD_BIN_DEG): Int =
    floor(spreadDeg(views) / binDeg + 1e-9).toInt()

  /**
   * [kept] are the views a group holds, in the order they were kept; [new] is offered. Returns the
   * one to leave out: [new] (skip it), one of [kept] (evict it), or null when the group has room.
   * With a full group every way of leaving one out is scored; they are tried in the order [new],
   * then [kept] in order, and a later one wins only with a strictly better score, so ties keep
   * what is already on disk.
   */
  fun leaveOut(
    kept: List<ViewCandidate>,
    new: ViewCandidate,
    perGroup: Int = StudyConfig.VIEWS_PER_GROUP,
    binDeg: Double = StudyConfig.SPREAD_BIN_DEG,
  ): ViewCandidate? {
    if (kept.size < perGroup) return null
    val pool = listOf(new) + kept
    var bestOut: ViewCandidate? = null
    var bestBin = Int.MIN_VALUE
    var bestQuality = Double.NEGATIVE_INFINITY
    for (out in pool) {
      val set = pool.filter { it !== out }
      val bin = spreadBin(set, binDeg)
      val quality = set.sumOf { it.quality }
      if (bestOut == null || bin > bestBin || (bin == bestBin && quality > bestQuality)) {
        bestOut = out
        bestBin = bin
        bestQuality = quality
      }
    }
    return bestOut
  }
}

/** What [ViewSelector.propose] suggests for one observation. Nothing changes until [ViewSelector.commit]. */
class ViewProposal(
  val candidate: ViewCandidate,
  val group: Int,
  val keep: Boolean,
  /** The view to delete when this one is kept, or null. */
  val evict: ViewCandidate?,
  /** [StudyConfig.VIEW_KEPT], or why not. */
  val reason: String,
)

/**
 * View retention of one scan (plan rule 2), independent of the matcher: observations are grouped by
 * place and each group keeps up to [perGroup] views that differ most in viewing direction, with at
 * most [perScan] views in the scan. Not thread-safe: the scan worker owns it.
 *
 * Groups: an observation joins the group with the nearest anchor within [radiusM] (ties: the older
 * group); otherwise it starts a group whose anchor is its own map point. Anchors never move.
 */
class ViewSelector(
  val perGroup: Int = StudyConfig.VIEWS_PER_GROUP,
  val perScan: Int = StudyConfig.VIEWS_PER_SCAN,
  val radiusM: Double = StudyConfig.GROUP_RADIUS_M,
  val binDeg: Double = StudyConfig.SPREAD_BIN_DEG,
) {
  private class Group(val anchor: Vec3, val views: MutableList<ViewCandidate> = ArrayList())

  private val groups = ArrayList<Group>()

  /** Views kept right now (written and not evicted). */
  var retained = 0
    private set

  val groupCount: Int
    get() = groups.size

  /** The group of an observation at [mapPoint], created when no anchor is within [radiusM]. */
  fun assign(mapPoint: Vec3): Int {
    var best = -1
    var bestD = Double.MAX_VALUE
    for ((i, g) in groups.withIndex()) {
      val d = distance(g.anchor, mapPoint)
      if (d <= radiusM && d < bestD) {
        best = i
        bestD = d
      }
    }
    if (best >= 0) return best
    groups.add(Group(mapPoint))
    return groups.size - 1
  }

  fun propose(group: Int, c: ViewCandidate): ViewProposal {
    val g = groups[group]
    if (g.views.size < perGroup) {
      return if (retained >= perScan) ViewProposal(c, group, false, null, StudyConfig.VIEW_SCAN_CAP)
      else ViewProposal(c, group, true, null, StudyConfig.VIEW_KEPT)
    }
    val out = ViewRetention.leaveOut(g.views, c, perGroup, binDeg)
    return if (out == null || out === c) ViewProposal(c, group, false, null, StudyConfig.VIEW_NOT_COMPLEMENTARY)
    else ViewProposal(c, group, true, out, StudyConfig.VIEW_KEPT)
  }

  /** Apply a kept proposal (once its file is written). */
  fun commit(p: ViewProposal) {
    if (!p.keep) return
    val g = groups[p.group]
    val ev = p.evict
    if (ev != null && g.views.remove(ev)) retained--
    g.views.add(p.candidate)
    retained++
  }

  fun retainedIds(): Set<String> = groups.flatMapTo(LinkedHashSet()) { g -> g.views.map { it.obsId } }

  fun viewsOf(group: Int): List<ViewCandidate> = groups[group].views.toList()
}

/** JSON builders for the lines of `obs/`. Field names and units are those of the plan's section 3. */
object ObsJson {
  fun vec(v: Vec3?): Any = v?.let { JSONArray().put(it.x).put(it.y).put(it.z) } ?: JSONObject.NULL

  fun box(b: Box): JSONArray = JSONArray().apply { for (x in b.toList()) put(x) }

  fun orNull(v: Any?): Any = v ?: JSONObject.NULL

  /** NaN and infinities are not JSON: they become null. */
  fun num(d: Double?): Any = if (d == null || d.isNaN() || d.isInfinite()) JSONObject.NULL else d

  /**
   * `{"world": {"t", "q"}, "map": {"t", "q", "forward", "yawDeg", "pitchDeg"} | null}`: `Camera.getPose()`
   * in the world frame and the same pose in the map frame (null when T_WM is unknown).
   */
  fun camera(pose: CameraPose, mapToWorld: Pose4?): JSONObject =
    JSONObject().apply {
      put("world", JSONObject().apply {
        put("t", vec(pose.position))
        put("q", JSONArray(pose.quaternion()))
      })
      if (mapToWorld == null) {
        put("map", JSONObject.NULL)
      } else {
        val m = StudyGeometry.mapPose(pose, mapToWorld)
        val f = StudyGeometry.forwardMap(pose, mapToWorld)
        put("map", JSONObject().apply {
          put("t", vec(m.position))
          put("q", JSONArray(m.quaternion()))
          put("forward", vec(f))
          put("yawDeg", num(StudyGeometry.yawDeg(f)))
          put("pitchDeg", num(StudyGeometry.pitchDeg(f)))
        })
      }
    }

  fun mapToWorld(t: Pose4?): Any =
    t?.let {
      JSONObject().apply {
        put("x", it.x)
        put("y", it.y)
        put("z", it.z)
        put("yawDeg", Math.toDegrees(it.yaw))
      }
    } ?: JSONObject.NULL

  fun intrinsics(k: Intrinsics): JSONObject =
    JSONObject().apply {
      put("fx", k.fx)
      put("fy", k.fy)
      put("cx", k.cx)
      put("cy", k.cy)
      put("width", k.width)
      put("height", k.height)
    }

  /** The baseline matcher's decision on one observation, kept apart from the observation itself. */
  fun decision(r: ObservationResult?): Any {
    if (r == null) return JSONObject.NULL
    return JSONObject().apply {
      put("matcher", StudyConfig.MATCHER_BASELINE)
      put("kind", r.kind)
      put("entry", orNull(r.entryId))
      put("sim", r.similarity?.toDouble() ?: JSONObject.NULL)
      put("distM", num(r.distanceM))
      put("byLabel", orNull(r.byLabel))
      put(
        "candidates",
        JSONArray().apply {
          for (c in r.candidates ?: emptyList()) {
            put(
              JSONObject().apply {
                put("entry", c.entryId)
                put("distM", c.distanceM)
                put("sim", c.similarity.toDouble())
                put("sameLabel", c.sameLabel)
                put("withinRadius", c.withinRadius)
                put("tentative", c.tentative)
              }
            )
          }
        },
      )
    }
  }

  /** What a view decision adds to an observation line. */
  class ViewFields(
    val group: Int?,
    val rect: Box?,
    val path: String?,
    val widthPx: Int?,
    val heightPx: Int?,
    val reason: String,
    val replaces: String?,
  )

  /** Fields common to both kinds of observation line, in the plan's order. */
  class Common(
    val obsId: String,
    val kind: String,
    val scan: String?,
    val session: String,
    val mapId: String?,
    val room: String?,
    val keyframe: Int?,
    val det: Int?,
    val tMs: Long,
    val studyTag: String,
    val imageWidth: Int,
    val imageHeight: Int,
    val rotation: Int,
    val pose: CameraPose,
    val mapToWorld: Pose4?,
    val intrinsics: Intrinsics,
    val embedding: FloatArray,
    val embeddingModel: String,
    val personInFrame: Boolean?,
    val personOverlap: Double?,
  )

  private fun begin(c: Common): JSONObject =
    JSONObject().apply {
      put("v", StudyConfig.SCHEMA)
      put("type", StudyConfig.TYPE_OBS)
      put("obs", c.obsId)
      put("kind", c.kind)
      put("scan", orNull(c.scan))
      put("session", c.session)
      put("mapId", orNull(c.mapId))
      put("room", orNull(c.room))
      put("kf", orNull(c.keyframe))
      put("det", orNull(c.det))
      put("t", c.tMs)
      put("studyTag", c.studyTag)
    }

  private fun finishCommon(o: JSONObject, c: Common, v: ViewFields) {
    o.put("image", JSONObject().put("w", c.imageWidth).put("h", c.imageHeight))
    o.put("rotation", c.rotation)
    o.put("camera", camera(c.pose, c.mapToWorld))
    o.put("mapToWorld", mapToWorld(c.mapToWorld))
    o.put("intrinsics", intrinsics(c.intrinsics))
    o.put("emb", EmbeddingFile.encode(c.embedding))
    o.put("embDim", c.embedding.size)
    o.put("embModel", c.embeddingModel)
    o.put("personInFrame", orNull(c.personInFrame))
    o.put("personOverlap", num(c.personOverlap))
    setView(o, v)
  }

  /**
   * The view fields of an observation line. The recorder builds a line before it writes the view
   * file, then sets the outcome here (these puts cannot fail), so a line that cannot be built never
   * leaves a file behind.
   */
  fun setView(o: JSONObject, v: ViewFields) {
    o.put("viewGroup", orNull(v.group))
    o.put("viewRect", v.rect?.let { box(it) } ?: JSONObject.NULL)
    o.put("view", orNull(v.path))
    o.put("viewPx", if (v.widthPx != null && v.heightPx != null) JSONArray().put(v.widthPx).put(v.heightPx) else JSONObject.NULL)
    o.put("viewReason", v.reason)
    o.put("viewReplaces", orNull(v.replaces))
  }

  /** One kept scan detection (`kind: "scan"`). */
  fun scanObservation(c: Common, obs: Observation, rawDet: Int?, view: ViewFields, result: ObservationResult?): JSONObject {
    val o = begin(c)
    o.put("rawDet", orNull(rawDet))
    o.put("label", obs.label)
    o.put("score", obs.score.toDouble())
    o.put("box", box(obs.box))
    o.put("depthM", obs.depthM)
    o.put("sizeM", obs.sizeM)
    o.put("distanceM", obs.distanceM)
    o.put("mapPoint", vec(obs.mapPoint))
    o.put("worldPoint", vec(obs.worldPoint))
    o.put("quality", obs.quality.toDouble())
    finishCommon(o, c, view)
    o.put("decision", decision(result))
    return o
  }

  /** What the verification knew about one capture, for an extra-view line. */
  class ExtraFields(
    val entryId: String?,
    val recordId: String,
    val queryId: String,
    val attempt: Int,
    val captureMode: String,
    val requestStartMs: Long,
    val pressMs: Long?,
    val msSinceQueryStart: Long,
    val msSinceNear: Long?,
    val similarity: Double,
    val outcome: String,
    val threshold: Double,
    val bestKind: String,
    val positionSource: String,
    val rect: Box,
    val distanceM: Double,
    val targetMap: Vec3?,
    val targetWorld: Vec3,
  )

  /** One verification capture (`kind: "extra_view"`). */
  fun extraView(c: Common, x: ExtraFields, view: ViewFields): JSONObject {
    val o = begin(c)
    o.put("rawDet", JSONObject.NULL)
    o.put("label", JSONObject.NULL)
    o.put("score", JSONObject.NULL)
    o.put("box", box(x.rect))
    o.put("depthM", JSONObject.NULL)
    o.put("sizeM", JSONObject.NULL)
    o.put("distanceM", x.distanceM)
    o.put("mapPoint", JSONObject.NULL)
    o.put("worldPoint", JSONObject.NULL)
    o.put("quality", JSONObject.NULL)
    finishCommon(o, c, view)
    o.put("decision", JSONObject.NULL)
    o.put("entry", orNull(x.entryId))
    o.put("record", x.recordId)
    o.put("query", x.queryId)
    o.put("attempt", x.attempt)
    o.put("captureMode", x.captureMode)
    o.put("requestStartMs", x.requestStartMs)
    o.put("pressMs", orNull(x.pressMs))
    o.put("msSinceQueryStart", x.msSinceQueryStart)
    o.put("msSinceNear", orNull(x.msSinceNear))
    o.put(
      "verify",
      JSONObject().apply {
        put("sim", x.similarity)
        put("outcome", x.outcome)
        put("threshold", x.threshold)
        put("bestKind", x.bestKind)
        put("positionSource", x.positionSource)
      },
    )
    o.put("target", JSONObject().put("mapPoint", vec(x.targetMap)).put("worldPoint", vec(x.targetWorld)))
    return o
  }

  /** A view written earlier in this scan was deleted: [by] replaced it in its group. */
  fun viewEvicted(scanId: String, evicted: ViewCandidate, view: String, by: String, group: Int, tMs: Long): JSONObject =
    JSONObject().apply {
      put("v", StudyConfig.SCHEMA)
      put("type", StudyConfig.TYPE_VIEW_EVICTED)
      put("scan", scanId)
      put("obs", evicted.obsId)
      put("view", view)
      put("by", by)
      put("viewGroup", group)
      put("t", tMs)
    }
}
