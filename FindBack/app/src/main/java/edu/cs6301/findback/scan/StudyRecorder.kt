package edu.cs6301.findback.scan

import android.graphics.Bitmap
import android.os.SystemClock
import edu.cs6301.findback.map.Pose4
import edu.cs6301.findback.map.Vec3
import edu.cs6301.findback.memory.MemoryStore
import edu.cs6301.findback.ml.CameraFrame
import java.io.File
import java.io.FileOutputStream
import org.json.JSONArray
import org.json.JSONObject

/** v0.8: study logging state of one scan. Touched by the scan worker only. */
class StudyScan(val scanId: String) {
  val selector = ViewSelector()
  var observations = 0
  var obsBytes = 0L
  var viewsWritten = 0
  var viewsEvicted = 0
  /** Bytes of this scan's views on disk now (written minus evicted). */
  var viewBytes = 0L
  var keyframesWithPerson = 0
  /** Observations whose line could not be built (none expected: a non-finite number). */
  var lineErrors = 0
  var totalMs = 0.0
  val skipped = LinkedHashMap<String, Int>()
  val viewBytesOf = HashMap<String, Long>()
  val viewPathOf = HashMap<String, String>()

  fun skip(reason: String) {
    skipped[reason] = (skipped[reason] ?: 0) + 1
  }

  fun viewsJson(): JSONObject =
    JSONObject().apply {
      put("kept", selector.retained)
      put("written", viewsWritten)
      put("evicted", viewsEvicted)
      put("skipped", JSONObject().apply { for ((k, v) in skipped) put(k, v) })
      put("groups", selector.groupCount)
      put("bytes", viewBytes)
    }
}

/** The view outcome before it is known: the recorder replaces it with [ObsJson.setView]. */
private const val VIEW_PENDING = "pending"

/** v0.8: what study logging added to one keyframe (`scan_keyframe.study`). */
class StudyKeyframeStats(
  val observations: Int,
  val viewsWritten: Int,
  val viewsEvicted: Int,
  val viewsSkipped: Int,
  val obsBytes: Long,
  val viewBytes: Long,
  val personInFrame: Boolean,
  val lineErrors: Int,
  val ms: Double,
) {
  fun toJson(): JSONObject =
    JSONObject().apply {
      put("obs", observations)
      put("viewsWritten", viewsWritten)
      put("viewsEvicted", viewsEvicted)
      put("viewsSkipped", viewsSkipped)
      put("obsBytes", obsBytes)
      put("viewBytes", viewBytes)
      put("personInFrame", personInFrame)
      put("lineErrors", lineErrors)
      put("ms", ms)
    }
}

/**
 * v0.8: one verification capture of a scanned entry, handed from the verification coroutine to the
 * scan worker, which checks for people, writes the crop and the `extra_view` line.
 */
class ExtraViewCapture(
  val obsId: String,
  val frame: CameraFrame,
  /** The scored candidate rectangle that won (sensor pixels, clamped): the crop. */
  val rect: Box,
  val crop: Bitmap,
  val embedding: FloatArray,
  val embeddingModel: String,
  val pose: CameraPose,
  val mapToWorld: Pose4?,
  val intrinsics: Intrinsics,
  val mapId: String?,
  val room: String?,
  val tMs: Long,
  val studyTag: String,
  val extra: ObsJson.ExtraFields,
)

/**
 * v0.8 study logging on the device (docs/PLAN_v0.8_study_logging.md): `obs/<scanId>.jsonl`, the
 * view crops in `views/<scanId>/`, and the extra views of verification. Runs on the scan worker,
 * never on the GL thread. Each keyframe's lines go to the file in one append, after its new view
 * files are written and before evicted ones are deleted, so a crash leaves at most orphan files,
 * never a line naming a file that was not written.
 */
class StudyRecorder(private val files: MemoryStore, private val sessionId: String) {
  private fun append(relativePath: String, text: String): Long {
    val f = File(files.root, relativePath)
    f.parentFile?.mkdirs()
    val bytes = text.toByteArray(Charsets.UTF_8)
    FileOutputStream(f, true).use { it.write(bytes) }
    return bytes.size.toLong()
  }

  /**
   * The observation lines of one keyframe. [observations] are the kept detections in the order
   * their ids were given (`j`), [rects] their crop rectangles, [rawIndex] their index in the
   * detector's raw output, [reports] every raw detection (for the person rule).
   */
  fun keyframe(
    a: ActiveScan,
    st: StudyScan,
    kf: CapturedKeyframe,
    imageWidth: Int,
    imageHeight: Int,
    intrinsics: Intrinsics,
    cameraMap: Vec3,
    reports: List<DetectionReport>,
    observations: List<Observation>,
    rects: List<Box>,
    rawIndex: List<Int>,
    resultOf: Map<Observation, ObservationResult>,
    embeddingModel: String,
  ): StudyKeyframeStats {
    val t0 = SystemClock.elapsedRealtimeNanos()
    val persons = PersonRule.boxes(reports)
    val personInFrame = persons.isNotEmpty()
    if (personInFrame) st.keyframesWithPerson++
    val sb = StringBuilder()
    val toDelete = ArrayList<String>()
    var written = 0
    var evicted = 0
    var skipped = 0
    var viewBytes = 0L
    var lineErrors = 0
    for ((j, o) in observations.withIndex()) {
      val obsId = o.obsId ?: StudyConfig.obsId(a.id, kf.index, j)
      val rect = rects[j]
      // The line first, without its view outcome: one that cannot be built leaves no file behind
      // and joins no retention group, so an offline replay of the groups stays exact.
      val line =
        try {
          val common =
            ObsJson.Common(
              obsId = obsId,
              kind = StudyConfig.KIND_SCAN,
              scan = a.id,
              session = sessionId,
              mapId = a.mapId,
              room = kf.roomId,
              keyframe = kf.index,
              det = j,
              tMs = kf.tMs,
              studyTag = a.studyTag,
              imageWidth = imageWidth,
              imageHeight = imageHeight,
              rotation = kf.rotation,
              pose = kf.pose,
              mapToWorld = kf.mapToWorld,
              intrinsics = intrinsics,
              embedding = o.embedding,
              embeddingModel = embeddingModel,
              personInFrame = personInFrame,
              personOverlap = PersonRule.overlap(rect, persons),
            )
          ObsJson.scanObservation(common, o, rawIndex[j], ObsJson.ViewFields(null, rect, null, null, null, VIEW_PENDING, null), resultOf[o])
        } catch (e: Exception) {
          lineErrors++
          st.lineErrors++
          continue
        }
      val group = st.selector.assign(o.mapPoint)
      var path: String? = null
      var size: Pair<Int, Int>? = null
      var replaces: String? = null
      var evictLine: JSONObject? = null
      val reason: String
      val bmp = o.crop as? Bitmap
      if (!PersonRule.allowsView(rect, persons)) {
        reason = StudyConfig.VIEW_PERSON
      } else if (bmp == null) {
        reason = StudyConfig.VIEW_NO_CROP
      } else {
        val p = st.selector.propose(group, ViewCandidate(obsId, o.mapPoint, cameraMap, o.quality.toDouble()))
        if (!p.keep) {
          reason = p.reason
        } else {
          val rel = StudyConfig.viewPath(a.id, obsId)
          val bytes =
            try {
              files.saveJpeg(rel, bmp, ScanConfig.CROP_JPEG_QUALITY)
            } catch (e: Exception) {
              -1L
            }
          if (bytes < 0) {
            reason = StudyConfig.VIEW_WRITE_ERROR
          } else {
            st.selector.commit(p)
            reason = StudyConfig.VIEW_KEPT
            path = rel
            size = bmp.width to bmp.height
            st.viewBytesOf[obsId] = bytes
            st.viewPathOf[obsId] = rel
            st.viewBytes += bytes
            st.viewsWritten++
            written++
            viewBytes += bytes
            val ev = p.evict
            if (ev != null) {
              replaces = ev.obsId
              val evPath = st.viewPathOf.remove(ev.obsId) ?: StudyConfig.viewPath(a.id, ev.obsId)
              st.viewBytes -= st.viewBytesOf.remove(ev.obsId) ?: 0L
              st.viewsEvicted++
              evicted++
              toDelete.add(evPath)
              evictLine = ObsJson.viewEvicted(a.id, ev, evPath, obsId, group, System.currentTimeMillis())
            }
          }
        }
      }
      if (path == null) {
        st.skip(reason)
        skipped++
      }
      ObsJson.setView(line, ObsJson.ViewFields(group, rect, path, size?.first, size?.second, reason, replaces))
      sb.append(line.toString()).append('\n')
      if (evictLine != null) sb.append(evictLine.toString()).append('\n')
    }
    var obsBytes = 0L
    if (sb.isNotEmpty()) {
      obsBytes = append(StudyConfig.obsFile(a.id), sb.toString())
      st.obsBytes += obsBytes
      st.observations += observations.size - lineErrors
    }
    // Only now, with the eviction lines on disk: views of this scan that left their group.
    for (rel in toDelete) File(files.root, rel).delete()
    val ms = (SystemClock.elapsedRealtimeNanos() - t0) / 1e6
    st.totalMs += ms
    return StudyKeyframeStats(observations.size - lineErrors, written, evicted, skipped, obsBytes, viewBytes, personInFrame, lineErrors, ms)
  }

  /**
   * The last line of `obs/<scanId>.jsonl` (`type: "scan_end"`), and the counts for the `scan_end`
   * event. [summary] is null when the scan's end failed.
   */
  fun scanEnd(a: ActiveScan, st: StudyScan, reason: String, summary: SessionSummary?, keyframes: Int): JSONObject {
    val retained = st.selector.retainedIds()
    val counts =
      JSONObject().apply {
        put("schema", StudyConfig.SCHEMA)
        put("obsFile", StudyConfig.obsFile(a.id))
        put("observations", st.observations)
        put("obsBytes", st.obsBytes)
        put("views", st.viewsJson())
        put("keyframesWithPerson", st.keyframesWithPerson)
        put("lineErrors", st.lineErrors)
        put("studyMsTotal", st.totalMs)
        put("historyItems", summary?.results?.count { r -> r.entry.history.lastOrNull()?.scan == a.id } ?: 0)
        put("confirmedSet", summary?.results?.count { r -> r.entry.confirmed?.scan == a.id } ?: 0)
      }
    val line =
      JSONObject().apply {
        put("v", StudyConfig.SCHEMA)
        put("type", StudyConfig.TYPE_SCAN_END)
        put("scan", a.id)
        put("session", sessionId)
        put("t", System.currentTimeMillis())
        put("reason", reason)
        put("keyframes", keyframes)
        put("observations", st.observations)
        put("views", st.viewsJson())
        put("retained", JSONArray(retained.toList()))
        put("created", JSONArray(summary?.createdIds ?: emptyList<String>()))
        put(
          "tentativeDropped",
          JSONArray().apply {
            for (id in summary?.droppedTentative ?: emptyList()) {
              put(JSONObject().put("entry", id).put("obs", JSONArray(a.session.linkedObservations(id))))
            }
          },
        )
        put("error", if (summary == null) "finish_failed" else JSONObject.NULL)
      }
    append(StudyConfig.obsFile(a.id), line.toString() + "\n")
    return counts
  }

  /**
   * One verification capture: the person check on the whole frame with the scene detector, the
   * crop under `views/extra/` when no person box touches it, and the `extra_view` line. Returns the
   * fields of the `study_extra_view` event.
   */
  fun extraView(x: ExtraViewCapture, detector: SceneDetector?): Map<String, Any?> {
    val t0 = SystemClock.elapsedRealtimeNanos()
    var persons: List<Box>? = null
    var detectError: String? = null
    if (detector != null) {
      try {
        persons = PersonRule.fromDetections(detector.detect(x.frame))
      } catch (e: Exception) {
        detectError = e.toString()
      }
    }
    val common =
      ObsJson.Common(
        obsId = x.obsId,
        kind = StudyConfig.KIND_EXTRA_VIEW,
        scan = null,
        session = sessionId,
        mapId = x.mapId,
        room = x.room,
        keyframe = null,
        det = null,
        tMs = x.tMs,
        studyTag = x.studyTag,
        imageWidth = x.frame.width,
        imageHeight = x.frame.height,
        rotation = x.frame.rotation,
        pose = x.pose,
        mapToWorld = x.mapToWorld,
        intrinsics = x.intrinsics,
        embedding = x.embedding,
        embeddingModel = x.embeddingModel,
        personInFrame = persons?.isNotEmpty(),
        personOverlap = persons?.let { PersonRule.overlap(x.rect, it) },
      )
    // The line first (see [keyframe]); it throws before any file is written.
    val line = ObsJson.extraView(common, x.extra, ObsJson.ViewFields(null, x.rect, null, null, null, VIEW_PENDING, null))
    var path: String? = null
    var bytes = 0L
    val reason =
      when {
        persons == null -> StudyConfig.VIEW_PERSON_CHECK_FAILED
        !PersonRule.allowsView(x.rect, persons) -> StudyConfig.VIEW_PERSON
        else -> {
          val rel = StudyConfig.extraViewPath(x.obsId)
          try {
            bytes = files.saveJpeg(rel, x.crop, ScanConfig.CROP_JPEG_QUALITY)
            path = rel
            StudyConfig.VIEW_KEPT
          } catch (e: Exception) {
            StudyConfig.VIEW_WRITE_ERROR
          }
        }
      }
    ObsJson.setView(
      line,
      ObsJson.ViewFields(null, x.rect, path, if (path != null) x.crop.width else null, if (path != null) x.crop.height else null, reason, null),
    )
    val obsBytes = append(StudyConfig.EXTRA_OBS_FILE, line.toString() + "\n")
    return mapOf(
      "obs" to x.obsId,
      "query" to x.extra.queryId,
      "attempt" to x.extra.attempt,
      "inventoryId" to x.extra.entryId,
      "view" to path,
      "viewReason" to reason,
      "personInFrame" to persons?.isNotEmpty(),
      "detectError" to detectError,
      "obsFile" to StudyConfig.EXTRA_OBS_FILE,
      "obsBytes" to obsBytes,
      "viewBytes" to bytes,
      "ms" to (SystemClock.elapsedRealtimeNanos() - t0) / 1e6,
    )
  }
}
