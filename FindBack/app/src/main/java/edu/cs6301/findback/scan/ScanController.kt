package edu.cs6301.findback.scan

import android.content.Context
import android.graphics.Bitmap
import android.graphics.RectF
import android.media.Image
import android.os.SystemClock
import android.util.Log
import edu.cs6301.findback.map.Pose4
import edu.cs6301.findback.map.Vec3
import edu.cs6301.findback.map.distance
import edu.cs6301.findback.memory.EventLog
import edu.cs6301.findback.memory.MemoryStore
import edu.cs6301.findback.ml.CameraFrame
import edu.cs6301.findback.ml.Embedder
import edu.cs6301.findback.ml.VerifyWindow
import edu.cs6301.findback.ml.YuvConverter
import edu.cs6301.findback.ml.YuvPlanes
import java.io.File
import java.nio.ByteOrder
import java.util.UUID
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicReference
import org.json.JSONArray
import org.json.JSONObject

/** One hit test of the depth self-check (plan section 5), made on the GL thread at capture. */
class HitSample(
  /** "centre" or "offCentre". */
  val point: String,
  /** The test point in CPU image pixels. */
  val u: Double,
  val v: Double,
  /** World position of the hit, or null when nothing was hit (or the point is off screen). */
  val hit: Vec3?,
  /** ARCore trackable type of the hit, or why there is none ("none", "outside_view"). */
  val hitType: String,
)

/**
 * Everything a keyframe needs once the ARCore frame is gone, copied on the GL thread (plan
 * section 4). The camera and depth images are already closed.
 */
class CapturedKeyframe(
  val scan: ActiveScan,
  val index: Int,
  /** Capture time, epoch ms and monotonic ms. */
  val tMs: Long,
  val elapsedMs: Long,
  val yuv: YuvPlanes,
  /** Sensor to display rotation, degrees clockwise. */
  val rotation: Int,
  /** CPU image intrinsics, scaled to the CPU image delivered. */
  val intrinsics: Intrinsics,
  /** `Camera.getPose()`: the physical camera in the world frame. */
  val pose: CameraPose,
  /** T_WM at capture (map frame to world frame). */
  val mapToWorld: Pose4,
  val roomId: String?,
  val depth: DepthMap,
  /** CPU image pixels to view pixels at capture (for the overlay), or null. */
  val imageToView: Affine2?,
  val hits: List<HitSample>,
  /** Time the GL thread spent on this capture. */
  val glMs: Double,
)

/** A box of the overlay in view pixels. [known]: matched an existing entry; else a new one. */
class OverlayBox(val left: Float, val top: Float, val right: Float, val bottom: Float, val label: String, val known: Boolean)

/** One running scan. Fields are written where their comments say. */
class ActiveScan(
  val id: String,
  val mapId: String,
  val detectorId: String,
  /** `scan_start.target`: "open" (containers are a later block). */
  val target: String,
  val container: String?,
  val saveFrames: Boolean,
  val studyTag: String,
  val startMs: Long,
  val startElapsedMs: Long,
  val session: ScanSession,
) {
  // GL thread.
  var nextIndex = 0
  var lastCameraPos: Vec3? = null
  @Volatile var skippedNoDepth = 0
  @Volatile var skippedNoImage = 0
  @Volatile var metersWalked = 0.0
  /** Monotonic time of the last keyframe taken (the scan start before the first one). */
  @Volatile var lastKeyframeElapsedMs = startElapsedMs

  // Scan worker.
  /** Entries this scan has detected: the count in the status hint. */
  @Volatile var itemCount = 0
  @Volatile var keyframesProcessed = 0
  /** Set once the scan has ended; keyframes still in the queue are then dropped. */
  @Volatile var finished = false
  val latencies = LinkedHashMap<String, MutableList<Double>>()
  /** Self-check distances at the image centre (metres). */
  val depthChecks = ArrayList<Double>()

  /** GL thread, every tracking frame: the camera path length during the scan. */
  fun trackWalk(p: Vec3) {
    val last = lastCameraPos
    if (last != null) {
      val d = distance(last, p)
      if (d < ScanConfig.WALK_JUMP_M) metersWalked += d // a longer step is a tracking jump
    }
    lastCameraPos = p
  }
}

/** What the UI shows when a scan ends. */
class ScanEnd(
  val scanId: String,
  val reason: String,
  val durationMs: Long,
  val metersWalked: Double,
  val keyframes: Int,
  val summary: SessionSummary?,
  val depthCheckMedianM: Double?,
  val depthChecks: Int,
  val error: String?,
)

/**
 * Owns scan state (plan section 10). The GL thread starts scans, paces and captures keyframes
 * ([pacer], [submit]); one keyframe at a time is processed on a single worker thread, which is the
 * only thread that touches the [ScanSession]; the UI hears back through [Ui] (always posted to
 * the main thread by the implementation). [stop] may be called from any thread.
 */
class ScanController(
  private val context: Context,
  private val events: EventLog,
  val inventory: InventoryStore,
  /** Used for its file helpers (JPEG and text under the files root). */
  private val files: MemoryStore,
  private val sessionId: String,
  private val embedder: () -> Embedder?,
  private val ui: Ui,
) {
  companion object {
    private const val TAG = "FindBackScan"
    /** At most one `internal_error` per site this often. */
    private const val ERROR_LOG_MS = 5000L

    fun newId(): String = UUID.randomUUID().toString().substring(0, 8)

    /** Copy ARCore's 16-bit depth image (one plane, little-endian millimetres) into an array. */
    fun copyDepth(image: Image): Triple<Int, Int, ShortArray> {
      val w = image.width
      val h = image.height
      val plane = image.planes[0]
      val buf = plane.buffer.duplicate().order(ByteOrder.LITTLE_ENDIAN)
      val rowStride = plane.rowStride
      val pixelStride = plane.pixelStride
      val out = ShortArray(w * h)
      for (y in 0 until h) {
        val row = y * rowStride
        for (x in 0 until w) out[y * w + x] = buf.getShort(row + x * pixelStride)
      }
      return Triple(w, h, out)
    }

    fun median(values: List<Double>): Double? {
      if (values.isEmpty()) return null
      val s = values.sorted()
      return if (s.size % 2 == 1) s[s.size / 2] else (s[s.size / 2 - 1] + s[s.size / 2]) / 2
    }

    fun vec(v: Vec3?): Any? = v?.let { JSONArray().put(it.x).put(it.y).put(it.z) }
  }

  interface Ui {
    fun scanStarted(scan: ActiveScan)

    fun overlay(boxes: List<OverlayBox>)

    fun scanEnded(end: ScanEnd)

    fun toast(message: String)
  }

  /** GL thread only. */
  val pacer = KeyframePacer()

  private val activeRef = AtomicReference<ActiveScan?>(null)
  val active: ActiveScan?
    get() = activeRef.get()

  /** A keyframe is on its way or being processed (set on the GL thread, cleared by the worker). */
  val inFlight = AtomicBoolean(false)

  private val worker: ExecutorService =
    Executors.newSingleThreadExecutor { r -> Thread(r, "FindBackScan").apply { priority = Thread.NORM_PRIORITY - 1 } }

  // Scan worker only.
  private var detector: SceneDetector? = null
  private var lastFlushMs = 0L
  private val lastErrorMs = HashMap<String, Long>()

  // ------------------------------------------------------------------ start and stop

  /**
   * GL thread: start a scan of the open area in map [mapId]. The caller has checked that the app
   * is located and idle. Returns the scan, or null when one is already running.
   */
  fun start(mapId: String, studyTag: String, captureMode: String, saveFrames: Boolean, roomId: String?, drift: Map<String, Any?>): ActiveScan? {
    if (activeRef.get() != null) return null
    val id = newId()
    val nowMs = System.currentTimeMillis()
    val model = embedder()?.modelName ?: ""
    val session =
      ScanSession(id, ScanSession.Mode.SCAN, mapId, null, ScanConfig.DEFAULT_DETECTOR, sessionId, model, nowMs, inventory)
    val a =
      ActiveScan(id, mapId, ScanConfig.DEFAULT_DETECTOR, ScanConfig.TARGET_OPEN, null, saveFrames, studyTag, nowMs, SystemClock.elapsedRealtime(), session)
    if (!activeRef.compareAndSet(null, a)) return null
    pacer.scanStarted()
    events.log(
      "scan_start",
      mapOf(
        "scan" to id,
        "detector" to a.detectorId,
        "target" to a.target,
        "container" to a.container,
        "mapId" to mapId,
        "studyTag" to studyTag,
        "captureMode" to captureMode,
        "saveFrames" to saveFrames,
        "passiveUpdates" to false,
        "room" to roomId,
        "entries" to inventory.snapshot().count { it.mapId == mapId },
        "embeddingModel" to model,
        "config" to configJson(),
      ) + drift
    )
    ui.scanStarted(a)
    return a
  }

  /** Any thread: end the running scan. Its keyframes still queued are processed first. */
  fun stop(reason: String) {
    val a = activeRef.getAndSet(null) ?: return
    worker.execute { finish(a, reason) }
  }

  /** GL thread: hand a captured keyframe to the worker. [inFlight] must have been free. */
  fun submit(kf: CapturedKeyframe) {
    inFlight.set(true)
    try {
      worker.execute {
        try {
          process(kf)
        } finally {
          inFlight.set(false)
        }
      }
    } catch (e: Exception) {
      inFlight.set(false) // the worker was shut down
    }
  }

  fun close() {
    stop("closed")
    worker.execute {
      try {
        detector?.close()
      } catch (e: Exception) {
        Log.w(TAG, "Detector close failed", e)
      }
      detector = null
    }
    worker.shutdown()
  }

  // ------------------------------------------------------------------ worker

  private fun process(kf: CapturedKeyframe) {
    val a = kf.scan
    if (a.finished) return
    try {
      processKeyframe(a, kf)
    } catch (e: Exception) {
      logError("scan_keyframe", e)
    }
  }

  private fun detectorOrLoad(): SceneDetector? {
    detector?.let { return it }
    val t0 = SystemClock.elapsedRealtimeNanos()
    return try {
      EffDet0Detector(context).also {
        detector = it
        events.log("model_loaded", mapOf("model" to it.modelName, "loadMs" to msSince(t0)))
      }
    } catch (e: Exception) {
      Log.e(TAG, "Detector failed to load", e)
      logError("scan_detector", e)
      null
    }
  }

  private fun processKeyframe(a: ActiveScan, kf: CapturedKeyframe) {
    val det = detectorOrLoad()
    val emb = embedder()
    if (det == null || emb == null) {
      ui.toast(if (det == null) "Scan detector failed to load; scan stopped." else "Model still loading; scan stopped.")
      stop("model_unavailable")
      return
    }
    val lat = LinkedHashMap<String, Double>()
    lat["queueMs"] = (SystemClock.elapsedRealtime() - kf.elapsedMs).toDouble()
    lat["glMs"] = kf.glMs
    val t0 = SystemClock.elapsedRealtimeNanos()
    val cf = CameraFrame(YuvConverter.toBitmap(kf.yuv), kf.rotation)
    lat["convertMs"] = msSince(t0)
    val k = kf.intrinsics.scaledTo(cf.width, cf.height)
    val geom = KeyframeGeometry(k, kf.pose, kf.mapToWorld, kf.depth)

    var t = SystemClock.elapsedRealtimeNanos()
    val raw = det.detect(cf)
    lat["detectMs"] = msSince(t)

    t = SystemClock.elapsedRealtimeNanos()
    val reports = DetectionFilter.apply(raw, geom)
    lat["filterMs"] = msSince(t)

    // Crop and embed the kept detections, as tap registration crops its box.
    t = SystemClock.elapsedRealtimeNanos()
    val reportOf = HashMap<Observation, DetectionReport>()
    val observations = ArrayList<Observation>()
    for (r in reports) {
      if (!r.kept) continue
      val b = r.detection.box
      val rect = cf.clamp(CameraFrame.pad(RectF(b.left, b.top, b.right, b.bottom), ScanConfig.CROP_PAD))
      val crop = cf.cropUpright(rect)
      val o =
        Observation(
          label = r.detection.label,
          score = r.detection.score,
          box = b,
          mapPoint = r.mapPoint!!,
          worldPoint = r.worldPoint!!,
          depthM = r.depthM!!,
          distanceM = r.distanceM!!,
          sizeM = r.sizeM!!,
          embedding = emb.embed(crop),
          cropWidthPx = rect.width(),
          cropHeightPx = rect.height(),
          crop = crop,
        )
      observations.add(o)
      reportOf[o] = r
    }
    lat["embedMs"] = msSince(t)

    t = SystemClock.elapsedRealtimeNanos()
    val info = KeyframeInfo(kf.index, kf.tMs, kf.roomId, geom.cameraMap, cf.width, cf.height)
    val merge = a.session.merge(info, observations)
    a.itemCount = a.session.itemCount
    lat["mergeMs"] = msSince(t)

    // Evidence (plan section 7) for the entries no detection matched in this keyframe.
    t = SystemClock.elapsedRealtimeNanos()
    val checks = a.session.evidence(info, geom, merge.matchedIds) { entry, vc -> appearance(cf, entry, vc, emb) }
    lat["evidenceMs"] = msSince(t)

    // Photos of entries that became real or got a better crop.
    t = SystemClock.elapsedRealtimeNanos()
    for ((id, crop) in merge.cropUpdates) {
      val bmp = crop as? Bitmap ?: continue
      try {
        files.saveJpeg(InventoryEntry.cropPath(id), bmp, ScanConfig.CROP_JPEG_QUALITY)
      } catch (e: Exception) {
        logError("scan_crop", e)
      }
    }
    val resultOf = merge.results.associateBy { it.observation }
    val hasPerson = DetectionFilter.hasPerson(reports)
    val frameFile = if (a.saveFrames) saveFrame(a, kf, cf, reports, reportOf, resultOf, hasPerson) else null
    lat["saveMs"] = msSince(t)
    lat["totalMs"] = msSince(t0)

    for (e in merge.added) {
      events.log("inv_add", mapOf("inventoryId" to e.id, "scan" to a.id, "keyframe" to kf.index, "entry" to e.toJson(withEmbedding = false)))
    }
    if (kf.hits.isNotEmpty()) logDepthCheck(a, kf, geom)

    val kinds = merge.results.groupingBy { it.kind }.eachCount()
    val dropped = reports.filter { !it.kept }.groupingBy { it.reason!! }.eachCount()
    events.log(
      "scan_keyframe",
      mapOf(
        "scan" to a.id,
        "index" to kf.index,
        "latency" to JSONObject(lat),
        "detections" to raw.size,
        "kept" to observations.size,
        "matched" to (kinds[ScanSession.KIND_KNOWN] ?: 0),
        "new" to (kinds[ScanSession.KIND_NEW] ?: 0),
        "tentative" to (kinds[ScanSession.KIND_TENTATIVE] ?: 0),
        "promoted" to (kinds[ScanSession.KIND_PROMOTED] ?: 0),
        "dropped" to JSONObject(dropped),
        "persons" to (dropped[DetectionFilter.PERSON] ?: 0),
        "validDepth" to kf.depth.validShare(),
        "depthSize" to "${kf.depth.width}x${kf.depth.height}",
        "imageSize" to "${cf.width}x${cf.height}",
        "rotation" to kf.rotation,
        "room" to kf.roomId,
        "cameraMap" to vec(geom.cameraMap),
        "itemsInScan" to a.itemCount,
        "evidenceChecks" to checks.size,
        "observable" to checks.count { it.view.visibility == Visibility.OBSERVABLE },
        "hidden" to checks.count { it.view.visibility == Visibility.HIDDEN },
        "noDepth" to checks.count { it.view.visibility == Visibility.NO_DEPTH },
        "appearanceChecks" to checks.count { it.similarity != null },
        "appearanceSeen" to checks.count { (it.similarity ?: -1f) >= ScanConfig.SEEN_SIMILARITY },
        "checks" to
          JSONArray().apply {
            for (c in checks) {
              put(
                JSONObject().apply {
                  put("entry", c.entryId)
                  put("vis", c.view.visibility.code)
                  put("u", c.view.u)
                  put("v", c.view.v)
                  put("zM", c.view.zM)
                  put("measuredM", c.view.measuredM ?: JSONObject.NULL)
                  put("sim", c.similarity?.toDouble() ?: JSONObject.NULL)
                }
              )
            }
          },
        "results" to
          JSONArray().apply {
            for (r in merge.results) {
              put(
                JSONObject().apply {
                  put("label", r.observation.label)
                  put("score", r.observation.score.toDouble())
                  put("kind", r.kind)
                  put("entry", r.entryId ?: JSONObject.NULL)
                  put("depthM", r.observation.depthM)
                  put("sizeM", r.observation.sizeM)
                  put("sim", r.similarity?.toDouble() ?: JSONObject.NULL)
                  put("distM", r.distanceM ?: JSONObject.NULL)
                }
              )
            }
          },
        "frameFile" to frameFile,
      )
    )
    for ((name, ms) in lat) a.latencies.getOrPut(name) { ArrayList() }.add(ms)
    a.keyframesProcessed++

    // The overlay: this keyframe's kept boxes in view pixels.
    val toView = kf.imageToView
    if (toView != null) {
      ui.overlay(
        merge.results.map { r ->
          val v = Upright.mapBox(r.observation.box, toView)
          OverlayBox(v[0], v[1], v[2], v[3], r.observation.label, r.kind == ScanSession.KIND_KNOWN)
        }
      )
    }

    val now = SystemClock.elapsedRealtime()
    if (now - lastFlushMs >= ScanConfig.STORE_FLUSH_MS) {
      lastFlushMs = now
      inventory.flush()
    }
  }

  /**
   * The appearance match (plan section 7): crop the window verification would crop for this
   * entry here (`VerifyWindow`, the same expressions as `handleVerify`), embed it and compare with
   * the entry's embedding. Null (no check) when the entry was embedded by another model.
   */
  private fun appearance(cf: CameraFrame, entry: InventoryEntry, vc: ViewCheck, emb: Embedder): Float? {
    if (entry.embeddingModel.isNotEmpty() && entry.embeddingModel != emb.modelName) return null
    val d = vc.distanceM.toFloat().coerceAtLeast(0.1f) // as handleVerify's d1
    val w =
      VerifyWindow.expected(
        entry.captureDistance, entry.cropWidthPx, entry.cropHeightPx, entry.imageWidth, d,
        cf.width, cf.height, vc.u.toFloat(), vc.v.toFloat(),
      )
    val crop = cf.cropUpright(cf.clamp(RectF(w.left, w.top, w.right, w.bottom)))
    val e = emb.embed(crop)
    if (e.size != entry.embedding.size) return null
    return Embedder.cosine(e, entry.embedding)
  }

  /**
   * A confirmed verification during Find of a scanned entry counts as `seen` (plan section 7),
   * once per query. Called on the verification's worker; [entryId] may have been deleted since.
   */
  fun onVerifiedSeen(entryId: String, queryId: String, attempt: Int, similarity: Float) {
    try {
      val now = System.currentTimeMillis()
      val e =
        inventory.update(entryId) { it.withOutcome(ScanConfig.OUTCOME_SEEN, now, ScanConfig.SOURCE_VERIFY, queryId) } ?: return
      inventory.logEvidence(entryId)
      inventory.flush()
      events.log(
        "inv_evidence",
        mapOf(
          "scan" to null,
          "source" to ScanConfig.SOURCE_VERIFY,
          "query" to queryId,
          "attempt" to attempt,
          "inventoryId" to entryId,
          "name" to e.name,
          "outcome" to ScanConfig.OUTCOME_SEEN,
          "similarity" to similarity,
          "counts" to countsJson(e),
        )
      )
    } catch (ex: Exception) {
      logError("inv_verify", ex)
    }
  }

  private fun countsJson(e: InventoryEntry): JSONObject =
    JSONObject().apply {
      put("seen", e.evidence.seenCount)
      put("missed", e.evidence.missedCount)
      put("notObserved", e.evidence.notObservedCount)
    }

  /**
   * `scan_depth_check`: the hit test made at capture against this pipeline's point for the same
   * pixel (depth: median of the 5×5 depth patch, then the unprojection), for the image centre and
   * one off-centre point. Only the off-centre point can reveal a sign error of u or v.
   */
  private fun logDepthCheck(a: ActiveScan, kf: CapturedKeyframe, geom: KeyframeGeometry) {
    fun one(h: HitSample): JSONObject {
      val depth = kf.depth.medianPatch(h.u, h.v)
      val pipe = depth?.let { geom.worldPoint(h.u, h.v, it) }
      val hit = h.hit
      val d = if (pipe != null && hit != null) distance(pipe, hit) else null
      return JSONObject().apply {
        put("u", h.u)
        put("v", h.v)
        put("hitType", h.hitType)
        put("distanceM", d ?: JSONObject.NULL)
        put("pipelineDepthM", depth ?: JSONObject.NULL)
        put("hitDepthM", hit?.let { -kf.pose.toCamera(it).z } ?: JSONObject.NULL)
        put("hitRangeM", hit?.let { distance(it, kf.pose.position) } ?: JSONObject.NULL)
        // The pipeline's point minus the hit, in the camera frame (+x right, +y up, -z ahead).
        put(
          "deltaCamera",
          if (pipe != null && hit != null) vec(kf.pose.toCamera(pipe) - kf.pose.toCamera(hit)) else JSONObject.NULL,
        )
        put("hitWorld", vec(hit) ?: JSONObject.NULL)
        put("pipelineWorld", vec(pipe) ?: JSONObject.NULL)
      }
    }
    val centre = kf.hits.firstOrNull { it.point == "centre" } ?: return
    val c = one(centre)
    if (!c.isNull("distanceM")) a.depthChecks.add(c.getDouble("distanceM"))
    val fields = LinkedHashMap<String, Any?>()
    fields["scan"] = a.id
    fields["index"] = kf.index
    fields["distanceM"] = if (c.isNull("distanceM")) null else c.getDouble("distanceM")
    fields["hitType"] = centre.hitType
    for (k in listOf("u", "v", "pipelineDepthM", "hitDepthM", "hitRangeM", "deltaCamera", "hitWorld", "pipelineWorld")) {
      fields[k] = if (c.isNull(k)) null else c.get(k)
    }
    fields["offCentre"] = kf.hits.firstOrNull { it.point == "offCentre" }?.let { one(it) }
    events.log("scan_depth_check", fields)
  }

  /**
   * Saved keyframe (plan section 9) in `scan/<scanId>/`: the JPEG (960 px wide, sensor
   * orientation), the depth as raw little-endian uint16, and the JSON. A keyframe with a person in
   * it keeps only the JSON, without the person's box, the image or the depth.
   */
  private fun saveFrame(
    a: ActiveScan,
    kf: CapturedKeyframe,
    cf: CameraFrame,
    reports: List<DetectionReport>,
    reportOf: Map<Observation, DetectionReport>,
    resultOf: Map<Observation, ObservationResult>,
    hasPerson: Boolean,
  ): String? =
    try {
      val dir = "scan/${a.id}"
      val stem = "kf%04d".format(kf.index)
      val k = kf.intrinsics.scaledTo(cf.width, cf.height)
      var image: JSONObject? = null
      var depth: JSONObject? = null
      if (!hasPerson) {
        val w = ScanConfig.SAVED_FRAME_WIDTH
        val h = (cf.height.toLong() * w / cf.width).toInt()
        val small = Bitmap.createScaledBitmap(cf.sensor, w, h, true)
        files.saveJpeg("$dir/$stem.jpg", small, ScanConfig.SAVED_FRAME_JPEG_QUALITY)
        image =
          JSONObject().apply {
            put("file", "$stem.jpg")
            put("width", w)
            put("height", h)
            put("sourceWidth", cf.width)
            put("sourceHeight", cf.height)
            put("orientation", "sensor")
          }
        val f = File(files.root, "$dir/${stem}_depth.u16")
        f.parentFile?.mkdirs()
        f.writeBytes(kf.depth.littleEndianBytes())
        depth =
          JSONObject().apply {
            put("file", "${stem}_depth.u16")
            put("width", kf.depth.width)
            put("height", kf.depth.height)
            put("format", "uint16le_mm")
          }
      }
      val obsOf = reportOf.entries.associate { (o, r) -> r to o }
      val o =
        JSONObject().apply {
          put("scan", a.id)
          put("index", kf.index)
          put("t", kf.tMs)
          put("session", sessionId)
          put("detector", a.detectorId)
          put("mapId", a.mapId)
          put("room", kf.roomId ?: JSONObject.NULL)
          put("rotation", kf.rotation)
          put("image", image ?: JSONObject.NULL)
          put("imageSkipped", if (hasPerson) "person" else JSONObject.NULL)
          put("depth", depth ?: JSONObject.NULL)
          put("validDepth", kf.depth.validShare())
          put("depthFromImage", JSONArray(kf.depth.fromImage.toList()))
          put("imageToView", kf.imageToView?.let { JSONArray(it.toList()) } ?: JSONObject.NULL)
          put(
            "intrinsics",
            JSONObject().apply {
              put("fx", k.fx)
              put("fy", k.fy)
              put("cx", k.cx)
              put("cy", k.cy)
              put("width", k.width)
              put("height", k.height)
            },
          )
          put(
            "cameraPose",
            JSONObject().apply {
              put("t", vec(kf.pose.position))
              put("q", JSONArray(kf.pose.quaternion()))
            },
          )
          put(
            "mapToWorld",
            JSONObject().apply {
              put("x", kf.mapToWorld.x)
              put("y", kf.mapToWorld.y)
              put("z", kf.mapToWorld.z)
              put("yawDeg", Math.toDegrees(kf.mapToWorld.yaw))
            },
          )
          put("personDetections", reports.count { it.reason == DetectionFilter.PERSON })
          put(
            "detections",
            JSONArray().apply {
              for (r in reports) {
                if (r.reason == DetectionFilter.PERSON) continue // never stored
                val res = obsOf[r]?.let { resultOf[it] }
                put(
                  JSONObject().apply {
                    put("box", JSONArray(r.detection.box.toList()))
                    put("label", r.detection.label)
                    put("score", r.detection.score.toDouble())
                    put("kept", r.kept)
                    put("reason", r.reason ?: JSONObject.NULL)
                    put("depthM", r.depthM ?: JSONObject.NULL)
                    put("sizeM", r.sizeM ?: JSONObject.NULL)
                    put("mapPoint", vec(r.mapPoint) ?: JSONObject.NULL)
                    put("kind", res?.kind ?: JSONObject.NULL)
                    put("entry", res?.entryId ?: JSONObject.NULL)
                  }
                )
              }
            },
          )
        }
      files.saveText("$dir/$stem.json", o.toString())
      "$dir/$stem.json"
    } catch (e: Exception) {
      logError("scan_frame", e)
      null
    }

  private fun finish(a: ActiveScan, reason: String) {
    var summary: SessionSummary? = null
    var error: String? = null
    try {
      summary = a.session.finish(System.currentTimeMillis())
    } catch (e: Exception) {
      error = e.toString()
      logError("scan_finish", e)
    }
    a.finished = true
    // One outcome per entry of the map (plan section 7), each logged.
    for (r in summary?.results ?: emptyList()) {
      val t = r.tally
      events.log(
        "inv_evidence",
        mapOf(
          "scan" to a.id,
          "source" to ScanConfig.SOURCE_SCAN,
          "inventoryId" to r.entry.id,
          "name" to r.entry.name,
          "label" to r.entry.label,
          "room" to r.entry.roomId,
          "outcome" to r.outcome,
          "eligible" to r.eligible,
          "isNew" to r.isNew,
          "matchedKeyframes" to (t?.matchedKeyframes ?: 0),
          "inViewKeyframes" to (t?.inViewKeyframes ?: 0),
          "observableKeyframes" to (t?.observableKeyframes ?: 0),
          "checkedKeyframes" to (t?.checkedKeyframes ?: 0),
          "hiddenKeyframes" to (t?.hiddenKeyframes ?: 0),
          "noDepthKeyframes" to (t?.noDepthKeyframes ?: 0),
          "bestSimilarity" to t?.bestSimilarity,
          "appearanceSeen" to (t?.appearanceSeen ?: false),
          "counts" to countsJson(r.entry),
        )
      )
    }
    val nowElapsed = SystemClock.elapsedRealtime()
    val med = JSONObject()
    for ((name, list) in a.latencies) median(list)?.let { med.put(name, it) }
    val depthMedian = median(a.depthChecks)
    events.log(
      "scan_end",
      mapOf(
        "scan" to a.id,
        "reason" to reason,
        "durationMs" to nowElapsed - a.startElapsedMs,
        "metersWalked" to a.metersWalked,
        "keyframes" to a.keyframesProcessed,
        "skippedNoDepth" to a.skippedNoDepth,
        "skippedNoImage" to a.skippedNoImage,
        "entries" to
          JSONObject().apply {
            put("new", summary?.newCount ?: 0)
            put("seenAgain", summary?.seenAgain ?: 0)
            put("missed", summary?.missed ?: 0)
            put("notObserved", summary?.notObserved ?: 0)
            put("total", summary?.results?.size ?: 0)
          },
        "tentativeDropped" to (summary?.tentativeDropped ?: 0),
        "medianLatency" to med,
        "depthChecks" to a.depthChecks.size,
        "depthCheckMedianM" to depthMedian,
        "error" to error,
      )
    )
    try {
      inventory.flush()
    } catch (e: Exception) {
      logError("scan_flush", e)
    }
    lastFlushMs = SystemClock.elapsedRealtime()
    ui.overlay(emptyList())
    ui.scanEnded(ScanEnd(a.id, reason, nowElapsed - a.startElapsedMs, a.metersWalked, a.keyframesProcessed, summary, depthMedian, a.depthChecks.size, error))
  }

  // ------------------------------------------------------------------ helpers

  private fun configJson(): JSONObject =
    JSONObject().apply {
      put("minScore", ScanConfig.MIN_SCORE.toDouble())
      put("maxPerKeyframe", ScanConfig.MAX_PER_KEYFRAME)
      put("rawScoreMin", ScanConfig.RAW_SCORE_MIN.toDouble())
      put("maxObjectM", ScanConfig.MAX_OBJECT_M)
      put("keyframeMinIntervalMs", ScanConfig.KEYFRAME_MIN_INTERVAL_MS)
      put("keyframeMoveM", ScanConfig.KEYFRAME_MOVE_M)
      put("keyframeTurnDeg", ScanConfig.KEYFRAME_TURN_DEG)
      put("maxTurnRateDegS", ScanConfig.MAX_TURN_RATE_DEG_S)
      put("depthRangeM", JSONArray(listOf(ScanConfig.DEPTH_MIN_M, ScanConfig.DEPTH_MAX_M)))
      put("mergeSimilarity", ScanConfig.MERGE_SIMILARITY.toDouble())
      put("tentativeConfirmMs", ScanConfig.TENTATIVE_CONFIRM_MS)
      put("seenSimilarity", ScanConfig.SEEN_SIMILARITY.toDouble())
      put("appearanceChecksPerKeyframe", ScanConfig.APPEARANCE_CHECKS_PER_KEYFRAME)
      put("missedMinKeyframes", ScanConfig.MISSED_MIN_KEYFRAMES)
      put("viewRangeM", JSONArray(listOf(ScanConfig.VIEW_MIN_M, ScanConfig.VIEW_MAX_M)))
      put("viewMargin", ScanConfig.VIEW_MARGIN)
    }

  private fun logError(site: String, e: Exception) {
    Log.e(TAG, "Error in $site", e)
    val now = SystemClock.elapsedRealtime()
    synchronized(lastErrorMs) {
      if (now - (lastErrorMs[site] ?: -ERROR_LOG_MS) < ERROR_LOG_MS) return
      lastErrorMs[site] = now
    }
    events.log("internal_error", mapOf("site" to site, "error" to e.toString()))
  }

  private fun msSince(t0: Long) = (SystemClock.elapsedRealtimeNanos() - t0) / 1e6
}
