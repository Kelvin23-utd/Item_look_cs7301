package edu.cs6301.findback

import android.graphics.Bitmap
import android.graphics.PointF
import android.graphics.RectF
import android.opengl.Matrix
import android.os.SystemClock
import android.util.Log
import androidx.lifecycle.DefaultLifecycleObserver
import androidx.lifecycle.LifecycleOwner
import com.google.ar.core.Anchor
import com.google.ar.core.AugmentedImage
import com.google.ar.core.Camera
import com.google.ar.core.Coordinates2d
import com.google.ar.core.DepthPoint
import com.google.ar.core.Frame
import com.google.ar.core.HitResult
import com.google.ar.core.Plane
import com.google.ar.core.Point
import com.google.ar.core.Pose
import com.google.ar.core.Session
import com.google.ar.core.TrackingState
import com.google.ar.core.examples.java.common.helpers.DisplayRotationHelper
import com.google.ar.core.examples.java.common.samplerender.SampleRender
import com.google.ar.core.examples.java.common.samplerender.arcore.BackgroundRenderer
import com.google.ar.core.exceptions.CameraNotAvailableException
import com.google.ar.core.exceptions.NotYetAvailableException
import edu.cs6301.findback.map.FixType
import edu.cs6301.findback.map.GuidanceInput
import edu.cs6301.findback.map.GuidanceMode
import edu.cs6301.findback.map.GuidancePlanner
import edu.cs6301.findback.map.HomeMap
import edu.cs6301.findback.map.Localizer
import edu.cs6301.findback.map.Room
import edu.cs6301.findback.map.Route
import edu.cs6301.findback.map.Vec3
import edu.cs6301.findback.map.horizontalDistance
import edu.cs6301.findback.memory.EmbeddingFile
import edu.cs6301.findback.memory.Lineage
import edu.cs6301.findback.memory.ObjectRecord
import edu.cs6301.findback.ml.BoxDetector
import edu.cs6301.findback.ml.CameraFrame
import edu.cs6301.findback.ml.Detection
import edu.cs6301.findback.ml.Embedder
import edu.cs6301.findback.policy.PolicyDecision
import edu.cs6301.findback.policy.PolicyInputs
import edu.cs6301.findback.policy.PositionSource
import edu.cs6301.findback.policy.VerificationPolicy
import edu.cs6301.findback.render.LabelRender
import edu.cs6301.findback.render.PointCloudRender
import edu.cs6301.findback.study.CaptureMode
import edu.cs6301.findback.study.CaptureRules
import edu.cs6301.findback.study.GroundTruth
import edu.cs6301.findback.study.StudySettings
import java.util.UUID
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.ConcurrentLinkedQueue
import java.util.concurrent.atomic.AtomicReference
import kotlin.math.abs
import kotlin.math.atan2
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.MainScope
import kotlinx.coroutines.cancel
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.json.JSONArray
import org.json.JSONObject

/** A capture awaiting the user's confirmation and name. */
class RegistrationDraft(
  val id: String,
  val crop: Bitmap,
  val embedding: FloatArray,
  val rawLabel: String,
  val rawConfidence: Float,
  val boxSource: String,
  val cropRect: RectF,
  val worldPosition: FloatArray,
  val captureDistance: Float,
  val imageWidth: Int,
  val imageHeight: Int,
  val hitType: String,
  val anchor: Anchor,
  val detections: Int,
  val rejectedBoxes: Int,
  val fx: Float,
  val latencies: Map<String, Double>,
  // v0.3 home map (null when not located at capture).
  val mapId: String?,
  val mapPosition: FloatArray?,
  val viewPosition: FloatArray?,
  /** Room suggested in the dialog: the current room, else the room of the nearest labelled node. */
  val defaultRoomId: String?,
  /** Drift proxies at capture, as log fields. */
  val drift: Map<String, Any?>,
  /** The home map exists (rooms can be picked even when not located). */
  val hasMap: Boolean,
  /** v0.4 "Update location": the record this capture will replace, or null for a new item. */
  val updateOf: ObjectRecord? = null,
)

/** A camera-view tap on its way to the GL thread, with the "Update location" target at tap time. */
class PendingTap(val point: PointF, val updateOf: ObjectRecord?)

/** One "find my item" query, from selecting the item to answering the ground-truth prompt. */
class Query(
  val id: String,
  val record: ObjectRecord,
  val startMs: Long,
  val inputs: PolicyInputs,
  val decision: PolicyDecision,
  // v0.4: the study settings are fixed for the whole query, so all of its events agree.
  val studyTag: String = "",
  val captureMode: CaptureMode = CaptureMode.FREE,
  /** Query start on the monotonic clock (SystemClock.elapsedRealtime), for the v0.4 timing fields. */
  val startElapsedMs: Long = 0L,
) {
  /** Capture attempts started (an attempt that fails before its `verify` event still counts). */
  @Volatile var verifications = 0
  @Volatile var lastOutcome: String? = null
  @Volatile var lastSimilarity: Float? = null
  /**
   * After ground truth "present" or "displaced": the next tap marks the object's true centre, to
   * measure the marker error or the displacement.
   */
  @Volatile var measuring = false
  /** The ground truth chosen at Done while [measuring] ("present" or "displaced"). */
  @Volatile var truth: String? = null

  // v0.4 study state.
  /** Captures that reached their `verify` event. PAIRED allows one. */
  @Volatile var captures = 0
  /** A capture is being processed (set on the GL thread, cleared when its result is shown). */
  @Volatile var capturing = false
  /** `msSinceQueryStart` of the first logged capture, or null. */
  @Volatile var msToFirstVerify: Long? = null
  /** Monotonic time of the first NEAR, or null. */
  @Volatile var nearElapsedMs: Long? = null
  /** The item can be resolved in this frame (GL thread, every frame). */
  @Volatile var resolvable = false
  // UI thread only.
  /** [resolvable] when Done led to the ground-truth prompt: decides `noCaptureReason`. */
  var resolvableAtDone = false
  /** `query_end` was logged; a late second answer is ignored. */
  var ended = false

  // Guidance, updated on the GL thread about 5 times a second; read by the UI at the end.
  @Volatile var mode: GuidanceMode? = null
  @Volatile var nearRule: String? = null
  /** Elapsed ms from query start to the first NEAR, or null. */
  @Volatile var timeToNearMs: Long? = null
  /** Modes in the order first seen. */
  @Volatile var modesSeen: List<String> = emptyList()
  @Volatile var routeM: Double? = null
  @Volatile var straightM: Double? = null
  @Volatile var itemRoom: String? = null
  @Volatile var userRoom: String? = null
  /** FAR_ROUTE: world position of the chevron's waypoint. */
  @Volatile var waypointWorld: FloatArray? = null
  // GL thread only.
  var route: Route? = null
  var lastPlanMs = 0L
  var lastGuidanceMs = 0L
}

/** Ghost placement data: view corners (0,0),(w,0),(0,h) in CPU-image normalized coordinates. */
class GhostMapping(val cornersImageNorm: FloatArray, val rotation: Int, val viewWidth: Int, val viewHeight: Int)

/**
 * GL-thread AR loop: locates the home map (through [Localizer]), turns taps into object records,
 * resolves stored positions into the current session, plans guidance, draws labels and runs
 * verification captures. UI requests arrive as commands ([runOnGl]); the UI reads snapshots
 * published at 5 Hz.
 */
class FindBackRenderer(private val activity: MainActivity) :
  DefaultLifecycleObserver, SampleRender.Renderer {
  companion object {
    private const val TAG = "FindBack"
    private const val UI_UPDATE_MS = 200L

    /** Detector boxes larger than this fraction of the frame are scene boxes, not objects. */
    private const val MAX_BOX_FRACTION = 0.25f
    /** Detector boxes physically wider than this (metres) are rejected as non-objects. */
    private const val MAX_OBJECT_M = 0.6f
    /** Fallback crop: a fixed physical window around the tapped 3D point (metres). */
    private const val DEFAULT_WINDOW_M = 0.22f
    /**
     * Indoor sanity bound on hit distance. With little phone motion, ARCore's depth-from-motion
     * can return points 15-20 m away for objects ~1.5 m away (seen on the Pixel 8).
     */
    private const val MAX_HIT_DISTANCE_M = 5f

    /** Outside a query, draw labels only for items within this (horizontal) distance. */
    const val LABEL_MAX_M = 6.0
    /** Ghost images are stored at this fraction of the CPU image size. */
    private const val GHOST_SCALE = 0.5f
  }

  private lateinit var view: MainView
  private val scope = MainScope()
  private val displayRotationHelper = DisplayRotationHelper(activity)

  private lateinit var backgroundRenderer: BackgroundRenderer
  private val pointCloudRender = PointCloudRender()
  private val labelRender = LabelRender()
  private val viewMatrix = FloatArray(16)
  private val projectionMatrix = FloatArray(16)
  private val viewProjectionMatrix = FloatArray(16)

  private val store get() = activity.store
  private val events get() = activity.events

  // Frozen models shared by all conditions.
  private val detector = BoxDetector()
  @Volatile private var embedder: Embedder? = null

  // UI -> GL requests.
  private val pendingTap = AtomicReference<PendingTap?>(null)
  @Volatile private var pendingVerify = false
  @Volatile private var mlBusy = false
  private val pendingQuery = AtomicReference<ObjectRecord?>(null)
  private val glCommands = ConcurrentLinkedQueue<() -> Unit>()
  @Volatile var query: Query? = null
    private set
  /**
   * v0.4 "Update location": the record waiting for a tap on the item where it is now, or null.
   * Written on the UI thread only; the tap carries it to the GL thread ([PendingTap]).
   */
  @Volatile var updateTarget: ObjectRecord? = null
    private set

  // Same-session anchors keyed by record id; anchors to release on the GL thread.
  private val sessionAnchors = ConcurrentHashMap<String, Anchor>()
  private val anchorsToDetach = ConcurrentLinkedQueue<Anchor>()

  // Reference image state (written on GL thread). Positions now come from the map, not the image.
  @Volatile private var refFullTracking = false
  @Volatile private var refEverLocked = false
  private var firstFrameMs = 0L

  @Volatile private var cameraTracking = TrackingState.PAUSED
  private var lastUiUpdateMs = 0L
  private var overlayShown = false
  private var viewWidth = 0
  private var viewHeight = 0

  // Home map: GL-thread owner, and the snapshots the UI reads.
  private val localizer: Localizer
  /** Latest location state, published at 5 Hz for the UI thread. */
  @Volatile var uiState: Localizer.State? = null
    private set
  @Volatile var ghostMapping: GhostMapping? = null
    private set
  /** Records whose 3D label may be drawn outside a query (recomputed at 5 Hz). */
  @Volatile private var labelIds: Set<String> = emptySet()
  @Volatile private var ghostCaptureRequested = false
  private val pendingGhost = AtomicReference<Bitmap?>(null)
  private var miniMapCache: Pair<HomeMap, MiniMapView.Data>? = null

  private val locUi =
    object : Localizer.Ui {
      override fun askSetHome(onAnswer: (Boolean) -> Unit) =
        view.post { view.dialogs.askSetHome { yes -> runOnGl { onAnswer(yes) } } }

      override fun askUseNewFix(jumpM: Double, jumpDeg: Double, fixName: String, onAnswer: (Boolean) -> Unit) =
        view.post { view.dialogs.askUseNewFix(jumpM, jumpDeg, fixName) { yes -> runOnGl { onAnswer(yes) } } }

      override fun askAddPicture(onAnswer: (Boolean) -> Unit) =
        view.post { view.dialogs.askAddPicture { yes -> runOnGl { onAnswer(yes) } } }

      override fun spotCaptured(capture: Localizer.SpotCapture) =
        view.post {
          val n = (uiState?.map?.fixPoints?.count { it.type == FixType.SPOT } ?: 0) + 1
          view.dialogs.askSpotName(
            if (capture.isHome) "Home spot" else "Spot $n",
            capture.isHome,
            pendingGhost.get(),
            onSave = { name -> saveSpot(capture, name) },
            onCancel = {
              pendingGhost.set(null)
              events.log("spot_cancel", mapOf("fixPoint" to capture.id))
            },
          )
        }

      override fun lockFinished(ok: Boolean, message: String) =
        view.post {
          if (ok) view.exitLockMode() else view.lockFailed()
          if (message.isNotEmpty()) view.toast(message)
        }

      override fun toast(message: String) = view.post { view.toast(message) }
    }

  init {
    localizer =
      Localizer(
        events = activity.events,
        mapStore = activity.mapStore,
        sessionId = activity.sessionId,
        refWidthM = activity.refWidthM,
        anchorsToDetach = anchorsToDetach,
        ui = locUi,
        loaded = activity.loadedMap,
      )
    scope.launch(Dispatchers.Default) {
      val t0 = SystemClock.elapsedRealtimeNanos()
      try {
        val e = Embedder(activity)
        embedder = e
        events.log("model_loaded", mapOf("model" to e.modelName, "loadMs" to msSince(t0)))
      } catch (ex: Exception) {
        Log.e(TAG, "Embedder failed to load", ex)
        activity.runOnUiThread {
          if (::view.isInitialized) view.toast("Embedding model failed to load: ${ex.message}")
        }
      }
    }
  }

  /** Run [block] on the GL thread at the start of the next frame. */
  fun runOnGl(block: () -> Unit) {
    glCommands.add(block)
  }

  fun bindView(view: MainView) {
    this.view = view
    view.itemsButton.setOnClickListener {
      view.showItems(
        store.snapshot().sortedByDescending { it.lastSeenMs },
        store.root,
        uiState,
        onFind = { startQuery(it) },
        onUpdate = { startUpdate(it) },
        onRename = { rec, name ->
          store.rename(rec.id, name)
          events.log("rename", mapOf("record" to rec.id, "from" to rec.name, "to" to name))
        },
        onDelete = { rec ->
          sessionAnchors.remove(rec.id)?.let { anchorsToDetach.add(it) }
          store.delete(rec.id)
          events.log("delete", mapOf("record" to rec.id, "name" to rec.name))
        }
      )
    }
    view.verifyButton.setOnClickListener { pendingVerify = true }
    view.doneButton.setOnClickListener { onDonePressed() }
  }

  /** UI thread. Done is "Cancel" during "Update location" and "Skip" during a measurement. */
  private fun onDonePressed() {
    if (updateTarget != null) return cancelUpdate("waiting")
    val q = query ?: return
    if (q.measuring) {
      endQuery(q, q.truth ?: GroundTruth.PRESENT, null) // measurement skipped
      return
    }
    // PAIRED and BENCHMARK tie every capture to this query's end: let a running capture finish.
    if (q.captureMode != CaptureMode.FREE && q.capturing) return view.toast("Still processing the capture…")
    val resolvable = q.resolvable
    if (CaptureRules.askCaptureFirst(q.captureMode, q.captures, resolvable)) {
      view.askCaptureFirst(q.record.name) { askGroundTruth(q, resolvable) }
    } else {
      askGroundTruth(q, resolvable)
    }
  }

  /** UI thread: the five-way ground truth; "present" and "displaced" go on to the measuring tap. */
  private fun askGroundTruth(q: Query, resolvable: Boolean) {
    if (query !== q || q.ended) return
    q.resolvableAtDone = resolvable
    view.askGroundTruth(q.record.name) { truth ->
      if (q.ended) return@askGroundTruth // a measuring tap finished the query while this was open
      if (GroundTruth.measured(truth)) {
        q.truth = truth
        q.measuring = true
        view.enterMeasureMode(truth == GroundTruth.DISPLACED)
      } else {
        endQuery(q, truth, null)
      }
    }
  }

  override fun onResume(owner: LifecycleOwner) = displayRotationHelper.onResume()

  override fun onPause(owner: LifecycleOwner) {
    displayRotationHelper.onPause()
    // The GL thread is already paused (MainView pauses the surface first): save the latest map.
    activity.mapStore.flush()
  }

  override fun onDestroy(owner: LifecycleOwner) {
    activity.mapStore.close()
    scope.cancel()
    detector.close()
    embedder?.close()
  }

  /** Called on the UI thread for a single tap on the camera view (VIEW pixel coordinates). */
  fun onTap(x: Float, y: Float) {
    if (view.lockModeActive) {
      view.toast("Line up the view, then press Lock (or Cancel).")
      return
    }
    val q = query
    if (q != null && !q.measuring) {
      view.toast("Press Done to finish this search before registering new items.")
      return
    }
    pendingTap.set(PendingTap(PointF(x, y), updateTarget))
  }

  // ---------------------------------------------------------------- update location (UI thread)

  /**
   * Items → the item → "Update location": wait for a tap on the item where it is now. The tap runs
   * the normal capture; saving its dialog replaces [rec] (see [saveRegistration]).
   */
  private fun startUpdate(rec: ObjectRecord) {
    if (query != null || updateTarget != null) return
    updateTarget = rec
    view.enterUpdateMode(rec, store.loadBitmap(rec.cropFile))
  }

  /** Back to normal with nothing changed. [stage]: "waiting" (before a tap) or "dialog". */
  private fun cancelUpdate(stage: String) {
    val rec = updateTarget ?: return
    updateTarget = null
    view.exitUpdateMode()
    events.log("update_cancel", mapOf("record" to rec.id, "stage" to stage))
    view.toast("Location of \"${rec.name}\" not changed.")
  }

  // ---------------------------------------------------------------- location requests (UI thread)

  fun requestPickRoom(roomId: String?, newRoom: Room?, source: String) = runOnGl {
    newRoom?.let { localizer.createRoom(it) }
    localizer.pickRoom(roomId, source)
  }

  fun requestRenameRoom(id: String, name: String) = runOnGl { localizer.renameRoom(id, name) }

  fun requestUseSpot(fixId: String) = runOnGl { localizer.startSpotLock(fixId) }

  fun requestAddSpot() = runOnGl {
    pendingGhost.set(null)
    ghostCaptureRequested = true
    localizer.startSpotCapture()
  }

  fun requestCancelLock() = runOnGl {
    ghostCaptureRequested = false
    localizer.cancelLock()
  }

  /** UI thread: save the ghost image (waiting briefly for its conversion), then add the spot on GL. */
  private fun saveSpot(c: Localizer.SpotCapture, name: String) {
    scope.launch(Dispatchers.IO) {
      var ghost = pendingGhost.getAndSet(null)
      var waited = 0
      while (ghost == null && waited < 20) {
        delay(100)
        waited++
        ghost = pendingGhost.getAndSet(null)
      }
      val file = ghost?.let {
        val path = "fix/${c.id}.jpg"
        store.saveJpeg(path, it, 85)
        path
      }
      runOnGl { localizer.addSpot(c, name, file) }
    }
  }

  // ---------------------------------------------------------------- query lifecycle

  /** UI thread: the query starts on the GL thread, where the map and guidance state live. */
  private fun startQuery(rec: ObjectRecord) {
    pendingQuery.set(rec)
  }

  /** GL thread. */
  private fun startQueryGl(rec: ObjectRecord, camera: Camera) {
    val nowMs = SystemClock.elapsedRealtime()
    val wallMs = System.currentTimeMillis()
    val proxies = localizer.proxies(nowMs)
    val inputs =
      PolicyInputs(
        ageMinutes = (wallMs - rec.lastSeenMs) / 60000.0,
        maxSimilarityToOthers = store.maxSimilarityToOthers(rec),
        positionSource = positionSourceFor(rec),
        referenceFullTracking = refFullTracking,
        metersSinceFix = proxies?.metersSinceFix,
        trackingLossesSinceFix = proxies?.trackingLossesSinceFix,
        msSinceFix = proxies?.msSinceFix,
        fixType = proxies?.fixType?.name,
      )
    val decision = VerificationPolicy.decide(inputs)
    val q =
      Query(
        UUID.randomUUID().toString().substring(0, 8),
        rec,
        wallMs,
        inputs,
        decision,
        studyTag = activity.studyTag,
        captureMode = activity.captureMode,
        startElapsedMs = nowMs,
      )
    guard("guidance") { updateGuidance(q, camera, nowMs) }
    q.resolvable = resolve(rec) != null
    query = q
    events.log(
      "query_start",
      mapOf(
        "query" to q.id,
        "record" to rec.id,
        "name" to rec.name,
        "ageMin" to inputs.ageMinutes,
        "maxSimOthers" to inputs.maxSimilarityToOthers,
        "positionSource" to inputs.positionSource.name,
        "refFullTracking" to inputs.referenceFullTracking,
        "refLocked" to refEverLocked,
        "cameraTracking" to cameraTracking.name,
        "policyVerify" to decision.verify,
        "policyReasons" to JSONArray(decision.reasons),
        // v0.3
        "mapId" to localizer.currentMapId,
        "recordMapId" to rec.mapId,
        "located" to localizer.located,
        "itemRoom" to q.itemRoom,
        "itemRoomName" to localizer.roomName(q.itemRoom),
        "userRoom" to q.userRoom,
        "userRoomName" to localizer.roomName(q.userRoom),
        "guidanceMode" to q.mode?.name,
        "nearRule" to q.nearRule,
        "routeM" to q.routeM,
        "straightM" to q.straightM,
        // v0.4
        "mobility" to rec.mobility,
        "lineageId" to rec.lineageId,
        "moveCount" to rec.moveCount,
        "lineageAgeMin" to Lineage.ageMinutes(rec, wallMs),
        "studyTag" to q.studyTag,
        "captureMode" to q.captureMode.name,
      ) + localizer.proxyFields(nowMs)
    )
    val roomName = localizer.roomName(q.itemRoom)
    view.post {
      view.enterGuideMode(
        rec, store.loadBitmap(rec.cropFile), decision, inputs.positionSource != PositionSource.NONE, roomName
      )
    }
  }

  /** UI thread. */
  private fun endQuery(q: Query, truth: String, markerError: JSONObject?) {
    if (q.ended) return
    q.ended = true
    events.log(
      "query_end",
      mapOf(
        "query" to q.id,
        "record" to q.record.id,
        "groundTruth" to truth,
        "verifications" to q.verifications,
        "lastOutcome" to q.lastOutcome,
        "lastSimilarity" to q.lastSimilarity,
        "markerError" to markerError,
        "durationMs" to System.currentTimeMillis() - q.startMs,
        // v0.3
        "nearReached" to (q.timeToNearMs != null),
        "timeToNearMs" to q.timeToNearMs,
        "modesSeen" to JSONArray(q.modesSeen),
        "finalMode" to q.mode?.name,
        // v0.4
        "studyTag" to q.studyTag,
        "captureMode" to q.captureMode.name,
        "captures" to q.captures,
        "noCaptureReason" to CaptureRules.noCaptureReason(q.captures, q.resolvableAtDone),
        "msToFirstVerify" to q.msToFirstVerify,
      )
    )
    query = null
    view.exitGuideMode()
    if (markerError != null) {
      val cm = markerError.getDouble("errorM") * 100
      view.toast(
        if (truth == GroundTruth.DISPLACED) "Displacement: %.0f cm".format(cm) else "Marker error: %.0f cm".format(cm)
      )
    }
  }

  // ---------------------------------------------------------------- GL callbacks

  override fun onSurfaceCreated(render: SampleRender) {
    backgroundRenderer = BackgroundRenderer(render).apply { setUseDepthVisualization(render, false) }
    pointCloudRender.onSurfaceCreated(render)
    labelRender.onSurfaceCreated(render)
  }

  override fun onSurfaceChanged(render: SampleRender?, width: Int, height: Int) {
    displayRotationHelper.onSurfaceChanged(width, height)
    viewWidth = width
    viewHeight = height
  }

  override fun onDrawFrame(render: SampleRender) {
    val session = activity.sessionHelper.session ?: return
    session.setCameraTextureNames(intArrayOf(backgroundRenderer.cameraColorTexture.textureId))
    displayRotationHelper.updateSessionIfNeeded(session)

    val frame =
      try {
        session.update()
      } catch (e: CameraNotAvailableException) {
        Log.e(TAG, "Camera not available during onDrawFrame", e)
        return
      }
    if (firstFrameMs == 0L) firstFrameMs = SystemClock.elapsedRealtime()
    while (true) anchorsToDetach.poll()?.detach() ?: break
    while (true) {
      val cmd = glCommands.poll() ?: break
      try {
        cmd()
      } catch (e: Exception) {
        Log.e(TAG, "GL command failed", e)
      }
    }

    backgroundRenderer.updateDisplayGeometry(frame)
    backgroundRenderer.drawBackground(render)

    val camera = frame.camera
    camera.getViewMatrix(viewMatrix, 0)
    camera.getProjectionMatrix(projectionMatrix, 0, 0.01f, 100.0f)
    Matrix.multiplyMM(viewProjectionMatrix, 0, projectionMatrix, 0, viewMatrix, 0)
    cameraTracking = camera.trackingState

    val image = updateReference(session)
    val nowMs = SystemClock.elapsedRealtime()
    guard("localizer") { localizer.onFrame(nowMs, session, camera, image) }
    pendingQuery.getAndSet(null)?.let { startQueryGl(it, camera) }
    // v0.4: whether a capture is possible right now, for the Done prompts (also while not tracking).
    query?.let { q -> guard("resolvable") { q.resolvable = resolve(q.record) != null } }

    if (camera.trackingState == TrackingState.TRACKING) {
      frame.acquirePointCloud().use { pointCloudRender.drawPointCloud(render, it, viewProjectionMatrix) }
      handleTap(frame, camera, session)
      handleVerify(frame, camera, session)
      guard("ghost") { captureGhostIfRequested(frame, session) }
      query?.let { q -> guard("guidance") { updateGuidance(q, camera, nowMs, throttle = true) } }
      drawLabels(render, camera)
      updateOverlay(camera)
    } else {
      pendingTap.set(null)
      hideOverlay()
      if (pendingVerify) {
        pendingVerify = false
        view.post { view.toast("Tracking lost: move the phone slowly, then press Verify again.") }
      }
    }

    guard("publishUi") { publishUi(frame, camera) }
  }

  private val lastGuardLogMs = HashMap<String, Long>()

  /**
   * Runs v0.3 map code so that a bug there is logged (at most every 5 s per site) instead of
   * stopping the AR loop in the middle of a study session. GL thread.
   */
  private inline fun guard(site: String, block: () -> Unit) {
    try {
      block()
    } catch (e: Exception) {
      val now = SystemClock.elapsedRealtime()
      if (now - (lastGuardLogMs[site] ?: -10_000L) >= 5000L) {
        lastGuardLogMs[site] = now
        Log.e(TAG, "Error in $site", e)
        events.log("internal_error", mapOf("site" to site, "error" to e.toString()))
      }
    }
  }

  // ---------------------------------------------------------------- reference image

  /** Tracks the reference picture's flags and events; returns its trackable for the localizer. */
  private fun updateReference(session: Session): AugmentedImage? {
    val img =
      session.getAllTrackables(AugmentedImage::class.java).firstOrNull {
        it.name == MainActivity.REF_NAME && it.trackingState == TrackingState.TRACKING
      }
    if (img == null) {
      refFullTracking = false
      return null
    }
    val full = img.trackingMethod == AugmentedImage.TrackingMethod.FULL_TRACKING
    refFullTracking = full
    if (full && !refEverLocked) {
      refEverLocked = true
      events.log(
        "reference_locked",
        mapOf(
          "sinceFirstFrameMs" to SystemClock.elapsedRealtime() - firstFrameMs,
          "extentX" to img.extentX,
          "extentZ" to img.extentZ,
        )
      )
      val useful = localizer.pictureUseful()
      view.post {
        view.toast(
          if (useful) "Picture found: hold the phone still near it for a moment"
          else "Picture found, but it is not a fix point of this map (see the location sheet)"
        )
      }
    }
    return img
  }

  private fun positionSourceFor(rec: ObjectRecord): PositionSource = resolve(rec)?.second ?: PositionSource.NONE

  private fun sessionAnchorTracking(rec: ObjectRecord): Boolean =
    sessionAnchors[rec.id]?.trackingState == TrackingState.TRACKING

  /**
   * Current-session world position for a record: a tracking same-session anchor, else the record's
   * map position through T_WM when located in the record's map, else null. GL thread.
   */
  private fun resolve(rec: ObjectRecord): Pair<FloatArray, PositionSource>? {
    sessionAnchors[rec.id]?.let { a ->
      if (a.trackingState == TrackingState.TRACKING) return a.pose.translation to PositionSource.SESSION
    }
    val t = localizer.tWM
    val mp = rec.mapPosition
    if (t != null && mp != null && rec.mapId != null && rec.mapId == localizer.currentMapId) {
      return t.transformPoint(mp) to PositionSource.MAP
    }
    return null
  }

  /** The item's room: as registered, else the room of the walk graph near its map position. */
  private fun itemRoomOf(rec: ObjectRecord): String? {
    rec.roomId?.let { return it }
    val mp = rec.mapPosition ?: return null
    if (rec.mapId != localizer.currentMapId) return null
    return localizer.graph.roomAt(Vec3.of(mp))
  }

  private fun userRoom(): String? = localizer.graph.currentRoom ?: localizer.userMap?.let { localizer.graph.roomAt(it) }

  // ---------------------------------------------------------------- hit testing

  private fun usableHit(hit: HitResult): Boolean =
    when (val t = hit.trackable) {
      is Plane -> t.isPoseInPolygon(hit.hitPose)
      is DepthPoint -> true
      is Point -> true
      else -> false
    }

  /** Prefer dense depth or plane hits; sparse feature points are the least reliable. */
  private fun bestHit(frame: Frame, x: Float, y: Float): HitResult? {
    val hits = frame.hitTest(x, y).filter { usableHit(it) }
    return hits.firstOrNull { it.trackable is DepthPoint || it.trackable is Plane } ?: hits.firstOrNull()
  }

  // ---------------------------------------------------------------- registration and measurement

  private fun handleTap(frame: Frame, camera: Camera, session: Session) {
    val tap = pendingTap.getAndSet(null) ?: return
    val q = query
    if (q != null) {
      if (q.measuring) handleMeasureTap(frame, q, tap.point)
      return
    }
    // v0.4 "Update location": a tap made in that mode replaces the record it was made for.
    val updateOf = tap.updateOf
    if (updateOf != null && updateTarget !== updateOf) return // cancelled before the tap arrived
    if (mlBusy) return view.post { view.toast("Still processing the previous capture…") }
    val emb = embedder ?: return view.post { view.toast("Model still loading…") }

    val hit =
      bestHit(frame, tap.point.x, tap.point.y)
        ?: return view.post {
          view.toast("No surface found there. Move the phone slowly to map the area, then tap again.")
        }
    val image =
      frame.tryAcquireCameraImage() ?: return view.post { view.toast("Camera image not ready; tap again.") }

    val imgPt = FloatArray(2)
    frame.transformCoordinates2d(
      Coordinates2d.VIEW, floatArrayOf(tap.point.x, tap.point.y), Coordinates2d.IMAGE_PIXELS, imgPt
    )
    val hitPos = hit.hitPose.translation
    val distance = dist(hitPos, camera.pose.translation).coerceAtLeast(0.05f)
    if (distance > MAX_HIT_DISTANCE_M) {
      image.close()
      events.log("hit_rejected", mapOf("distance" to distance, "hitType" to hit.trackable.javaClass.simpleName))
      return view.post {
        view.toast("Depth looks wrong (%.0f m). Move the phone sideways a little, then tap again.".format(distance))
      }
    }
    // v0.3: the remembered point is stored in the map frame (no longer relative to the picture).
    val toMap = localizer.tWM?.inverse()
    val mapPos = toMap?.transformPoint(hitPos)
    val viewPos = toMap?.transformPoint(camera.pose.translation)
    val defaultRoom = localizer.graph.currentRoom ?: viewPos?.let { localizer.graph.roomAt(Vec3.of(it)) }
    val mapId = if (toMap != null) localizer.currentMapId else null
    val drift = localizer.proxyFields(SystemClock.elapsedRealtime())
    val hasMap = localizer.currentMapId != null
    val anchor = hit.createAnchor()
    val hitType = hit.trackable.javaClass.simpleName
    val rotation = displayRotationHelper.getCameraSensorToDisplayRotation(session.cameraConfig.cameraId)
    val intrinsics = camera.imageIntrinsics
    val fxNative = intrinsics.focalLength[0]
    val intrWidth = intrinsics.imageDimensions[0]

    mlBusy = true
    scope.launch(Dispatchers.Default) {
      try {
        val t0 = SystemClock.elapsedRealtimeNanos()
        val cf = try { CameraFrame.from(image, rotation) } finally { image.close() }
        val convertMs = msSince(t0)

        val t1 = SystemClock.elapsedRealtimeNanos()
        val dets = detector.detect(cf)
        val detectMs = msSince(t1)

        // Pixels per metre at the tapped depth, from the camera intrinsics.
        val fx = fxNative * cf.width / intrWidth
        val pxPerM = fx / distance
        val frameArea = cf.width.toFloat() * cf.height
        val containing = dets.filter { it.sensorRect.contains(imgPt[0], imgPt[1]) }
        val plausible =
          containing.filter {
            val r = it.sensorRect
            r.width() * r.height() <= MAX_BOX_FRACTION * frameArea && max(r.width(), r.height()) / pxPerM <= MAX_OBJECT_M
          }
        val chosen: Detection? = plausible.minByOrNull { it.sensorRect.width() * it.sensorRect.height() }
        val rect =
          cf.clamp(
            if (chosen != null) CameraFrame.pad(chosen.sensorRect, 0.08f)
            else
              CameraFrame.square(
                imgPt[0],
                imgPt[1],
                (DEFAULT_WINDOW_M * pxPerM).coerceIn(160f, 0.6f * min(cf.width, cf.height))
              )
          )
        val boxSource =
          when {
            chosen != null -> "detector"
            containing.isNotEmpty() -> "tap_window_large_box_rejected"
            else -> "tap_window"
          }
        val crop = cf.cropUpright(rect)

        val t2 = SystemClock.elapsedRealtimeNanos()
        val embedding = emb.embed(crop)
        val embedMs = msSince(t2)

        val draft =
          RegistrationDraft(
            id = UUID.randomUUID().toString().substring(0, 8),
            crop = crop,
            embedding = embedding,
            rawLabel = chosen?.label ?: "",
            rawConfidence = chosen?.confidence ?: 0f,
            boxSource = boxSource,
            cropRect = rect,
            worldPosition = hitPos,
            captureDistance = distance,
            imageWidth = cf.width,
            imageHeight = cf.height,
            hitType = hitType,
            anchor = anchor,
            detections = dets.size,
            rejectedBoxes = containing.size - plausible.size,
            fx = fx,
            latencies = mapOf("convertMs" to convertMs, "detectMs" to detectMs, "embedMs" to embedMs),
            mapId = mapId,
            mapPosition = mapPos,
            viewPosition = viewPos,
            defaultRoomId = defaultRoom,
            drift = drift,
            hasMap = hasMap,
            updateOf = updateOf,
          )
        withContext(Dispatchers.Main) {
          if (updateOf != null && updateTarget !== updateOf) {
            // "Update location" was cancelled while the capture ran: drop it, nothing changes.
            anchorsToDetach.add(draft.anchor)
            return@withContext
          }
          view.showRegisterDialog(
            draft,
            defaultName = updateOf?.name ?: nextDefaultName(),
            rooms = uiState?.map?.rooms ?: emptyList(),
            update = updateOf,
            onSave = { name, typed, roomId, picked, newRoom, mobility ->
              if (picked) requestPickRoom(roomId, newRoom, "register")
              if (updateOf != null) {
                // Leave the waiting mode now; the record is replaced by the save below.
                updateTarget = null
                view.exitUpdateMode()
              }
              saveRegistration(
                draft, name, typed, roomId, newRoom?.name ?: uiState?.roomName(roomId), mobility, updateOf
              )
            },
            onCancel = {
              anchorsToDetach.add(draft.anchor)
              events.log(
                "register_cancel",
                mapOf("draft" to draft.id, "boxSource" to draft.boxSource, "updateOf" to updateOf?.id)
              )
              if (updateOf != null) cancelUpdate("dialog")
            }
          )
        }
      } catch (e: Exception) {
        Log.e(TAG, "Registration capture failed", e)
        anchorsToDetach.add(anchor)
        view.post { view.toast("Capture failed: ${e.message}") }
      } finally {
        mlBusy = false
      }
    }
  }

  private fun nextDefaultName(): String {
    val records = store.snapshot()
    val names = records.map { it.name }.toSet()
    var n = records.size + 1
    while ("item $n" in names) n++
    return "item $n"
  }

  /**
   * Store the confirmed capture. With [old] (v0.4 "Update location") the new record takes the old
   * one's place in the active list and continues its lineage; the old crop and logs are kept.
   */
  private fun saveRegistration(
    d: RegistrationDraft,
    name: String,
    typed: Boolean,
    roomId: String?,
    roomName: String?,
    mobility: String,
    old: ObjectRecord?,
  ) {
    scope.launch(Dispatchers.IO) {
      val cropFile = "crops/${d.id}.jpg"
      val cropBytes = store.saveJpeg(cropFile, d.crop)
      val nowMs = System.currentTimeMillis()
      val lineage = if (old != null) Lineage.successor(old) else Lineage.first(d.id, nowMs)
      val rec =
        ObjectRecord(
          id = d.id,
          name = name,
          rawLabel = d.rawLabel,
          rawConfidence = d.rawConfidence,
          boxSource = d.boxSource,
          cropFile = cropFile,
          embedding = d.embedding,
          refPosition = null, // v0.3: no longer written; positions live in the map frame
          worldPosition = d.worldPosition,
          sessionId = activity.sessionId,
          captureDistance = d.captureDistance,
          cropWidthPx = d.cropRect.width(),
          cropHeightPx = d.cropRect.height(),
          imageWidth = d.imageWidth,
          imageHeight = d.imageHeight,
          hitType = d.hitType,
          lastSeenMs = nowMs,
          mapId = d.mapId,
          mapPosition = d.mapPosition,
          viewPosition = d.viewPosition,
          roomId = roomId,
          mobility = mobility,
          lineageId = lineage.lineageId,
          supersedes = lineage.supersedes,
          moveCount = lineage.moveCount,
          lineageStartMs = lineage.lineageStartMs,
        )
      if (old != null) {
        store.supersede(old.id, rec)
        sessionAnchors.remove(old.id)?.let { anchorsToDetach.add(it) }
      } else {
        store.add(rec)
      }
      sessionAnchors[rec.id] = d.anchor
      events.log(
        "register",
        mapOf(
          "record" to rec.id,
          "userName" to name,
          "nameSource" to if (typed) "typed" else if (old != null) "kept" else "default",
          "rawLabel" to d.rawLabel,
          "rawConfidence" to d.rawConfidence,
          "boxSource" to d.boxSource,
          "detections" to d.detections,
          "rejectedBoxes" to d.rejectedBoxes,
          "hitType" to d.hitType,
          "distance" to d.captureDistance,
          "fx" to d.fx,
          "cropPx" to JSONArray(listOf(d.cropRect.width().toDouble(), d.cropRect.height().toDouble())),
          "refAvailable" to false, // v0.3 stores no reference-frame position
          "refWidthM" to activity.refWidthM,
          "refPosition" to null,
          "imageSize" to "${d.imageWidth}x${d.imageHeight}",
          "cropBytes" to cropBytes,
          "recordBytes" to store.bytesFor(rec),
          "embeddingDim" to d.embedding.size,
          "latency" to JSONObject(d.latencies),
          // v0.3
          "mapId" to d.mapId,
          "mapPosition" to d.mapPosition?.let { ObjectRecord.floats(it) },
          "viewPosition" to d.viewPosition?.let { ObjectRecord.floats(it) },
          "room" to roomId,
          "roomName" to roomName,
          "located" to (d.mapPosition != null),
          // v0.4
          "mobility" to rec.mobility,
          "lineageId" to rec.lineageId,
          "supersedes" to rec.supersedes,
          "moveCount" to rec.moveCount,
        ) + d.drift + activity.studyFields()
      )
      if (old != null) {
        // Distance between the old and the new place, when both are positions in the current map.
        val oldMap = old.mapPosition
        val newMap = d.mapPosition
        val moved =
          if (oldMap != null && newMap != null && d.mapId != null && old.mapId == d.mapId) dist(oldMap, newMap) else null
        events.log(
          "supersede",
          mapOf(
            "oldRecord" to old.id,
            "newRecord" to rec.id,
            "name" to name,
            "lineageId" to rec.lineageId,
            "moveCount" to rec.moveCount,
            "oldAgeMin" to (nowMs - old.lastSeenMs) / 60000.0,
            "distanceM" to moved,
            "oldRoom" to old.roomId,
            "newRoom" to roomId,
          ) + activity.studyFields()
        )
      }
      view.post {
        view.toast(
          (if (old != null) "Updated location of \"$name\"" else "Saved \"$name\"") +
            if (d.mapPosition == null) " (not located: valid this session only)"
            else roomName?.let { " in $it" } ?: ""
        )
      }
    }
  }

  /**
   * Ground truth "present" or "displaced" follow-up: the user taps the object's true centre. The
   * distance to the marker is the marker error ("present") or the displacement ("displaced").
   */
  private fun handleMeasureTap(frame: Frame, q: Query, tap: PointF) {
    val hit = bestHit(frame, tap.x, tap.y) ?: return view.post { view.toast("No surface there; tap again or Skip.") }
    val (marker, source) =
      resolve(q.record) ?: return view.post { view.toast("No marker to compare against; tap Skip.") }
    val truthValue = q.truth ?: GroundTruth.PRESENT
    val truth = hit.hitPose.translation
    if (dist(truth, frame.camera.pose.translation) > MAX_HIT_DISTANCE_M) {
      return view.post { view.toast("Depth looks wrong there. Move the phone a little and tap again.") }
    }
    val dx = truth[0] - marker[0]
    val dy = truth[1] - marker[1]
    val dz = truth[2] - marker[2]
    val err =
      JSONObject().apply {
        put("errorM", sqrt(dx * dx + dy * dy + dz * dz).toDouble())
        put("horizontalErrorM", sqrt(dx * dx + dz * dz).toDouble()) // ARCore world +Y is up
        put("verticalErrorM", abs(dy).toDouble())
        put("positionSource", source.name)
        put("hitType", hit.trackable.javaClass.simpleName)
        // v0.3: drift proxies at the measurement.
        for ((k, v) in localizer.proxyFields(SystemClock.elapsedRealtime())) put(k, v ?: JSONObject.NULL)
        // v0.4: true when this is the distance of a displaced item, not the marker's own error.
        put("displaced", truthValue == GroundTruth.DISPLACED)
      }
    q.measuring = false
    view.post { endQuery(q, truthValue, err) }
  }

  // ---------------------------------------------------------------- verification

  private fun handleVerify(frame: Frame, camera: Camera, session: Session) {
    if (!pendingVerify) return
    pendingVerify = false
    val q = query ?: return
    if (q.measuring) return
    if (!CaptureRules.verifyAllowed(q.captureMode, q.captures)) {
      // The button is disabled after the capture; this covers a press that was already on its way.
      return view.post { view.toast("PAIRED mode: this query already has its capture. Press Done.") }
    }
    if (mlBusy) return view.post { view.toast("Still processing…") }
    val emb = embedder ?: return view.post { view.toast("Model still loading…") }
    val (world, source) =
      resolve(q.record)
        ?: return view.post {
          view.toast("No spatial cue yet. Go to a fix point (picture or spot) so I can locate you.")
        }

    val clip = FloatArray(4)
    Matrix.multiplyMV(clip, 0, viewProjectionMatrix, 0, floatArrayOf(world[0], world[1], world[2], 1f), 0)
    if (clip[3] <= 0f) {
      return view.post { view.toast("The remembered spot is behind you. Follow the arrow, then press Verify.") }
    }
    val ndc = floatArrayOf(clip[0] / clip[3], clip[1] / clip[3])
    if (abs(ndc[0]) > 0.9f || abs(ndc[1]) > 0.9f) {
      return view.post { view.toast("Bring the ring into view, then press Verify.") }
    }
    val image = frame.tryAcquireCameraImage() ?: return view.post { view.toast("Camera image not ready; try again.") }
    val px = FloatArray(2)
    frame.transformCoordinates2d(Coordinates2d.OPENGL_NORMALIZED_DEVICE_COORDINATES, ndc, Coordinates2d.IMAGE_PIXELS, px)
    val d1 = dist(world, camera.pose.translation).coerceAtLeast(0.1f)
    val rotation = displayRotationHelper.getCameraSensorToDisplayRotation(session.cameraConfig.cameraId)
    val refFull = refFullTracking
    val drift = localizer.proxyFields(SystemClock.elapsedRealtime()) + mapOf("guidanceMode" to q.mode?.name)
    val attempt = ++q.verifications
    // v0.4 (S8): when this capture was taken, on the monotonic clock.
    val captureMs = SystemClock.elapsedRealtime()
    val msSinceQueryStart = captureMs - q.startElapsedMs
    val timing: Map<String, Any?> =
      mapOf("msSinceQueryStart" to msSinceQueryStart, "msSinceNear" to q.nearElapsedMs?.let { captureMs - it })

    mlBusy = true
    q.capturing = true
    scope.launch(Dispatchers.Default) {
      try {
        val tStart = SystemClock.elapsedRealtimeNanos()
        val cf = try { CameraFrame.from(image, rotation) } finally { image.close() }
        val convertMs = msSince(tStart)
        val rec = q.record

        // Expected crop size: the registration crop rescaled by distance and image resolution,
        // i.e. the same physical window around the remembered point.
        val scale = (rec.captureDistance / d1) * (cf.width.toFloat() / rec.imageWidth)
        val cw = (rec.cropWidthPx * scale).coerceIn(64f, cf.width.toFloat())
        val ch = (rec.cropHeightPx * scale).coerceIn(64f, cf.height.toFloat())
        val projected = cf.clamp(RectF(px[0] - cw / 2, px[1] - ch / 2, px[0] + cw / 2, px[1] + ch / 2))
        val expectedArea = cw * ch

        val t1 = SystemClock.elapsedRealtimeNanos()
        val dets = detector.detect(cf)
        val detectMs = msSince(t1)

        // Candidates: the projected window plus detector boxes that overlap it and have a
        // plausible size (scene-sized boxes would let background alone produce a match).
        val boxCandidates =
          dets
            .filter { RectF.intersects(it.sensorRect, projected) }
            .filter {
              val ratio = it.sensorRect.width() * it.sensorRect.height() / expectedArea
              ratio in 0.25f..4f
            }
            .map { "detector" to cf.clamp(CameraFrame.pad(it.sensorRect, 0.08f)) }
        val candidates = listOf("projected" to projected) + boxCandidates

        val t2 = SystemClock.elapsedRealtimeNanos()
        // v0.4 (S5): keep each candidate's embedding as computed for its score (same order).
        val embeddings = ArrayList<FloatArray>(candidates.size)
        val scored =
          candidates.map { (kind, r) ->
            val crop = cf.cropUpright(r)
            val e = emb.embed(crop)
            embeddings.add(e)
            Triple(kind, r, crop) to Embedder.cosine(e, rec.embedding)
          }
        val embedMs = msSince(t2)
        val best = scored.maxBy { it.second }
        val similarity = best.second
        val confirmed = similarity >= VerificationPolicy.MATCH_THRESHOLD
        val outcome = if (confirmed) "CONFIRMED" else "INSUFFICIENT_EVIDENCE"
        val totalMs = msSince(tStart)

        // Keep the paired fresh evidence so conditions and thresholds can be re-run offline.
        val vid = "${q.id}_$attempt"
        store.saveJpeg("verify/${vid}_frame.jpg", cf.upright, 85)
        store.saveJpeg("verify/${vid}_best.jpg", best.first.third)
        val embFile =
          saveEmbeddings(
            "verify/${vid}_emb.json",
            emb.modelName,
            rec.id,
            q.id,
            attempt,
            scored.mapIndexed { i, (c, s) ->
              val r = c.second
              EmbeddingFile.Candidate(c.first, floatArrayOf(r.left, r.top, r.right, r.bottom), s, embeddings[i])
            },
          )
        val up = cf.sensorToUpright(px[0], px[1])

        q.lastOutcome = outcome
        q.lastSimilarity = similarity
        events.log(
          "verify",
          mapOf(
            "query" to q.id,
            "record" to rec.id,
            "attempt" to attempt,
            "positionSource" to source.name,
            "refFullTracking" to refFull,
            "distance" to d1,
            "projectedUpright" to JSONArray(listOf(up.x.toDouble(), up.y.toDouble())),
            "rotation" to rotation,
            "detectorBoxes" to dets.size,
            "scores" to
              JSONArray(
                scored.map { (c, s) ->
                  JSONObject().apply {
                    put("kind", c.first)
                    put("sim", s.toDouble())
                    put("rect", JSONArray(listOf(c.second.left, c.second.top, c.second.right, c.second.bottom).map { it.toDouble() }))
                  }
                }
              ),
            "bestSimilarity" to similarity,
            "bestKind" to best.first.first,
            "threshold" to VerificationPolicy.MATCH_THRESHOLD,
            "outcome" to outcome,
            "inferenceCalls" to 1 + candidates.size,
            "latency" to
              JSONObject(
                mapOf("convertMs" to convertMs, "detectMs" to detectMs, "embedMs" to embedMs, "totalMs" to totalMs)
              ),
            "frameFile" to "verify/${vid}_frame.jpg",
            "cropFile" to "verify/${vid}_best.jpg",
          ) + drift + // v0.3: drift proxies at capture
            timing + // v0.4
            mapOf("embFile" to embFile, "studyTag" to q.studyTag, "captureMode" to q.captureMode.name)
        )
        if (q.msToFirstVerify == null) q.msToFirstVerify = msSinceQueryStart
        q.captures++ // only this capture writes it: mlBusy admits one capture at a time
        val storedThumb = store.loadBitmap(rec.cropFile)
        withContext(Dispatchers.Main) {
          if (query === q) {
            view.setVerifyLine(outcome, similarity)
            if (!CaptureRules.verifyAllowed(q.captureMode, q.captures)) view.verifyButton.isEnabled = false
          }
          view.showVerifyResult(storedThumb, best.first.third, similarity, confirmed) {
            // v0.4 BENCHMARK: once the result is dismissed, the experimenter labels the capture.
            if (q.captureMode == CaptureMode.BENCHMARK) askVerifyLabel(q, rec.id, attempt)
          }
        }
      } catch (e: Exception) {
        Log.e(TAG, "Verification failed", e)
        view.post { view.toast("Verification failed: ${e.message}") }
      } finally {
        mlBusy = false
        q.capturing = false
      }
    }
  }

  /**
   * S5: write the candidates' embeddings next to the verification frame. Returns the path for the
   * `verify` event, or null when writing failed (the verification itself still counts).
   */
  private fun saveEmbeddings(
    path: String,
    model: String,
    recordId: String,
    queryId: String,
    attempt: Int,
    candidates: List<EmbeddingFile.Candidate>,
  ): String? =
    try {
      store.saveText(path, EmbeddingFile.toJson(model, recordId, queryId, attempt, candidates).toString())
      path
    } catch (e: Exception) {
      Log.e(TAG, "Could not write $path", e)
      null
    }

  /** UI thread, BENCHMARK: exactly one `verify_label` per capture, also when the dialog is dismissed. */
  private fun askVerifyLabel(q: Query, recordId: String, attempt: Int) {
    if (activity.isFinishing || activity.isDestroyed) return // no window to show the dialog in
    view.askVerifyLabel(q.record.name) { label, condition ->
      events.log(
        "verify_label",
        mapOf(
          "query" to q.id,
          "record" to recordId,
          "attempt" to attempt,
          "label" to label,
          "condition" to condition,
          "studyTag" to q.studyTag,
          "captureMode" to q.captureMode.name,
        )
      )
    }
  }

  // ---------------------------------------------------------------- spots

  /** "Add a spot here": keep one CPU frame of the pose as the ghost image (upright, half size). */
  private fun captureGhostIfRequested(frame: Frame, session: Session) {
    if (!ghostCaptureRequested) return
    // Any acquire failure (not ready, or images still held by a capture in progress): next frame.
    val image =
      try {
        frame.acquireCameraImage()
      } catch (e: Exception) {
        return
      }
    ghostCaptureRequested = false
    val rotation = displayRotationHelper.getCameraSensorToDisplayRotation(session.cameraConfig.cameraId)
    scope.launch(Dispatchers.Default) {
      try {
        val cf = try { CameraFrame.from(image, rotation) } finally { image.close() }
        val up = cf.upright
        val w = max(1, (up.width * GHOST_SCALE).toInt())
        val h = max(1, (up.height * GHOST_SCALE).toInt())
        pendingGhost.set(Bitmap.createScaledBitmap(up, w, h, true))
      } catch (e: Exception) {
        Log.e(TAG, "Ghost capture failed", e)
      }
    }
  }

  // ---------------------------------------------------------------- guidance (GL thread)

  /** Plan guidance for [q]; with [throttle], at most every [UI_UPDATE_MS] (about 5 Hz). */
  private fun updateGuidance(q: Query, camera: Camera, nowMs: Long, throttle: Boolean = false) {
    if (throttle && nowMs - q.lastGuidanceMs < UI_UPDATE_MS) return
    q.lastGuidanceMs = nowMs
    val rec = q.record
    val r = resolve(rec)
    val cam = Vec3.of(camera.pose.translation)
    val straight = r?.let { horizontalDistance(cam, Vec3.of(it.first)) }
    val t = localizer.tWM
    val user = localizer.userMap
    val targetMap = if (t != null && r != null) t.inverse().transformPoint(Vec3.of(r.first)) else null
    val viewPos = rec.viewPosition
    val routeEnd =
      if (viewPos != null && rec.mapId == localizer.currentMapId) Vec3.of(viewPos) else targetMap
    if (t != null && user != null && routeEnd != null && r != null) {
      if (GuidancePlanner.shouldReplan(nowMs, q.lastPlanMs, q.route, user)) {
        q.route = localizer.graph.route(user, routeEnd)
        q.lastPlanMs = nowMs
      }
    } else {
      q.route = null
    }
    val route = q.route
    val remaining = if (route != null && user != null) route.remainingFrom(user) else null
    val itemRoom = itemRoomOf(rec)
    val userRoom = userRoom()
    val d =
      GuidancePlanner.decide(
        GuidanceInput(r != null, t != null, sessionAnchorTracking(rec), straight, remaining, userRoom, itemRoom),
        q.mode,
      )
    q.waypointWorld =
      if (d.mode == GuidanceMode.FAR_ROUTE && route != null && user != null && t != null) {
        val w = route.pointAhead(user, GuidancePlanner.WAYPOINT_AHEAD_M)
        t.transformPoint(Vec3(w.x, user.y, w.z)).toFloatArray() // at camera height
      } else null
    q.mode = d.mode
    q.nearRule = d.nearRule
    q.routeM = remaining
    q.straightM = straight
    q.itemRoom = itemRoom
    q.userRoom = userRoom
    if (d.mode.name !in q.modesSeen) q.modesSeen = q.modesSeen + d.mode.name
    if (d.mode == GuidanceMode.NEAR && q.timeToNearMs == null) q.timeToNearMs = System.currentTimeMillis() - q.startMs
    if (d.mode == GuidanceMode.NEAR && q.nearElapsedMs == null) q.nearElapsedMs = nowMs
  }

  // ---------------------------------------------------------------- drawing and UI

  private fun drawLabels(render: SampleRender, camera: Camera) {
    val q = query
    val records =
      if (q != null) {
        // During a query, only the target's label, and only when near.
        if (q.mode == GuidanceMode.NEAR) listOf(q.record) else emptyList()
      } else {
        val ids = labelIds
        store.snapshot().filter { it.id in ids }
      }
    for (rec in records) {
      val (pos, _) = resolve(rec) ?: continue
      labelRender.draw(
        render,
        viewProjectionMatrix,
        Pose.makeTranslation(pos[0], pos[1], pos[2]),
        camera.pose,
        rec.name,
        highlight = q != null
      )
    }
  }

  /** Outside a query: items within [LABEL_MAX_M] that are not known to be in another room. */
  private fun computeLabelIds(camera: Camera): Set<String> {
    val cam = Vec3.of(camera.pose.translation)
    val userRoom = userRoom()
    val out = HashSet<String>()
    for (rec in store.snapshot()) {
      val (pos, _) = resolve(rec) ?: continue
      if (horizontalDistance(cam, Vec3.of(pos)) > LABEL_MAX_M) continue
      val itemRoom = itemRoomOf(rec)
      if (userRoom != null && itemRoom != null && userRoom != itemRoom) continue
      out.add(rec.id)
    }
    return out
  }

  /** Push the guidance target's screen position to the 2D overlay (every frame while guiding). */
  private fun updateOverlay(camera: Camera) {
    val q = query
    if (q == null || q.measuring) return hideOverlay()
    val world: FloatArray
    val kind: GuidanceOverlay.Kind
    val label: String?
    when (q.mode) {
      GuidanceMode.NEAR -> {
        world = resolve(q.record)?.first ?: return hideOverlay()
        kind = GuidanceOverlay.Kind.RING
        label = null
      }
      GuidanceMode.FAR_ROUTE -> {
        world = q.waypointWorld ?: return hideOverlay()
        kind = GuidanceOverlay.Kind.CHEVRON
        label = q.routeM?.let { "%.0f m walk".format(it) }
      }
      else -> return hideOverlay()
    }
    val clip = FloatArray(4)
    Matrix.multiplyMV(clip, 0, viewProjectionMatrix, 0, floatArrayOf(world[0], world[1], world[2], 1f), 0)
    val inFront = clip[3] > 0f
    val local = camera.displayOrientedPose.inverse().transformPoint(world)
    view.overlay.setTarget(
      GuidanceOverlay.Target(
        ndcX = if (inFront) clip[0] / clip[3] else 0f,
        ndcY = if (inFront) clip[1] / clip[3] else 0f,
        inFront = inFront,
        localX = local[0],
        localY = local[1],
        distance = sqrt(local[0] * local[0] + local[1] * local[1] + local[2] * local[2]),
        state =
          when (q.lastOutcome) {
            null -> GuidanceOverlay.State.UNVERIFIED
            "CONFIRMED" -> GuidanceOverlay.State.CONFIRMED
            else -> GuidanceOverlay.State.NOT_CONFIRMED
          },
        kind = kind,
        label = label,
      )
    )
    overlayShown = true
  }

  private fun hideOverlay() {
    if (!overlayShown) return
    overlayShown = false
    view.overlay.setTarget(null)
  }

  private fun publishUi(frame: Frame, camera: Camera) {
    val now = SystemClock.elapsedRealtime()
    if (now - lastUiUpdateMs < UI_UPDATE_MS) return
    lastUiUpdateMs = now

    val st = localizer.state()
    uiState = st
    val gm = ghostMapping
    if (viewWidth > 0 && viewHeight > 0 && (gm == null || gm.viewWidth != viewWidth || gm.viewHeight != viewHeight)) {
      val corners = FloatArray(6)
      frame.transformCoordinates2d(
        Coordinates2d.VIEW,
        floatArrayOf(0f, 0f, viewWidth.toFloat(), 0f, 0f, viewHeight.toFloat()),
        Coordinates2d.IMAGE_NORMALIZED,
        corners,
      )
      val session = activity.sessionHelper.session
      val rotation = session?.let { displayRotationHelper.getCameraSensorToDisplayRotation(it.cameraConfig.cameraId) } ?: 90
      ghostMapping = GhostMapping(corners, rotation, viewWidth, viewHeight)
    }

    val tracking = camera.trackingState == TrackingState.TRACKING
    val trackingDetail =
      if (tracking) "" else camera.trackingFailureReason.name.lowercase().replace('_', ' ')
    val q = query
    val upd = updateTarget
    if (q == null && tracking) labelIds = computeLabelIds(camera)
    val items = store.snapshot().size
    val located = q?.let { resolve(it.record) }
    val itemRoomName = q?.let { localizer.roomName(it.itemRoom) }
    val hint =
      when {
        mlBusy -> "Analyzing…"
        q != null && q.measuring -> "Tap the true centre of \"${q.record.name}\""
        q != null ->
          when (q.mode) {
            GuidanceMode.NO_FIX ->
              if (st.pictureUsable) "Go to a fix point so I can locate you (picture, or tap here → Use a spot)"
              else "Go to a fix point so I can locate you (tap here → Use a spot)"
            GuidanceMode.FAR_ROUTE -> "Follow the chevron along your walked path"
            GuidanceMode.FAR_NO_ROUTE ->
              if (st.located) "No walked path yet: use the map (top right)"
              else "Far away and not located: go to a fix point for a route"
            else -> "Find \"${q.record.name}\": follow the arrow to the ring"
          }
        st.lockActive -> "Hold still…"
        !tracking -> "Move the phone slowly to resume tracking"
        upd != null -> "Update location: tap \"${upd.name}\" where it is now, or press Cancel"
        !st.hasMap && st.setHomeDeclined -> "No home map yet: tap here → Set home here with a spot (or restart to use the picture)"
        !st.hasMap ->
          "No home map yet: show the picture (%.1f cm) or tap here → Add a spot to set home".format(activity.refWidthM * 100)
        !st.located && st.pictureUsable -> "Not located: look at the picture, or tap here → Use a spot"
        !st.located && st.imageFixNote != null -> "Not located: picture ignored (width changed). Tap here → Use a spot"
        !st.located -> "Not located: tap here → Use a spot"
        st.currentRoomId == null -> "Located. Pick this room with the room button, then walk the rooms and tap objects"
        items == 0 -> "Walk through the rooms, then tap an object to remember it"
        else -> "Tap an object to remember it · Items to find one"
      }

    val cue =
      if (q == null || q.measuring) null
      else {
        val r = located
        if (q.mode != GuidanceMode.NEAR || r == null) {
          GuidancePlanner.cue(q.mode ?: GuidanceMode.NO_FIX, itemRoomName, q.routeM, q.straightM, st.located)
        } else {
          val local = camera.displayOrientedPose.inverse().transformPoint(r.first)
          val distance = sqrt(local[0] * local[0] + local[1] * local[1] + local[2] * local[2])
          val angle = Math.toDegrees(atan2(local[0], -local[2]).toDouble())
          val dir =
            when {
              abs(angle) <= 15 -> "straight ahead"
              abs(angle) >= 135 -> "behind you"
              angle > 0 -> "turn right %.0f°".format(angle)
              else -> "turn left %.0f°".format(-angle)
            }
          val vertical =
            when {
              local[1] < -0.35f -> " · below"
              local[1] > 0.35f -> " · above"
              else -> ""
            }
          "%.1f m · %s%s".format(distance, dir, vertical) +
            if (r.second == PositionSource.MAP) " (via home map)" else ""
        }
      }

    val locLine =
      when {
        !st.hasMap -> "No home map"
        st.located -> "Located · %s · %.0f m walked".format(st.fixName, st.metersSinceFix ?: 0.0)
        else -> "Not located"
      }
    val locState =
      when {
        !st.hasMap -> MainView.LocState.NO_MAP
        st.located -> MainView.LocState.LOCATED
        else -> MainView.LocState.NOT_LOCATED
      }
    val roomChip = if (st.hasMap) (st.roomName(st.currentRoomId) ?: "not set") else null
    val mini = miniMapData(st, q)
    val near = q?.mode == GuidanceMode.NEAR
    val studyLine = StudySettings.statusLine(activity.studyTag, activity.captureMode)

    view.post {
      view.renderStatus(tracking, trackingDetail, locState, locLine, items, hint, roomChip, studyLine)
      if (cue != null && query === q) view.setGuideCue(cue, located != null, near)
      view.miniMap.setData(mini)
    }
  }

  /** Mini-map snapshot (map frame, x-z). Nodes and edges are rebuilt only when the map changed. */
  private fun miniMapData(st: Localizer.State, q: Query?): MiniMapView.Data? {
    val m = st.map ?: return null
    if (m.nodes.isEmpty()) return null
    val base =
      miniMapCache?.takeIf { it.first === m }?.second
        ?: run {
          val index = HashMap<Int, Int>()
          val nodes = FloatArray(m.nodes.size * 2)
          m.nodes.forEachIndexed { i, n ->
            index[n.id] = i
            nodes[2 * i] = n.x.toFloat()
            nodes[2 * i + 1] = n.z.toFloat()
          }
          val edges = ArrayList<Int>(m.edges.size * 2)
          for ((a, b) in m.edges) {
            val ia = index[a] ?: continue
            val ib = index[b] ?: continue
            edges.add(ia)
            edges.add(ib)
          }
          val rooms =
            localizer.graph.roomCentroids().mapNotNull { (id, c) ->
              m.room(id)?.let { MiniMapView.Label(it.name, c.x.toFloat(), c.z.toFloat()) }
            }
          val fixes = m.fixPoints.map { MiniMapView.Mark(it.x.toFloat(), it.z.toFloat(), it.type.name) }
          MiniMapView.Data(nodes, edges.toIntArray(), rooms, fixes, emptyList(), null, null, null, null)
            .also { miniMapCache = m to it }
        }
    val items =
      store.snapshot().mapNotNull { r ->
        val mp = r.mapPosition
        if (mp != null && r.mapId == m.mapId) MiniMapView.Mark(mp[0], mp[2], r.name) else null
      }
    val target =
      q?.record?.let { r ->
        val mp = r.mapPosition
        if (mp != null && r.mapId == m.mapId) MiniMapView.Mark(mp[0], mp[2], r.name)
        else {
          val res = resolve(r)
          val t = localizer.tWM
          if (res != null && t != null) t.inverse().transformPoint(Vec3.of(res.first)).let { MiniMapView.Mark(it.x.toFloat(), it.z.toFloat(), r.name) }
          else null
        }
      }
    val route =
      q?.route?.points?.let { pts ->
        FloatArray(pts.size * 2).also { a ->
          pts.forEachIndexed { i, p ->
            a[2 * i] = p.x.toFloat()
            a[2 * i + 1] = p.z.toFloat()
          }
        }
      }
    val user = if (st.located) localizer.userMap else null
    val fwd = if (st.located) localizer.userForwardMap else null
    return base.copy(
      items = items,
      target = target,
      route = route,
      user = user?.let { MiniMapView.Mark(it.x.toFloat(), it.z.toFloat(), "") },
      userForward = fwd?.let { floatArrayOf(it.x.toFloat(), it.z.toFloat()) },
    )
  }

  // ---------------------------------------------------------------- helpers

  private fun Frame.tryAcquireCameraImage() =
    try {
      acquireCameraImage()
    } catch (e: NotYetAvailableException) {
      null
    }

  private fun dist(a: FloatArray, b: FloatArray): Float {
    val dx = a[0] - b[0]
    val dy = a[1] - b[1]
    val dz = a[2] - b[2]
    return sqrt(dx * dx + dy * dy + dz * dz)
  }

  private fun msSince(t0: Long) = (SystemClock.elapsedRealtimeNanos() - t0) / 1e6
}
