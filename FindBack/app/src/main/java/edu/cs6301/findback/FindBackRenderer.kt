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
import edu.cs6301.findback.scan.ActiveScan
import edu.cs6301.findback.scan.Affine2
import edu.cs6301.findback.scan.CameraPose
import edu.cs6301.findback.scan.CapturedKeyframe
import edu.cs6301.findback.scan.DepthMap
import edu.cs6301.findback.scan.HitSample
import edu.cs6301.findback.scan.InventoryEntry
import edu.cs6301.findback.scan.Intrinsics
import edu.cs6301.findback.scan.OverlayBox
import edu.cs6301.findback.scan.ScanConfig
import edu.cs6301.findback.scan.ScanController
import edu.cs6301.findback.scan.ScanEnd
import edu.cs6301.findback.ml.YuvPlanes
import edu.cs6301.findback.study.CaptureMode
import edu.cs6301.findback.study.CaptureRules
import edu.cs6301.findback.study.GroundTruth
import edu.cs6301.findback.trace.FrameTracer
import edu.cs6301.findback.ui.Action
import edu.cs6301.findback.ui.FoundAnswer
import edu.cs6301.findback.ui.UiModel
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

/**
 * Items → Find, on its way to the GL thread. v0.5: [source] is "tap" for a tapped record and
 * "scan" for a scanned entry, whose record [InventoryEntry.toObjectRecord] built.
 */
class QueryRequest(val record: ObjectRecord, val source: String, val inventoryId: String?)

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
  /** v0.5: "tap" (a tapped record) or "scan" (a scanned entry). */
  val source: String = SOURCE_TAP,
  /** v0.5: the inventory entry the record was built from, or null for a tapped record. */
  val inventoryId: String? = null,
) {
  companion object {
    const val SOURCE_TAP = "tap"
    const val SOURCE_SCAN = "scan"
  }

  /**
   * v0.6: study mode (a tag is set or the mode is not FREE), fixed for the whole query. Study mode
   * keeps the v0.5 prompts; everyday mode asks "Did you find it?" at Done.
   */
  val studyMode: Boolean = UiModel.studyMode(studyTag, captureMode)

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
  /** v0.5: a confirmed capture already counted as `seen` for the scanned entry (once per query). */
  @Volatile var inventorySeen = false

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

  // v0.7 detailed trace: null when the setting is off (then nothing below is sampled or logged).
  private val frameTracer: FrameTracer? = activity.trace?.let { t -> FrameTracer(t.writer, activity.events, t::error) }
  private val density = activity.resources.displayMetrics.density
  /** The frame (elapsed ms) in which a scan keyframe capture was tried; the depth sample waits then. */
  private var keyframeAttemptMs = -1L
  private var lastTraceErrorMs = -10_000L

  // Frozen models shared by all conditions.
  private val detector = BoxDetector()
  @Volatile private var embedder: Embedder? = null

  // UI -> GL requests.
  private val pendingTap = AtomicReference<PendingTap?>(null)
  @Volatile private var pendingVerify = false
  @Volatile private var mlBusy = false
  private val pendingQuery = AtomicReference<QueryRequest?>(null)
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

  /** v0.5 scan mode. Unused (and idle on every thread) until the user starts a scan. */
  val scan: ScanController

  private val scanUi =
    object : ScanController.Ui {
      override fun scanStarted(scan: ActiveScan) = view.post { onScanStarted(scan) }

      override fun overlay(boxes: List<OverlayBox>) = view.post { onScanOverlay(boxes) }

      override fun scanEnded(end: ScanEnd) = view.post { onScanEnded(end) }

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
    scan = ScanController(activity, activity.events, activity.inventory, activity.store, activity.sessionId, { embedder }, scanUi)
    if (frameTracer != null) {
      // v0.7: one `lock_attempt` per Lock press (logging only).
      localizer.lockListener = { a ->
        events.log(
          "lock_attempt",
          mapOf(
            "spot" to a.spot,
            "kind" to a.kind,
            "ok" to a.ok,
            "reason" to a.reason,
            "samples" to a.samples,
            "posSpreadM" to a.posSpreadM,
            "yawSpreadDeg" to a.yawSpreadDeg,
            "durationMs" to a.durationMs,
            "attemptInBar" to a.attemptInBar,
            "rejectedRolled" to a.rejectedRolled,
            "rejectedTracking" to a.rejectedTracking,
          ),
        )
      }
    }
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
    view.onAction = { onAction(it) }
  }

  /** UI thread (v0.6): a bottom-bar button, or Locate in the guide card. Lock and Cancel stay in [MainView]. */
  private fun onAction(a: Action) {
    when (a) {
      Action.ITEMS -> showItems()
      Action.LOCATE -> view.dialogs.locate(uiState)
      Action.SCAN -> requestStartScan()
      Action.STOP_SCAN -> requestStopScan()
      Action.SET_HOME -> view.dialogs.startAddSpot(home = true)
      Action.DONE -> onDonePressed()
      Action.SKIP -> onSkipPressed()
      Action.VERIFY -> pendingVerify = true
      Action.CANCEL_UPDATE -> cancelUpdate("waiting")
      Action.LOCK, Action.CANCEL_LOCK -> {}
    }
  }

  /** UI thread: the Items panel, tapped records and (v0.5) scanned entries together. */
  private fun showItems() {
    if (scan.active != null) return view.toast("Stop the scan first.")
    view.items.open(itemsData(), itemActions)
  }

  private fun itemsData() =
    ItemsPanel.Data(store.snapshot(), scan.inventory.snapshot(), scan.inventory.lastScanId, uiState, store.root)

  /** UI thread: show the change in the Items panel, if it is open. */
  private fun refreshItems() = view.items.refresh(itemsData())

  private val itemActions =
    ItemsPanel.Actions(
      onFind = { startQuery(it) },
      onMoved = { startUpdate(it) },
      onRename = { rec, name ->
        store.rename(rec.id, name)
        events.log("rename", mapOf("record" to rec.id, "from" to rec.name, "to" to name))
        refreshItems()
      },
      onDelete = { rec ->
        sessionAnchors.remove(rec.id)?.let { anchorsToDetach.add(it) }
        store.delete(rec.id)
        events.log("delete", mapOf("record" to rec.id, "name" to rec.name))
        refreshItems()
      },
      onFindEntry = { e -> startQuery(e.toObjectRecord(), Query.SOURCE_SCAN, e.id) },
      onRenameEntry = { e, name -> renameEntry(e, name) },
      onDeleteEntry = { e -> deleteEntry(e) },
    )

  /** UI thread → IO: a scanned entry's new name; it no longer follows the detector's label. */
  private fun renameEntry(e: InventoryEntry, name: String) {
    scope.launch(Dispatchers.IO) {
      val renamed = scan.inventory.rename(e.id, name) ?: return@launch
      events.log("inv_rename", mapOf("inventoryId" to e.id, "from" to e.name, "to" to name, "entry" to renamed.toJson(withEmbedding = false)))
      view.post { refreshItems() }
    }
  }

  /** UI thread → IO: remove a scanned entry from the list. Its photo and log lines stay. */
  private fun deleteEntry(e: InventoryEntry) {
    scope.launch(Dispatchers.IO) {
      val removed = scan.inventory.delete(e.id) ?: return@launch
      events.log("inv_delete", mapOf("inventoryId" to e.id, "name" to removed.name, "entry" to removed.toJson(withEmbedding = false)))
      view.post { refreshItems() }
    }
  }

  /** UI thread, study mode: Skip during a measurement ends the query without one. */
  private fun onSkipPressed() {
    val q = query ?: return
    if (q.measuring) endQuery(q, q.truth ?: GroundTruth.PRESENT, null) // measurement skipped
  }

  /**
   * UI thread: Done. Study mode keeps the v0.5 prompts (PAIRED capture-first, then the five-way
   * ground truth); everyday mode asks "Did you find it?" (v0.6, plan section 6.3).
   */
  private fun onDonePressed() {
    if (updateTarget != null) return cancelUpdate("waiting")
    val q = query ?: return
    if (q.measuring) return onSkipPressed()
    // PAIRED and BENCHMARK tie every capture to this query's end: let a running capture finish.
    if (q.captureMode != CaptureMode.FREE && q.capturing) return view.toast("Still processing the capture…")
    val resolvable = q.resolvable
    if (!q.studyMode) return askFoundIt(q, resolvable)
    if (CaptureRules.askCaptureFirst(q.captureMode, q.captures, resolvable)) {
      view.askCaptureFirst(q.record.name) { askGroundTruth(q, resolvable) }
    } else {
      askGroundTruth(q, resolvable)
    }
  }

  /**
   * UI thread, everyday mode: "Did you find it?". The answer ends the query with PRESENT,
   * DISPLACED or ABSENT and no measuring tap; "Keep looking" or Back changes nothing. "It was
   * somewhere else" goes on to "Moved it" for a tapped record.
   */
  private fun askFoundIt(q: Query, resolvable: Boolean) {
    if (query !== q || q.ended) return
    view.askFoundIt { answer ->
      if (query !== q || q.ended) return@askFoundIt
      q.resolvableAtDone = resolvable
      endQuery(q, answer.truth, null)
      if (answer == FoundAnswer.ELSEWHERE) {
        if (q.source == Query.SOURCE_TAP) {
          store.get(q.record.id)?.let { startUpdate(it) }
        } else {
          view.toast("Scan that area again to update it")
        }
      }
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
        view.refresh() // the guide card asks for the measuring tap; Skip replaces Done
      } else {
        endQuery(q, truth, null)
      }
    }
  }

  override fun onResume(owner: LifecycleOwner) = displayRotationHelper.onResume()

  override fun onPause(owner: LifecycleOwner) {
    displayRotationHelper.onPause()
    // v0.5: a scan ends when the app leaves the screen; its keyframes so far are kept.
    scan.stop("paused")
    // The GL thread is already paused (MainView pauses the surface first): save the latest map.
    activity.mapStore.flush()
  }

  override fun onDestroy(owner: LifecycleOwner) {
    activity.mapStore.close()
    scan.close()
    scope.cancel()
    detector.close()
    embedder?.close()
  }

  /** Called on the UI thread for a single tap on the camera view (VIEW pixel coordinates). */
  fun onTap(x: Float, y: Float) {
    if (view.lockModeActive) {
      rejected("register", "lock_bar")
      view.toast("Line up the view, then press Lock")
      return
    }
    val q = query
    if (q != null && !q.measuring) {
      rejected("register", "search_active", q = q)
      view.toast("Press Done to end this search")
      return
    }
    // v0.5: registration taps are ignored during a scan (plan section 8).
    if (scan.active != null) {
      rejected("register", "scanning")
      view.toast("Press Stop scan first")
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
    val st = uiState
    val roomName = if (rec.mapId != null && rec.mapId == st?.map?.mapId) st.roomName(rec.roomId) else null
    view.enterUpdateMode(rec, store.loadBitmap(rec.cropFile), roomName)
  }

  /** Back to normal with nothing changed. [stage]: "waiting" (before a tap) or "dialog". */
  private fun cancelUpdate(stage: String) {
    val rec = updateTarget ?: return
    updateTarget = null
    view.exitUpdateMode()
    events.log("update_cancel", mapOf("record" to rec.id, "stage" to stage))
    view.toast("The place of \"${rec.name}\" is not changed.")
  }

  // ---------------------------------------------------------------- location requests (UI thread)

  fun requestPickRoom(roomId: String?, newRoom: Room?, source: String) = runOnGl {
    newRoom?.let { localizer.createRoom(it) }
    localizer.pickRoom(roomId, source)
  }

  fun requestRenameRoom(id: String, name: String) = runOnGl { localizer.renameRoom(id, name) }

  /** UI thread. The Lock press number in the open bar goes with the request, for `lock_attempt` only. */
  fun requestUseSpot(fixId: String) {
    val attempt = view.lockPressesInBar
    runOnGl { localizer.startSpotLock(fixId, attempt) }
  }

  fun requestAddSpot() {
    val attempt = view.lockPressesInBar
    runOnGl {
      pendingGhost.set(null)
      ghostCaptureRequested = true
      localizer.startSpotCapture(attempt)
    }
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
  private fun startQuery(rec: ObjectRecord, source: String = Query.SOURCE_TAP, inventoryId: String? = null) {
    pendingQuery.set(QueryRequest(rec, source, inventoryId))
  }

  /** GL thread. */
  private fun startQueryGl(req: QueryRequest, camera: Camera) {
    val rec = req.record
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
        source = req.source,
        inventoryId = req.inventoryId,
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
        // v0.5
        "source" to q.source,
        "inventoryId" to q.inventoryId,
      ) + localizer.proxyFields(nowMs)
    )
    val roomName = localizer.roomName(q.itemRoom)
    view.post { view.enterGuideMode(q, store.loadBitmap(rec.cropFile), roomName) }
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
        // v0.6: which prompt gave the ground truth ("simple": "Did you find it?").
        "truthPrompt" to UiModel.truthPrompt(q.studyMode),
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
    val ft = frameTracer ?: return drawFrame(render)
    // v0.7 `perf`: the time inside the frame callback, tracing included.
    val t0 = System.nanoTime()
    try {
      drawFrame(render)
    } finally {
      ft.frameEnd(t0, mlBusy)
    }
  }

  private fun drawFrame(render: SampleRender) {
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
      // v0.5: scan keyframes, only while a scan runs, after taps and captures had their turn.
      scan.active?.let { a -> guard("scan") { scanTick(a, frame, camera, session, nowMs) } }
      query?.let { q -> guard("guidance") { updateGuidance(q, camera, nowMs, throttle = true) } }
      drawLabels(render, camera)
      updateOverlay(camera)
    } else {
      scan.active?.lastCameraPos = null // no walking across a tracking gap
      if (pendingTap.getAndSet(null) != null) rejected(if (query?.measuring == true) "measure" else "register", "tracking", q = query)
      hideOverlay()
      if (pendingVerify) {
        pendingVerify = false
        rejected("verify", "tracking", q = query)
        view.post { view.toast("Tracking lost: move the phone slowly, then press Verify again.") }
      }
    }

    frameTracer?.let { ft -> traceFrame(ft, frame, camera, session, nowMs) }
    guard("publishUi") { publishUi(frame, camera) }
  }

  // ---------------------------------------------------------------- v0.7 detailed trace (GL thread)

  /** The `pose.csv`, `depth.csv` and `guidance.csv` samples due in this frame. */
  private fun traceFrame(ft: FrameTracer, frame: Frame, camera: Camera, session: Session, nowMs: Long) {
    val t0 = System.nanoTime()
    try {
      val ns = SystemClock.elapsedRealtimeNanos()
      if (ft.poseDue(ns)) {
        ft.pose(
          ns,
          camera,
          localizer.located,
          localizer.userMap,
          localizer.userForwardMap,
          localizer.proxies(nowMs)?.metersSinceFix,
          view.screenName,
          localizer.graph.currentRoom,
          query?.id,
        )
      }
      // A scan keyframe has priority: in a frame that tried one, the depth sample waits a frame.
      val keyframeNow = scan.active != null && keyframeAttemptMs == nowMs
      if (!keyframeNow && ft.depthDue(ns)) ft.depth(ns, frame, session)
      val q = query
      if (q != null && ft.guidanceDue(ns)) traceGuidance(ft, ns, q, camera)
    } catch (e: Exception) {
      val now = SystemClock.elapsedRealtime()
      if (now - lastTraceErrorMs >= 5000L) {
        lastTraceErrorMs = now
        Log.e(TAG, "Trace sampling failed", e)
        activity.trace?.error("gl", e.toString())
      }
    } finally {
      ft.addTraceNs(System.nanoTime() - t0)
    }
  }

  /** One `guidance.csv` row: where the item's remembered position is from the camera and on screen. */
  private fun traceGuidance(ft: FrameTracer, ns: Long, q: Query, camera: Camera) {
    val r = if (camera.trackingState == TrackingState.TRACKING) resolve(q.record) else null
    var distance = Double.NaN
    var bearing = Double.NaN
    var elevation = Double.NaN
    var inView: Boolean? = null
    var ringX = Double.NaN
    var ringY = Double.NaN
    if (r != null) {
      val w = r.first
      val local = camera.displayOrientedPose.inverse().transformPoint(w)
      val horiz = sqrt(local[0] * local[0] + local[2] * local[2])
      distance = sqrt(horiz * horiz + local[1] * local[1]).toDouble()
      bearing = Math.toDegrees(atan2(local[0], -local[2]).toDouble())
      elevation = Math.toDegrees(atan2(local[1], horiz).toDouble())
      // As updateOverlay and GuidanceOverlay.onDraw place the ring.
      val clip = FloatArray(4)
      Matrix.multiplyMV(clip, 0, viewProjectionMatrix, 0, floatArrayOf(w[0], w[1], w[2], 1f), 0)
      if (clip[3] > 0f && viewWidth > 0 && viewHeight > 0) {
        val sx = (clip[0] / clip[3] + 1f) / 2f * viewWidth
        val sy = (1f - clip[1] / clip[3]) / 2f * viewHeight
        ringX = sx.toDouble()
        ringY = sy.toDouble()
        inView = GuidanceOverlay.onScreen(true, sx, sy, viewWidth.toFloat(), viewHeight.toFloat(), density)
      } else {
        inView = false
      }
    }
    ft.guidance(ns, q.id, q.mode?.name, q.resolvable, distance, bearing, elevation, q.routeM, q.straightM, inView, ringX, ringY)
  }

  /**
   * v0.7 `capture_rejected`: a registration tap, a Verify press or a measuring tap refused before
   * any capture. Trace on only. Any thread.
   */
  private fun rejected(stage: String, reason: String, value: Any? = null, q: Query? = null) {
    if (frameTracer == null) return
    events.log("capture_rejected", mapOf("stage" to stage, "reason" to reason, "value" to value, "query" to q?.id))
  }

  /** [rejected], then the banner [message] (as the refusals always showed it). */
  private fun refuse(stage: String, reason: String, message: String, value: Any? = null, q: Query? = null) {
    rejected(stage, reason, value, q)
    view.post { view.toast(message) }
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
          else "Picture found, but this map does not use it (see the menu ⋮)"
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
    if (scan.active != null) return rejected("register", "scanning") // v0.5: a scan started after this tap was made
    // v0.4 "Update location": a tap made in that mode replaces the record it was made for.
    val updateOf = tap.updateOf
    if (updateOf != null && updateTarget !== updateOf) return // cancelled before the tap arrived
    if (mlBusy) return refuse("register", "busy", "Still processing the previous capture…")
    val emb = embedder ?: return refuse("register", "model_loading", "Model still loading…")

    val hit =
      bestHit(frame, tap.point.x, tap.point.y)
        ?: return refuse("register", "no_hit", "No surface found there. Move the phone slowly to map the area, then tap again.")
    val image =
      frame.tryAcquireCameraImage() ?: return refuse("register", "image_not_ready", "Camera image not ready; tap again.")

    val imgPt = FloatArray(2)
    frame.transformCoordinates2d(
      Coordinates2d.VIEW, floatArrayOf(tap.point.x, tap.point.y), Coordinates2d.IMAGE_PIXELS, imgPt
    )
    val hitPos = hit.hitPose.translation
    val distance = dist(hitPos, camera.pose.translation).coerceAtLeast(0.05f)
    if (distance > MAX_HIT_DISTANCE_M) {
      image.close()
      events.log("hit_rejected", mapOf("distance" to distance, "hitType" to hit.trackable.javaClass.simpleName))
      return refuse(
        "register",
        "depth_far",
        "Depth looks wrong (%.0f m). Move the phone sideways a little, then tap again.".format(distance),
        value = distance,
      )
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
            studyMode = UiModel.studyMode(activity.studyTag, activity.captureMode),
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
          (if (old != null) "Saved the new place of \"$name\"" else "Saved \"$name\"") +
            if (d.mapPosition == null) " (this session only: not located)"
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
    val hit = bestHit(frame, tap.x, tap.y) ?: return refuse("measure", "no_hit", "No surface there; tap again or Skip.", q = q)
    val (marker, source) =
      resolve(q.record) ?: return refuse("measure", "not_located", "No marker to compare against; tap Skip.", q = q)
    val truthValue = q.truth ?: GroundTruth.PRESENT
    val truth = hit.hitPose.translation
    val truthDistance = dist(truth, frame.camera.pose.translation)
    if (truthDistance > MAX_HIT_DISTANCE_M) {
      return refuse("measure", "depth_far", "Depth looks wrong there. Move the phone a little and tap again.", value = truthDistance, q = q)
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
      return refuse("verify", "capture_limit", "PAIRED mode: this query already has its capture. Press Done.", q = q)
    }
    if (mlBusy) return refuse("verify", "busy", "Still processing…", q = q)
    val emb = embedder ?: return refuse("verify", "model_loading", "Model still loading…", q = q)
    val (world, source) =
      resolve(q.record)
        ?: return refuse("verify", "not_located", "Not located: press Locate first.", q = q)

    val clip = FloatArray(4)
    Matrix.multiplyMV(clip, 0, viewProjectionMatrix, 0, floatArrayOf(world[0], world[1], world[2], 1f), 0)
    if (clip[3] <= 0f) {
      return refuse("verify", "behind", "The item's last place is behind you. Follow the arrow, then press Verify.", q = q)
    }
    val ndc = floatArrayOf(clip[0] / clip[3], clip[1] / clip[3])
    if (abs(ndc[0]) > 0.9f || abs(ndc[1]) > 0.9f) {
      return refuse("verify", "not_in_view", "Bring the ring into view, then press Verify.", value = max(abs(ndc[0]), abs(ndc[1])), q = q)
    }
    val image = frame.tryAcquireCameraImage() ?: return refuse("verify", "image_not_ready", "Camera image not ready; try again.", q = q)
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
        // v0.5: for a scanned entry, a confirmed capture is evidence that it is still there.
        val inventoryId = q.inventoryId
        if (confirmed && inventoryId != null && !q.inventorySeen) {
          q.inventorySeen = true
          scan.onVerifiedSeen(inventoryId, q.id, attempt, similarity)
        }
        val storedThumb = store.loadBitmap(rec.cropFile)
        withContext(Dispatchers.Main) {
          // The guide card reads q.lastOutcome; PAIRED hides Verify after its capture.
          if (query === q) view.refresh()
          if (q.studyMode) {
            view.showVerifyResult(storedThumb, best.first.third, similarity, confirmed) {
              // v0.4 BENCHMARK: once the result is dismissed, the experimenter labels the capture.
              if (q.captureMode == CaptureMode.BENCHMARK) askVerifyLabel(q, rec.id, attempt)
            }
          } else {
            // v0.6 everyday mode: the result is in the guide card; a tap on it shows both photos.
            view.setLastCapture(q.id, storedThumb, best.first.third, similarity, confirmed)
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

  // ---------------------------------------------------------------- scan mode (v0.5)

  /** UI thread: Scan pressed. The scan starts on the GL thread, where the map state lives. */
  fun requestStartScan() = runOnGl { guard("scan_start") { startScanGl() } }

  /** UI thread: Stop pressed. */
  fun requestStopScan() = scan.stop("stop")

  /** GL thread. A scan needs a located map and an idle app (no search, update, lock or capture). */
  private fun startScanGl() {
    val why =
      when {
        scan.active != null -> return
        cameraTracking != TrackingState.TRACKING -> "Move the phone slowly until tracking resumes, then press Scan."
        localizer.tWM == null || localizer.currentMapId == null -> "Press Locate first, then Scan."
        query != null -> "Finish this search (Done) before scanning."
        updateTarget != null -> "Finish \"Moved it\" before scanning."
        localizer.lockActive -> "Finish the spot lock before scanning."
        mlBusy -> "Still processing the previous capture…"
        embedder == null -> "Model still loading…"
        else -> null
      }
    if (why != null) return view.post { view.toast(why) }
    scan.start(
      localizer.currentMapId!!,
      activity.studyTag,
      activity.captureMode.name,
      activity.scanSaveFrames,
      userRoom(),
      localizer.proxyFields(SystemClock.elapsedRealtime()),
    )
  }

  /** GL thread, every tracking frame of a scan: pace keyframes and capture one when due. */
  private fun scanTick(a: ActiveScan, frame: Frame, camera: Camera, session: Session, nowMs: Long) {
    val p = camera.pose
    val pose = CameraPose.of(p.translation, p.rotationQuaternion)
    a.trackWalk(pose.position)
    // Taps, verification captures, ghost captures and spot locks always go first.
    val busy =
      scan.inFlight.get() || mlBusy || ghostCaptureRequested || pendingTap.get() != null || pendingVerify ||
        localizer.lockActive || query != null || updateTarget != null
    val located = localizer.tWM != null
    scan.pacer.onFrame(nowMs, pose, scanning = located, passiveAllowed = false, busy = busy) ?: return
    keyframeAttemptMs = nowMs // v0.7 trace: the depth sample waits for the next frame
    captureKeyframe(a, frame, camera, session, pose, nowMs)
  }

  /**
   * GL thread (plan section 4): copy out the depth image and the CPU image and close both at
   * once, with the pose, intrinsics, rotation, map transform, room and the image-to-depth and
   * image-to-view mappings (three points through `transformCoordinates2d`). No depth: no keyframe.
   */
  private fun captureKeyframe(a: ActiveScan, frame: Frame, camera: Camera, session: Session, pose: CameraPose, nowMs: Long) {
    val t0 = SystemClock.elapsedRealtimeNanos()
    val tWM = localizer.tWM ?: return
    val depthImage =
      try {
        frame.acquireDepthImage16Bits()
      } catch (e: Exception) {
        null // not available yet (or depth unsupported)
      }
    if (depthImage == null) {
      a.skippedNoDepth++
      scan.pacer.failed(nowMs)
      return
    }
    val (dw, dh, mm) = try { ScanController.copyDepth(depthImage) } finally { depthImage.close() }
    val image =
      try {
        frame.acquireCameraImage()
      } catch (e: Exception) {
        null
      }
    if (image == null) {
      a.skippedNoImage++
      scan.pacer.failed(nowMs)
      return
    }
    val yuv = try { YuvPlanes.copy(image) } finally { image.close() }
    val w = yuv.width.toFloat()
    val h = yuv.height.toFloat()
    val corners = floatArrayOf(0f, 0f, w, 0f, 0f, h)
    val src = DoubleArray(6) { corners[it].toDouble() }
    val tex = FloatArray(6)
    frame.transformCoordinates2d(Coordinates2d.IMAGE_PIXELS, corners, Coordinates2d.TEXTURE_NORMALIZED, tex)
    // Depth pixel = texture-normalized coordinate × depth image size (ARCore depth guide).
    val toDepth = Affine2.fromThreePoints(src, DoubleArray(6) { tex[it].toDouble() * if (it % 2 == 0) dw else dh })
    if (toDepth == null) {
      scan.pacer.failed(nowMs)
      return
    }
    val viewPts = FloatArray(6)
    frame.transformCoordinates2d(Coordinates2d.IMAGE_PIXELS, corners, Coordinates2d.VIEW, viewPts)
    val toView = Affine2.fromThreePoints(src, DoubleArray(6) { viewPts[it].toDouble() })
    val ki = camera.imageIntrinsics
    val f = ki.focalLength
    val c = ki.principalPoint
    val dims = ki.imageDimensions
    val intrinsics =
      Intrinsics(f[0].toDouble(), f[1].toDouble(), c[0].toDouble(), c[1].toDouble(), dims[0], dims[1]).scaledTo(yuv.width, yuv.height)
    val index = a.nextIndex++
    val hits = if (index < ScanConfig.SELF_CHECK_KEYFRAMES) selfCheckHits(frame, yuv.width, yuv.height) else emptyList()
    val rotation = displayRotationHelper.getCameraSensorToDisplayRotation(session.cameraConfig.cameraId)
    scan.pacer.taken(nowMs, pose)
    a.lastKeyframeElapsedMs = nowMs
    scan.submit(
      CapturedKeyframe(
        scan = a,
        index = index,
        tMs = System.currentTimeMillis(),
        elapsedMs = nowMs,
        yuv = yuv,
        rotation = rotation,
        intrinsics = intrinsics,
        pose = pose,
        mapToWorld = tWM,
        roomId = userRoom(),
        depth = DepthMap(dw, dh, mm, toDepth),
        imageToView = toView,
        hits = hits,
        glMs = msSince(t0),
      )
    )
  }

  /**
   * Self-check (plan section 5): hit-test the screen point of the image centre, and of one
   * off-centre point, in this frame. The worker compares them with its own points.
   */
  private fun selfCheckHits(frame: Frame, w: Int, h: Int): List<HitSample> {
    val points =
      listOf(
        "centre" to (w / 2.0 to h / 2.0),
        "offCentre" to (ScanConfig.SELF_CHECK_OFF_CENTRE_U * w.toDouble() to ScanConfig.SELF_CHECK_OFF_CENTRE_V * h.toDouble()),
      )
    return points.map { (name, uv) ->
      val (u, v) = uv
      val view = FloatArray(2)
      frame.transformCoordinates2d(Coordinates2d.IMAGE_PIXELS, floatArrayOf(u.toFloat(), v.toFloat()), Coordinates2d.VIEW, view)
      if (view[0] < 0f || view[1] < 0f || view[0] > viewWidth || view[1] > viewHeight) {
        HitSample(name, u, v, null, "outside_view")
      } else {
        val hit = bestHit(frame, view[0], view[1])
        HitSample(name, u, v, hit?.let { Vec3.of(it.hitPose.translation) }, hit?.trackable?.javaClass?.simpleName ?: "none")
      }
    }
  }

  /** UI thread: the scan bar replaces Items and Scan (a search, rename or delete waits for Stop scan). */
  private fun onScanStarted(a: ActiveScan) {
    if (scan.active !== a) return // already stopped again
    view.items.close("scan")
    view.refresh()
    view.toast("Scan started: walk slowly past the things around you. Press Stop scan when done.")
  }

  /** UI thread: the boxes of the latest keyframe. */
  private fun onScanOverlay(boxes: List<OverlayBox>) {
    view.scanOverlay.setBoxes(if (scan.active != null) boxes else emptyList())
  }

  /** UI thread. */
  private fun onScanEnded(end: ScanEnd) {
    view.scanOverlay.setBoxes(emptyList())
    view.refresh()
    if (activity.isFinishing || activity.isDestroyed) return
    view.showScanSummary(end) { if (scan.active == null) showItems() }
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
    val q = query
    if (q == null && tracking) labelIds = computeLabelIds(camera)
    val items = store.snapshot().size
    val located = q?.let { resolve(it.record) }
    val itemRoomName = q?.let { localizer.roomName(it.itemRoom) }
    val scanning = scan.active

    // v0.6: the guide card's cue; the rest of the screen comes from UiModel on the UI thread.
    val cue =
      if (q == null || q.measuring) null
      else {
        val r = located
        val nearText =
          if (q.mode == GuidanceMode.NEAR && r != null) {
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
            // The position source is a study detail.
            "%.1f m · %s%s".format(distance, dir, vertical) +
              if (q.studyMode && r.second == PositionSource.MAP) " (via home map)" else ""
          } else null
        UiModel.cue(q.mode, itemRoomName, q.routeM, q.straightM, st.located, nearText)
      }
    val frameUi =
      UiFrame(
        st = st,
        tracking = tracking,
        busy = mlBusy,
        itemCount = items,
        queryId = q?.id,
        cue = cue,
        scanItems = scanning?.itemCount ?: 0,
        scanHint = scanning?.let { scanHint(it, st.located, now) },
        mini = miniMapData(st, q),
      )
    view.post { view.render(frameUi) }
  }

  /** v0.6: the scan bar's hint ("Walk slowly", "Move the phone sideways so depth can start", ...). */
  private fun scanHint(a: ActiveScan, located: Boolean, nowMs: Long): String =
    UiModel.scanHint(located, localizer.lockActive, nowMs - a.lastKeyframeElapsedMs > ScanConfig.NO_KEYFRAME_HINT_MS)

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
