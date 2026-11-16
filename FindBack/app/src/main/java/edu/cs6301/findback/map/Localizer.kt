package edu.cs6301.findback.map

import android.os.SystemClock
import android.util.Log
import com.google.ar.core.Anchor
import com.google.ar.core.AugmentedImage
import com.google.ar.core.Camera
import com.google.ar.core.Pose
import com.google.ar.core.Session
import com.google.ar.core.TrackingState
import edu.cs6301.findback.memory.EventLog
import java.util.Queue
import java.util.UUID
import kotlin.math.abs
import org.json.JSONObject

/**
 * Session state of the home map, owned by the GL thread: the loaded map (fix points, rooms, walk
 * graph), whether and how it is located in this session (T_WM: map frame M to world frame W), the
 * ARCore anchor that carries the fix, and the drift proxies.
 *
 * Every method must be called on the GL thread. The UI reads [state] (an immutable snapshot) and
 * sends requests as GL-thread commands; [Ui] callbacks hand their answers back the same way.
 */
class Localizer(
  private val events: EventLog,
  private val mapStore: MapStore,
  private val sessionId: String,
  private val refWidthM: Float,
  private val anchorsToDetach: Queue<Anchor>,
  private val ui: Ui,
  loaded: HomeMap?,
) {
  companion object {
    private const val TAG = "FindBack"

    /** IMAGE gate: the camera must be within clamp(FACTOR × picture width, MIN, MAX) metres. */
    const val GATE_DISTANCE_FACTOR = 6.0
    const val GATE_DISTANCE_MIN_M = 0.5
    const val GATE_DISTANCE_MAX_M = 1.5
    /** IMAGE: use the picture's local +X as heading if its horizontal length is at least this, else +Z. */
    const val IMAGE_AXIS_CHOOSE_X = 0.5
    /** IMAGE: reject a sample whose stored heading axis is shorter than this horizontally. */
    const val IMAGE_AXIS_MIN = 0.3
    /** Accept a new IMAGE fix while located at most this often. */
    const val REFRESH_MIN_MS = 2000L
    /** A new fix that moves the map more than this (at the fix point) asks before applying. */
    const val DISAGREE_M = 0.75
    const val DISAGREE_DEG = 15.0
    /** After "Keep current", ignore automatic picture refreshes for this long. */
    const val REJECT_COOLDOWN_MS = 30_000L
    /** An applied fix that moved the map more than this breaks the walk chain. */
    const val CHAIN_BREAK_JUMP_M = 0.3
    /** Walk graph feed interval (about 5 Hz). */
    const val WALK_FEED_MS = 200L
    /** An IMAGE fix point measured at another picture width is ignored. */
    const val WIDTH_TOLERANCE_M = 0.0005

    fun newId(): String = UUID.randomUUID().toString().substring(0, 8)

    fun poseJson(p: Pose4): JSONObject =
      JSONObject().apply {
        put("x", p.x)
        put("y", p.y)
        put("z", p.z)
        put("yawDeg", Math.toDegrees(p.yaw))
      }
  }

  /** UI requests. Implementations show a dialog and deliver the answer back on the GL thread. */
  interface Ui {
    fun askSetHome(onAnswer: (Boolean) -> Unit)

    fun askUseNewFix(jumpM: Double, jumpDeg: Double, fixName: String, onAnswer: (Boolean) -> Unit)

    fun askAddPicture(onAnswer: (Boolean) -> Unit)

    /** "Add a spot here" collected its frames: ask for a name, then call [addSpot]. */
    fun spotCaptured(capture: SpotCapture)

    /** A spot lock or capture ended (success, failure or cancel). */
    fun lockFinished(ok: Boolean, message: String)

    fun toast(message: String)
  }

  /** A spot pose collected by "Add a spot here", waiting for its name. */
  data class SpotCapture(
    val id: String,
    val worldPose: Pose4,
    /** Map pose computed with T_WM at capture; null when this spot will create the map. */
    val mapPose: Pose4?,
    val estimate: FixEstimate,
    val isHome: Boolean,
    /** Display axis ("X" or "Y") the heading was taken from; stored on the fix point and reused. */
    val yawAxis: String,
  )

  /** What the UI shows: published with the 5 Hz update. */
  data class State(
    val map: HomeMap?,
    val located: Boolean,
    val fixName: String?,
    val fixType: FixType?,
    val metersSinceFix: Double?,
    val currentRoomId: String?,
    /** Set when the picture fix point is ignored because the width setting changed. */
    val imageFixNote: String?,
    val lockActive: Boolean,
    /** The map has a picture fix point at the current width, so looking at the picture locates. */
    val pictureUsable: Boolean,
    /** "Set home here?" was answered "Not now" this session (it is asked only once). */
    val setHomeDeclined: Boolean,
  ) {
    val hasMap: Boolean
      get() = map != null

    fun roomName(id: String?): String? = map?.room(id)?.name
  }

  private data class Jump(val m: Double, val horizM: Double, val deg: Double)

  // ---- map content
  private var mapId: String? = loaded?.mapId
  private var mapCreatedMs: Long = loaded?.createdMs ?: 0L
  private val fixPoints = ArrayList<FixPoint>(loaded?.fixPoints ?: emptyList())
  private val rooms = ArrayList<Room>(loaded?.rooms ?: emptyList())
  val graph = WalkGraph(loaded?.nodes ?: emptyList(), loaded?.edges ?: emptyList())
  private var metaVersion = 0
  private var snapshotKey: Pair<Int, Int>? = null
  private var snapshot: HomeMap? = loaded

  // ---- location
  /** Map to world transform for this session, or null when not located. */
  var tWM: Pose4? = null
    private set
  var activeFix: FixPoint? = null
    private set
  private var anchor: Anchor? = null
  /** World pose of the active fix while its anchor could not be created yet. */
  private var pendingAnchorPose: Pose4? = null
  private var lastFixMs = 0L
  private var rejectUntilMs = 0L
  private val drift = DriftTracker()
  private val lossDetector = TrackingLossDetector()
  private var session: Session? = null

  // ---- picture
  private val window = ImageFixWindow()
  private var askedSetHome = false
  private var setHomeDeclined = false
  private var askedAddPicture = false
  private var promptOpen = false

  // ---- spots
  private enum class LockMode { USE, CAPTURE }

  private var lockMode: LockMode? = null
  private var lockFix: FixPoint? = null
  private var lockCollector: SpotCollector? = null
  /** Display axis the heading is taken from during this lock: the spot's stored one, or chosen on capture. */
  private var lockAxis: String? = null
  private var lockStartMs = 0L

  // ---- user
  /** Camera position in M on the last tracking frame while located. */
  var userMap: Vec3? = null
    private set
  /** Camera heading in M (horizontal forward direction), when the phone is not rolled. */
  var userForwardMap: Vec3? = null
    private set
  private var lastWalkFeedMs = 0L

  val located: Boolean
    get() = tWM != null

  val currentMapId: String?
    get() = mapId

  fun roomName(id: String?): String? = if (id == null) null else rooms.firstOrNull { it.id == id }?.name

  fun proxies(nowMs: Long): DriftProxies? = drift.proxies(nowMs)

  /** Drift proxies as log fields (null values when not located). */
  fun proxyFields(nowMs: Long): Map<String, Any?> = drift.proxies(nowMs)?.toMap() ?: DriftProxies.emptyMap()

  fun usableImageFix(): FixPoint? =
    fixPoints.firstOrNull { it.type == FixType.IMAGE && it.widthM != null && abs(it.widthM - refWidthM) <= WIDTH_TOLERANCE_M }

  fun spots(): List<FixPoint> = fixPoints.filter { it.type == FixType.SPOT }

  /** Whether seeing the picture now leads anywhere (set home, locate, refresh, or add it). */
  fun pictureUseful(): Boolean =
    (mapId == null && !askedSetHome) || usableImageFix() != null || (tWM != null && !askedAddPicture)

  // ------------------------------------------------------------------ per frame

  /** Called once per frame after `session.update()`. [image] is the reference picture trackable, if any. */
  fun onFrame(nowMs: Long, session: Session, camera: Camera, image: AugmentedImage?) {
    this.session = session
    val tracking = camera.trackingState == TrackingState.TRACKING
    when (val ev = lossDetector.update(tracking, nowMs, camera.trackingFailureReason.name)) {
      is TrackingLossDetector.Event.Lost -> {
        drift.onTrackingLoss()
        graph.breakChain()
        events.log("tracking_lost", mapOf("reason" to ev.reason, "afterMs" to ev.afterMs, "located" to located))
      }
      is TrackingLossDetector.Event.Resumed ->
        events.log("tracking_resumed", mapOf("reason" to ev.reason, "durationMs" to ev.durationMs, "located" to located))
      null -> {}
    }
    if (!tracking) {
      drift.breakPath()
      lockCollector?.let {
        it.rejectTracking()
        checkLockTimeout(nowMs, it)
      }
      return
    }

    updateAnchor()
    val cameraPos = Vec3.of(camera.pose.translation)
    drift.onPosition(cameraPos)
    handleImage(nowMs, cameraPos, image)
    handleLock(nowMs, camera)

    // After any fix applied above, so the walk feed never uses a pre-jump map position.
    val t = tWM
    if (t != null) {
      val toMap = t.inverse()
      userMap = toMap.transformPoint(cameraPos)
      // Heading: the viewing direction, or the top of the screen when looking steeply up or down.
      // One of the two is always well away from vertical, so this works at any pitch and roll.
      val dp = camera.displayOrientedPose
      val viewDir = Vec3.of(dp.zAxis) * -1.0
      val dir = if (viewDir.horizontalLength() >= 0.5) viewDir else Vec3.of(dp.yAxis)
      val h = dir.horizontalLength()
      if (h >= IMAGE_AXIS_MIN) userForwardMap = toMap.rotate(Vec3(dir.x / h, 0.0, dir.z / h))
    } else {
      userMap = null
      userForwardMap = null
    }
    feedWalk(nowMs)
  }

  private fun updateAnchor() {
    val fix = activeFix ?: return
    val a = anchor
    if (a == null) {
      // The fix was applied while ARCore could not create an anchor; try again now.
      val p = pendingAnchorPose ?: return
      anchor = createAnchor(p)
      if (anchor != null) pendingAnchorPose = null
      return
    }
    when (a.trackingState) {
      TrackingState.TRACKING -> {
        // Re-extract the fix pose the same way every frame so ARCore's map corrections move the map.
        val p = a.pose
        tWM = Pose4.fromAxis(p.translation, p.xAxis).compose(fix.mapPose.inverse())
      }
      TrackingState.STOPPED -> loseLocation("anchor_stopped")
      else -> {}
    }
  }

  private fun createAnchor(p: Pose4): Anchor? =
    try {
      session?.createAnchor(Pose(p.translation.toFloatArray(), p.quaternion()))
    } catch (e: Exception) {
      Log.w(TAG, "createAnchor failed: ${e.message}")
      null
    }

  private fun loseLocation(reason: String) {
    events.log("map_fix_lost", mapOf("mapId" to mapId, "fixPoint" to activeFix?.id, "reason" to reason))
    anchor?.let { anchorsToDetach.add(it) }
    anchor = null
    pendingAnchorPose = null
    tWM = null
    activeFix = null
    drift.clear()
    graph.breakChain()
    ui.toast("Map position lost: look at the picture or use a spot again.")
  }

  // ------------------------------------------------------------------ picture fixes

  private fun handleImage(nowMs: Long, cameraPos: Vec3, image: AugmentedImage?) {
    if (image == null || image.trackingState != TrackingState.TRACKING) return
    if (image.trackingMethod != AugmentedImage.TrackingMethod.FULL_TRACKING) return
    val center = image.centerPose
    val centerPos = Vec3.of(center.translation)
    val camDist = distance(cameraPos, centerPos)
    val gate = (GATE_DISTANCE_FACTOR * refWidthM).coerceIn(GATE_DISTANCE_MIN_M, GATE_DISTANCE_MAX_M)
    if (camDist > gate) return

    val usable = usableImageFix()
    val axisName =
      usable?.yawAxis ?: window.axis ?: if (Vec3.of(center.xAxis).horizontalLength() >= IMAGE_AXIS_CHOOSE_X) "X" else "Z"
    val axis = Vec3.of(if (axisName == "Z") center.zAxis else center.xAxis)
    if (axis.horizontalLength() < IMAGE_AXIS_MIN) return
    if (window.axis != axisName) {
      window.reset()
      window.axis = axisName
    }
    window.add(FixSample(nowMs, Pose4.fromAxis(centerPos, axis), camDist))
    if (promptOpen || lockMode != null) return
    evaluateImage(nowMs, usable)
  }

  private fun evaluateImage(nowMs: Long, usable: FixPoint?) {
    val axisName = window.axis ?: return
    when {
      mapId == null -> {
        if (askedSetHome) return
        val est = window.estimate(nowMs, allowLoose = true) ?: return
        askedSetHome = true
        promptOpen = true
        window.reset()
        ui.askSetHome { yes ->
          promptOpen = false
          if (yes && mapId == null) createMapFromImage(est, axisName)
          else {
            setHomeDeclined = !yes
            events.log("map_create_declined", mapOf("fixType" to FixType.IMAGE.name))
          }
        }
      }
      tWM == null && usable != null -> {
        val est = window.estimate(nowMs, allowLoose = true) ?: return
        window.reset()
        applyFix(usable, est, "first", disagreement = false, accepted = null)
      }
      tWM != null && usable != null -> {
        if (nowMs - lastFixMs < REFRESH_MIN_MS || nowMs < rejectUntilMs) return
        // Refreshes must converge: a loose estimate never replaces a working fix.
        val est = window.estimate(nowMs, allowLoose = false) ?: return
        window.reset()
        proposeFix(usable, est, "refresh")
      }
      tWM != null && usable == null -> {
        if (askedAddPicture) return
        val est = window.estimate(nowMs, allowLoose = false) ?: return
        askedAddPicture = true
        promptOpen = true
        window.reset()
        val mapPose = tWM!!.inverse().compose(est.pose)
        ui.askAddPicture { yes ->
          promptOpen = false
          if (yes) {
            addFixPoint(
              FixPoint.at(newId(), "Picture", FixType.IMAGE, mapPose, axisName, refWidthM.toDouble(), null, System.currentTimeMillis(), sessionId),
              est,
            )
            ui.toast("Picture saved as a fix point")
          }
        }
      }
      else -> {} // a map exists, not located, and the picture is not one of its fix points
    }
  }

  private fun createMapFromImage(est: FixEstimate, axisName: String) {
    val fix =
      FixPoint.at(newId(), "Home picture", FixType.IMAGE, Pose4.IDENTITY, axisName, refWidthM.toDouble(), null, System.currentTimeMillis(), sessionId)
    createMap(fix, est)
  }

  private fun createMap(first: FixPoint, est: FixEstimate) {
    mapId = newId()
    mapCreatedMs = System.currentTimeMillis()
    events.log(
      "map_created",
      mapOf("mapId" to mapId, "fixPoint" to first.id, "fixType" to first.type.name, "pose" to poseJson(first.mapPose)),
    )
    addFixPoint(first, est)
    applyFix(first, est, "first", disagreement = false, accepted = null)
  }

  private fun addFixPoint(f: FixPoint, est: FixEstimate?) {
    fixPoints.add(f)
    metaVersion++
    events.log(
      "fix_point_added",
      mapOf(
        "mapId" to mapId,
        "fixPoint" to f.id,
        "name" to f.name,
        "fixType" to f.type.name,
        "pose" to poseJson(f.mapPose),
        "yawAxis" to f.yawAxis,
        "widthM" to f.widthM,
        "ghostFile" to f.ghostFile,
        "samples" to est?.samples,
        "posSpreadM" to est?.posSpreadM,
        "yawSpreadDeg" to est?.yawSpreadDeg,
      ),
    )
  }

  private fun jumpBetween(old: Pose4, new: Pose4, fix: FixPoint): Jump {
    val at = fix.mapPose.translation
    val a = old.transformPoint(at)
    val b = new.transformPoint(at)
    return Jump(distance(a, b), horizontalDistance(a, b), Math.toDegrees(abs(Pose4.angleDiff(new.yaw, old.yaw))))
  }

  /** A fix while located: apply silently unless it disagrees strongly with the current one. */
  private fun proposeFix(fix: FixPoint, est: FixEstimate, kind: String) {
    val old = tWM ?: return applyFix(fix, est, kind, disagreement = false, accepted = null)
    val jump = jumpBetween(old, est.pose.compose(fix.mapPose.inverse()), fix)
    if (jump.m <= DISAGREE_M && jump.deg <= DISAGREE_DEG) {
      applyFix(fix, est, kind, disagreement = false, accepted = null)
      return
    }
    promptOpen = true
    ui.askUseNewFix(jump.m, jump.deg, fix.name) { yes ->
      promptOpen = false
      if (yes) applyFix(fix, est, kind, disagreement = true, accepted = true)
      else {
        rejectUntilMs = SystemClock.elapsedRealtime() + REJECT_COOLDOWN_MS
        logFix(fix, est, kind, jump, drift.proxies(SystemClock.elapsedRealtime()), disagreement = true, accepted = false, applied = false, anchored = anchor != null)
      }
    }
  }

  private fun applyFix(fix: FixPoint, est: FixEstimate, kind: String, disagreement: Boolean, accepted: Boolean?) {
    val nowMs = SystemClock.elapsedRealtime()
    val old = tWM
    val newT = est.pose.compose(fix.mapPose.inverse())
    val jump = old?.let { jumpBetween(it, newT, fix) }
    anchor?.let { anchorsToDetach.add(it) }
    anchor = createAnchor(est.pose)
    pendingAnchorPose = if (anchor == null) est.pose else null
    tWM = newT
    activeFix = fix
    lastFixMs = nowMs
    val prev = drift.onFix(nowMs, fix.id, fix.type)
    if (old == null || (jump?.m ?: 0.0) > CHAIN_BREAK_JUMP_M) graph.breakChain()
    logFix(fix, est, if (old == null) "first" else kind, jump, prev, disagreement, accepted, applied = true, anchored = anchor != null)
    when {
      old == null -> ui.toast("Located via ${fix.name}")
      kind == "manual" && jump != null -> ui.toast("Re-fixed at %s: moved %.0f cm, %.0f°".format(fix.name, jump.m * 100, jump.deg))
    }
  }

  private fun logFix(
    fix: FixPoint,
    est: FixEstimate,
    kind: String,
    jump: Jump?,
    prev: DriftProxies?,
    disagreement: Boolean,
    accepted: Boolean?,
    applied: Boolean,
    anchored: Boolean,
  ) {
    events.log(
      "map_fix",
      mapOf(
        "mapId" to mapId,
        "fixPoint" to fix.id,
        "fixName" to fix.name,
        "fixType" to fix.type.name,
        "kind" to kind,
        "samples" to est.samples,
        "posSpreadM" to est.posSpreadM,
        "yawSpreadDeg" to est.yawSpreadDeg,
        "cameraDistanceM" to if (fix.type == FixType.IMAGE) est.cameraDistanceM else null,
        "quality" to est.quality,
        "jumpM" to jump?.m,
        "jumpHorizM" to jump?.horizM,
        "jumpDeg" to jump?.deg,
        "metersSincePrevFix" to prev?.metersSinceFix,
        "lossesSincePrevFix" to prev?.trackingLossesSinceFix,
        "msSincePrevFix" to prev?.msSinceFix,
        "prevFixPoint" to prev?.fixPointId,
        "disagreement" to disagreement,
        "accepted" to accepted,
        "applied" to applied,
        "anchored" to anchored,
        "worldPose" to poseJson(est.pose),
      ),
    )
  }

  // ------------------------------------------------------------------ spots

  /** "Use a spot" → Lock: average the gated camera pose and fix the map at [fixId]. */
  fun startSpotLock(fixId: String) {
    val fix = fixPoints.firstOrNull { it.id == fixId && it.type == FixType.SPOT }
    if (fix == null) {
      ui.lockFinished(false, "That spot is not in this map.")
      return
    }
    startLock(LockMode.USE, fix)
  }

  /** "Add a spot here" → Lock. Needs a located map, or no map at all (the spot becomes home). */
  fun startSpotCapture() {
    if (mapId != null && tWM == null) {
      ui.lockFinished(false, "Locate first (look at the picture or use a spot), then add a spot.")
      return
    }
    startLock(LockMode.CAPTURE, null)
  }

  private fun startLock(mode: LockMode, fix: FixPoint?) {
    lockMode = mode
    lockFix = fix
    lockAxis = if (mode == LockMode.USE) fix?.yawAxis ?: "X" else null
    lockCollector = SpotCollector()
    lockStartMs = SystemClock.elapsedRealtime()
  }

  fun cancelLock() {
    if (lockMode == null) return
    endLock()
  }

  private fun endLock() {
    lockMode = null
    lockFix = null
    lockAxis = null
    lockCollector = null
  }

  private fun checkLockTimeout(nowMs: Long, c: SpotCollector): Boolean {
    if (nowMs - lockStartMs <= SpotCollector.SPOT_TIMEOUT_MS) return false
    endLock()
    ui.lockFinished(false, c.failureHint())
    return true
  }

  private fun handleLock(nowMs: Long, camera: Camera) {
    val c = lockCollector ?: return
    if (checkLockTimeout(nowMs, c)) return
    // The heading comes from whichever display axis is more horizontal (+X when the phone is
    // upright, +Y when it lies on its side), so a spot can be docked in either orientation. The
    // choice is made once at capture, stored on the fix point, and reused on every later lock.
    val dp = camera.displayOrientedPose
    val xAxis = Vec3.of(dp.xAxis)
    val yAxis = Vec3.of(dp.yAxis)
    val axisName = lockAxis ?: SpotCollector.chooseAxis(xAxis, yAxis).also { lockAxis = it }
    val ax = if (axisName == "Y") yAxis else xAxis
    if (ax.horizontalLength() < SpotCollector.SPOT_MIN_AXIS) {
      c.rejectRolled()
      return
    }
    c.add(Pose4.fromAxis(Vec3.of(dp.translation), ax))
    if (!c.done) return
    val est = c.estimate() ?: return
    val mode = lockMode
    val fix = lockFix
    endLock()
    when (mode) {
      LockMode.USE -> {
        fix ?: return
        ui.lockFinished(true, "")
        if (tWM == null) applyFix(fix, est, "first", disagreement = false, accepted = null) else proposeFix(fix, est, "manual")
      }
      LockMode.CAPTURE -> {
        val home = mapId == null
        val mapPose = if (home) null else tWM?.inverse()?.compose(est.pose)
        if (!home && mapPose == null) {
          ui.lockFinished(false, "Map position lost during the capture; locate again.")
          return
        }
        ui.lockFinished(true, "")
        ui.spotCaptured(SpotCapture(newId(), est.pose, mapPose, est, home, axisName))
      }
      null -> {}
    }
  }

  /** Save a captured spot under [name]. [ghostFile] is where the renderer saved its ghost image. */
  fun addSpot(c: SpotCapture, name: String, ghostFile: String?) {
    val now = System.currentTimeMillis()
    if (c.isHome) {
      if (mapId != null) {
        ui.toast("A map was created meanwhile; add the spot again.")
        return
      }
      createMap(FixPoint.at(c.id, name, FixType.SPOT, Pose4.IDENTITY, c.yawAxis, null, ghostFile, now, sessionId), c.estimate)
    } else {
      val mp = c.mapPose ?: return
      addFixPoint(FixPoint.at(c.id, name, FixType.SPOT, mp, c.yawAxis, null, ghostFile, now, sessionId), c.estimate)
      ui.toast("Spot \"$name\" saved")
    }
  }

  // ------------------------------------------------------------------ rooms and walking

  fun createRoom(room: Room) {
    if (mapId == null) return
    rooms.add(room)
    metaVersion++
    events.log("room_created", mapOf("mapId" to mapId, "room" to room.id, "roomName" to room.name))
  }

  fun renameRoom(id: String, name: String) {
    val i = rooms.indexOfFirst { it.id == id }
    if (i < 0) return
    val old = rooms[i].name
    rooms[i] = rooms[i].copy(name = name)
    metaVersion++
    events.log("room_renamed", mapOf("mapId" to mapId, "room" to id, "from" to old, "to" to name))
  }

  /** Explicit pick from the chip or the register dialog; relabels nodes within 1 m when located. */
  fun pickRoom(roomId: String?, source: String) {
    if (mapId == null) return
    val prev = graph.currentRoom
    val n = graph.pickRoom(roomId, userMap)
    events.log(
      "room_set",
      mapOf(
        "mapId" to mapId,
        "room" to roomId,
        "roomName" to roomName(roomId),
        "previousRoom" to prev,
        "relabelled" to n,
        "source" to source,
        "located" to located,
      ),
    )
  }

  private fun feedWalk(nowMs: Long) {
    if (tWM == null || nowMs - lastWalkFeedMs < WALK_FEED_MS) return
    lastWalkFeedMs = nowMs
    val u = userMap ?: return
    val prevRoom = graph.currentRoom
    val v = graph.visit(u, sessionId, System.currentTimeMillis())
    if (v.followedRoom != null) {
      events.log(
        "room_auto",
        mapOf(
          "mapId" to mapId,
          "room" to v.followedRoom,
          "roomName" to roomName(v.followedRoom),
          "previousRoom" to prevRoom,
          "relabelled" to 0,
          "node" to v.nodeId,
        ),
      )
    }
  }

  // ------------------------------------------------------------------ snapshots

  /**
   * Immutable copy of the map for the UI and the store, rebuilt only when something changed.
   * Called from the renderer's 5 Hz update; a changed map is handed to [MapStore] for the autosave.
   */
  fun mapSnapshot(): HomeMap? {
    val id = mapId ?: return null
    val key = graph.version to metaVersion
    if (key == snapshotKey && snapshot != null) return snapshot
    val m = HomeMap(HomeMap.VERSION, id, mapCreatedMs, fixPoints.toList(), rooms.toList(), graph.nodes(), graph.edges())
    val firstBuild = snapshotKey == null
    snapshotKey = key
    snapshot = m
    // The first build after loading is the file's own content: nothing new to save.
    if (!firstBuild || metaVersion > 0 || graph.version > 0) mapStore.publish(m)
    return m
  }

  fun state(): State {
    val m = mapSnapshot()
    val ignored =
      fixPoints.firstOrNull {
        it.type == FixType.IMAGE && it.widthM != null && abs(it.widthM - refWidthM) > WIDTH_TOLERANCE_M
      }
    val note =
      if (ignored != null && usableImageFix() == null)
        "\"%s\" was saved with the picture %.1f cm wide; the setting is now %.1f cm, so it is ignored. Set the width back to use it."
          .format(ignored.name, ignored.widthM!! * 100, refWidthM * 100)
      else null
    return State(
      map = m,
      located = located,
      fixName = activeFix?.name,
      fixType = activeFix?.type,
      metersSinceFix = drift.proxies(SystemClock.elapsedRealtime())?.metersSinceFix,
      currentRoomId = graph.currentRoom,
      imageFixNote = note,
      lockActive = lockMode != null,
      pictureUsable = usableImageFix() != null,
      setHomeDeclined = setHomeDeclined,
    )
  }
}
