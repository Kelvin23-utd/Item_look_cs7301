package edu.cs6301.findback

import android.annotation.SuppressLint
import android.graphics.Bitmap
import android.graphics.Color
import android.graphics.Matrix
import android.opengl.GLSurfaceView
import android.os.Looper
import android.text.SpannableStringBuilder
import android.text.Spanned
import android.text.style.ForegroundColorSpan
import android.view.GestureDetector
import android.view.Gravity
import android.view.MotionEvent
import android.view.View
import android.widget.ArrayAdapter
import android.widget.Button
import android.widget.EditText
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.RadioButton
import android.widget.RadioGroup
import android.widget.ScrollView
import android.widget.Spinner
import android.widget.TextView
import androidx.activity.OnBackPressedCallback
import androidx.appcompat.app.AlertDialog
import androidx.lifecycle.DefaultLifecycleObserver
import androidx.lifecycle.LifecycleOwner
import com.google.ar.core.examples.java.common.samplerender.SampleRender
import edu.cs6301.findback.map.FixType
import edu.cs6301.findback.map.GhostGeometry
import edu.cs6301.findback.map.GuidanceMode
import edu.cs6301.findback.map.Localizer
import edu.cs6301.findback.map.Room
import edu.cs6301.findback.memory.ObjectRecord
import edu.cs6301.findback.policy.PositionSource
import edu.cs6301.findback.policy.VerificationPolicy
import edu.cs6301.findback.scan.InventoryText
import edu.cs6301.findback.scan.ScanEnd
import edu.cs6301.findback.study.CaptureRules
import edu.cs6301.findback.study.GroundTruth
import edu.cs6301.findback.study.Mobility
import edu.cs6301.findback.study.StudySettings
import edu.cs6301.findback.study.VerifyLabel
import edu.cs6301.findback.trace.UiTrace
import edu.cs6301.findback.ui.Action
import edu.cs6301.findback.ui.FoundAnswer
import edu.cs6301.findback.ui.GuideInput
import edu.cs6301.findback.ui.Screen
import edu.cs6301.findback.ui.Tone
import edu.cs6301.findback.ui.UiButton
import edu.cs6301.findback.ui.UiInput
import edu.cs6301.findback.ui.UiModel
import kotlin.math.abs

/**
 * What the GL thread publishes about 5 times a second for the screen (v0.6). The rest of the
 * screen state (query, "Moved it", scan, lock bar) is read live on the UI thread.
 */
class UiFrame(
  val st: Localizer.State,
  val tracking: Boolean,
  /** A capture is being analysed. */
  val busy: Boolean,
  val itemCount: Int,
  /** The query this frame's guidance describes, or null. */
  val queryId: String?,
  /** The guide card's cue for [queryId], or null. */
  val cue: String?,
  val scanItems: Int,
  val scanHint: String?,
  val mini: MiniMapView.Data?,
)

/**
 * All Android UI: overlays, the top status area, the bottom card and buttons, and dialogs. The
 * renderer owns the AR state. v0.6: the screen state, the instruction line and the bottom buttons
 * come from [UiModel] (docs/PLAN_v0.6_ui.md).
 */
@SuppressLint("ClickableViewAccessibility")
class MainView(val activity: MainActivity, private val renderer: FindBackRenderer) : DefaultLifecycleObserver {
  companion object {
    val GREEN = Color.rgb(0x34, 0xD3, 0x99)
    val AMBER = Color.rgb(0xFF, 0xC1, 0x07)
    val RED = Color.rgb(0xEF, 0x53, 0x50)
    val GREY = Color.rgb(0x9E, 0x9E, 0x9E)

    fun toneColor(t: Tone): Int =
      when (t) {
        Tone.NORMAL -> Color.WHITE
        Tone.GOOD -> GREEN
        Tone.WARN -> AMBER
        Tone.BAD -> RED
        Tone.MUTED -> GREY
      }

    /** A short message shows this long; one longer than [LONG_MESSAGE] characters a bit longer. */
    const val BANNER_MS = 3000L
    const val BANNER_LONG_MS = 5000L
    const val LONG_MESSAGE = 60
  }

  val root: View = View.inflate(activity, R.layout.activity_main, null)
  private val surfaceView: GLSurfaceView =
    root.findViewById<GLSurfaceView>(R.id.surfaceview).apply {
      SampleRender(this, renderer, activity.assets)
    }
  val overlay: GuidanceOverlay = root.findViewById(R.id.guidanceOverlay)
  val scanOverlay: ScanOverlay = root.findViewById(R.id.scanOverlay)
  val miniMap: MiniMapView = root.findViewById(R.id.miniMap)
  private val ghostImage: ImageView = root.findViewById(R.id.ghostImage)

  // Top.
  private val statusPill: TextView = root.findViewById(R.id.statusPill)
  private val studyLine: TextView = root.findViewById(R.id.studyLine)
  private val instructionText: TextView = root.findViewById(R.id.instructionText)
  private val banner: TextView = root.findViewById(R.id.messageBanner)

  // Bottom cards.
  private val lockBar: View = root.findViewById(R.id.lockBar)
  private val lockTitle: TextView = root.findViewById(R.id.lockTitle)
  private val lockText: TextView = root.findViewById(R.id.lockText)
  private val guideCard: View = root.findViewById(R.id.guideCard)
  private val guideThumb: ImageView = root.findViewById(R.id.guideThumb)
  private val guideTitle: TextView = root.findViewById(R.id.guideTitle)
  private val guideSubtitle: TextView = root.findViewById(R.id.guideSubtitle)
  private val guideCue: TextView = root.findViewById(R.id.guideCue)
  private val guideStatus: TextView = root.findViewById(R.id.guideStatus)
  private val guideLocate: Button = root.findViewById(R.id.guideLocate)
  private val guidePolicy: TextView = root.findViewById(R.id.guidePolicy)
  private val scanBar: View = root.findViewById(R.id.scanBar)
  private val scanTitle: TextView = root.findViewById(R.id.scanTitle)
  private val scanHint: TextView = root.findViewById(R.id.scanHint)
  private val welcomeCard: View = root.findViewById(R.id.welcomeCard)
  private val welcomePicture: View = root.findViewById(R.id.welcomePicture)

  // Bottom bar.
  private val secondaryButton: Button = root.findViewById(R.id.secondaryButton)
  private val primaryButton: Button = root.findViewById(R.id.primaryButton)
  private val buttonGap: View = root.findViewById(R.id.buttonGap)

  /** The screen as last drawn, for Back. */
  private var screen: Screen? = null

  /** v0.7: [screen]'s name for the GL thread's `pose.csv`. */
  @Volatile var screenName: String? = null
    private set

  /** v0.7 detailed trace: `ui_screen`, `ui_action`, `ui_search`, `ui_message` (no-ops when off). */
  val uiTrace = UiTrace(activity.events, activity.traceOn) { screen?.name }

  val dialogs = LocationDialogs(activity, renderer, this)
  val items = ItemsPanel(activity, root, uiTrace)

  /** The bottom-bar actions other than Lock and Cancel (the renderer handles them). */
  var onAction: ((Action) -> Unit)? = null

  /** True while the spot lock bar (and maybe the ghost) is shown; camera taps are ignored then. */
  var lockModeActive = false
    private set
  private var lockInstructions = ""
  private var onLock: (() -> Unit)? = null
  private var onLockCancel: (() -> Unit)? = null
  /** v0.7 `lock_attempt.attemptInBar`: Lock presses since the lock bar opened (UI thread). */
  var lockPressesInBar = 0
    private set

  /** The latest frame from the GL thread; null until the first one. */
  private var frame: UiFrame? = null

  init {
    val gestures =
      GestureDetector(
        activity,
        object : GestureDetector.SimpleOnGestureListener() {
          override fun onSingleTapUp(e: MotionEvent): Boolean {
            overlay.flashTap(e.x, e.y)
            renderer.onTap(e.x, e.y)
            return true
          }

          override fun onDown(e: MotionEvent) = true
        }
      )
    surfaceView.setOnTouchListener { _, event -> gestures.onTouchEvent(event) }

    // The pill picks the room when located, locates when not, and opens the menu without a map.
    statusPill.setOnClickListener {
      val st = renderer.uiState
      val map = st?.map
      uiTrace.action("pill", if (st == null) "starting" else if (map == null) "sheet" else if (!st.located) "locate" else "room")
      if (st == null) return@setOnClickListener toast("Starting…")
      if (map == null) dialogs.showSheet(st)
      else if (!st.located) dialogs.locate(st)
      else dialogs.pickRoom(map.rooms, st.currentRoomId) { id, newRoom -> renderer.requestPickRoom(id, newRoom, "chip") }
    }
    root.findViewById<View>(R.id.menuButton).setOnClickListener {
      uiTrace.action("menu")
      val st = renderer.uiState
      if (st == null) toast("Starting…") else dialogs.showSheet(st)
    }
    banner.setOnClickListener {
      uiTrace.action("banner_dismiss")
      hideBanner.run()
    }

    secondaryButton.setOnClickListener { v -> (v.tag as? Action)?.let { dispatch(it) } }
    primaryButton.setOnClickListener { v -> (v.tag as? Action)?.let { dispatch(it) } }
    guideLocate.setOnClickListener { dispatch(Action.LOCATE, "guide_card") }
    guideStatus.setOnClickListener {
      uiTrace.action("guide_status", if (guideOpensResult) "result" else null)
      showLastCapture()
    }

    // Section 8: Back closes the Items panel, cancels the lock bar or "Moved it", and does not
    // leave the app during a search or a scan.
    activity.onBackPressedDispatcher.addCallback(
      activity,
      object : OnBackPressedCallback(true) {
        override fun handleOnBackPressed() {
          uiTrace.action("back")
          when {
            items.isOpen -> items.close("back")
            lockModeActive -> dispatch(Action.CANCEL_LOCK, "back")
            screen == Screen.FINDING -> toast("Press Done to end the search")
            screen == Screen.MEASURING -> toast("Tap the true centre, or press Skip")
            screen == Screen.SCANNING -> toast("Press Stop scan first")
            screen == Screen.UPDATING -> dispatch(Action.CANCEL_UPDATE, "back")
            else -> {
              isEnabled = false
              activity.onBackPressedDispatcher.onBackPressed()
              isEnabled = true
            }
          }
        }
      },
    )
  }

  override fun onResume(owner: LifecycleOwner) = surfaceView.onResume()

  override fun onPause(owner: LifecycleOwner) = surfaceView.onPause()

  fun post(action: () -> Unit) {
    root.post { action() }
  }

  /** [source] (v0.7 `ui_action.detail`): null for the bottom bar, else "guide_card" or "back". */
  private fun dispatch(a: Action, source: String? = null) {
    uiTrace.action(a.name, source) // v0.7: every bottom-bar button, Locate in the guide card, Back's cancels
    when (a) {
      Action.LOCK -> {
        lockPressesInBar++
        lockText.text = "Hold still…"
        onLock?.invoke()
      }
      Action.CANCEL_LOCK -> {
        onLockCancel?.invoke()
        exitLockMode()
      }
      else -> onAction?.invoke(a)
    }
  }

  // ------------------------------------------------------------------ message banner

  private val hideBanner = Runnable { banner.visibility = View.GONE }

  /**
   * Section 5.1: a short message in the banner below the instruction line, for 3 s (5 s when
   * long). A new message replaces the old one; a tap dismisses it. Any thread.
   */
  fun toast(message: String) {
    if (Looper.myLooper() != Looper.getMainLooper()) {
      root.post { toast(message) }
      return
    }
    if (message.isEmpty()) return
    uiTrace.message(message) // v0.7: every banner message passes here
    banner.removeCallbacks(hideBanner)
    banner.text = message
    banner.visibility = View.VISIBLE
    banner.postDelayed(hideBanner, if (message.length > LONG_MESSAGE) BANNER_LONG_MS else BANNER_MS)
  }

  // ------------------------------------------------------------------ screen

  /** UI thread: a new frame from the GL thread. */
  fun render(f: UiFrame) {
    frame = f
    miniMap.setData(f.mini)
    refresh()
  }

  /**
   * UI thread: draw the screen from the latest frame and the live state. Called for every frame and
   * at once after every change of query, "Moved it", scan or lock bar, so buttons switch without
   * waiting for the next frame.
   */
  fun refresh() {
    val f = frame
    if (f == null) {
      statusPill.text = pillText("Starting…", GREY, false)
      return
    }
    val st = f.st
    val q = renderer.query
    val upd = renderer.updateTarget
    val scan = renderer.scan.active
    val hasSpots = st.map?.fixPoints?.any { it.type == FixType.SPOT } == true
    val input =
      UiInput(
        lockActive = lockModeActive,
        queryActive = q != null,
        measuring = q?.measuring == true,
        updating = upd != null,
        scanning = scan != null,
        hasMap = st.hasMap,
        located = st.located,
        tracking = f.tracking,
        busy = f.busy,
        roomName = st.roomName(st.currentRoomId),
        itemCount = f.itemCount,
        pictureUsable = st.pictureUsable,
        hasSpots = hasSpots,
        guidanceMode = q?.mode,
        itemName = q?.record?.name ?: upd?.name,
        studyMode = q?.studyMode == true,
        verifyAllowed = q?.let { CaptureRules.verifyAllowed(it.captureMode, it.captures) } ?: true,
        resolvable = q?.resolvable == true,
        policyRecommends = q?.decision?.verify == true,
      )
    val s = UiModel.screen(input)
    if (s != screen) uiTrace.screenChanged(screen?.name, s.name) // v0.7
    screen = s
    screenName = s.name

    // Top.
    val pill = UiModel.pill(input)
    statusPill.text = pillText(pill.text, toneColor(pill.tone), pill.opensRoomPicker)
    val study = StudySettings.statusLine(activity.studyTag, activity.captureMode)
    studyLine.text = study
    studyLine.visibility = if (study == null) View.GONE else View.VISIBLE
    val instruction = UiModel.instruction(input)
    instructionText.text = instruction
    instructionText.visibility = if (instruction == null) View.GONE else View.VISIBLE

    // Bottom card.
    lockBar.visibility = if (s == Screen.LOCKING) View.VISIBLE else View.GONE
    // The card waits for enterGuideMode of this query, so it never shows the previous item.
    val guideReady =
      when (s) {
        Screen.FINDING, Screen.MEASURING -> q != null && guideQueryId == q.id
        Screen.UPDATING -> true
        else -> false
      }
    guideCard.visibility = if (guideReady) View.VISIBLE else View.GONE
    scanBar.visibility = if (s == Screen.SCANNING) View.VISIBLE else View.GONE
    welcomeCard.visibility = if (s == Screen.NO_MAP) View.VISIBLE else View.GONE
    when (s) {
      Screen.FINDING, Screen.MEASURING -> if (q != null) drawGuide(q, f, st.located, hasSpots)
      Screen.UPDATING -> if (upd != null) drawUpdate(upd, st.located, hasSpots)
      Screen.SCANNING -> {
        scanTitle.text = UiModel.scanTitle(scan?.itemCount ?: f.scanItems)
        scanHint.text = f.scanHint ?: UiModel.scanHint(st.located, st.lockActive, false)
      }
      // Without a map the picture is not a fix point yet, but it can still set home (once asked).
      Screen.NO_MAP -> welcomePicture.visibility = if (st.setHomeDeclined) View.GONE else View.VISIBLE
      else -> {}
    }

    // Bottom bar.
    val b = UiModel.buttons(s, input)
    bind(secondaryButton, b.secondary)
    bind(primaryButton, b.primary)
    buttonGap.visibility = if (b.secondary != null && b.primary != null) View.VISIBLE else View.GONE
  }

  private fun bind(button: Button, spec: UiButton?) {
    if (spec == null) {
      button.visibility = View.GONE
      button.tag = null
      return
    }
    button.tag = spec.action
    if (button.text.toString() != spec.label) button.text = spec.label
    button.visibility = View.VISIBLE
  }

  private fun pillText(text: String, color: Int, arrow: Boolean): CharSequence {
    val s = SpannableStringBuilder("● ")
    s.setSpan(ForegroundColorSpan(color), 0, 1, Spanned.SPAN_EXCLUSIVE_EXCLUSIVE)
    s.append(text)
    if (arrow) s.append(" ▾")
    return s
  }

  // ------------------------------------------------------------------ spot lock / ghost

  /**
   * Show the lock bar (and the ghost image when given). Lock and Cancel are the bottom buttons; the
   * localizer ends the mode through [exitLockMode].
   */
  fun enterLockMode(title: String, text: String, ghost: Bitmap?, onLock: () -> Unit, onCancel: () -> Unit) {
    lockModeActive = true
    lockPressesInBar = 0
    lockInstructions = text
    lockTitle.text = title
    lockText.text = text
    this.onLock = onLock
    onLockCancel = onCancel
    if (ghost != null) {
      ghostImage.setImageBitmap(ghost)
      placeGhost(ghost)
      ghostImage.visibility = View.VISIBLE
    } else {
      // Entering "Add a spot here" from an open Locate must not keep the old ghost.
      ghostImage.visibility = View.GONE
      ghostImage.setImageBitmap(null)
    }
    refresh()
  }

  /**
   * Same view-to-image mapping as the camera background: the renderer publishes where three view
   * corners fall in the CPU image; fit the affine matrix from ghost pixels to view pixels.
   */
  private fun placeGhost(ghost: Bitmap) {
    val m = renderer.ghostMapping
    if (m == null) {
      ghostImage.scaleType = ImageView.ScaleType.FIT_XY
      return
    }
    val src = GhostGeometry.ghostPointsForViewCorners(m.cornersImageNorm, m.rotation, ghost.width, ghost.height)
    val dst = GhostGeometry.viewCorners(m.viewWidth, m.viewHeight)
    val matrix = Matrix()
    if (matrix.setPolyToPoly(src, 0, dst, 0, 3)) {
      ghostImage.scaleType = ImageView.ScaleType.MATRIX
      ghostImage.imageMatrix = matrix
    } else {
      ghostImage.scaleType = ImageView.ScaleType.FIT_XY
    }
  }

  /** A lock attempt failed: keep the bar (and ghost) so the user can line up and press Lock again. */
  fun lockFailed() {
    if (lockModeActive) lockText.text = lockInstructions
  }

  fun exitLockMode() {
    if (!lockModeActive) return
    lockModeActive = false
    onLock = null
    onLockCancel = null
    ghostImage.visibility = View.GONE
    ghostImage.setImageBitmap(null)
    refresh()
  }

  // ------------------------------------------------------------------ guide card

  /** The query the guide card was set up for, and its fixed parts. */
  private var guideQueryId: String? = null
  private var guideRoomName: String? = null
  private var guideLastCue: String? = null
  private var guideOpensResult = false

  /** Everyday mode: the last capture of the query, shown when its status line is tapped. */
  private class LastCapture(val queryId: String, val stored: Bitmap?, val fresh: Bitmap, val similarity: Float, val confirmed: Boolean)

  private var lastCapture: LastCapture? = null

  fun enterGuideMode(q: Query, thumb: Bitmap?, roomName: String?) {
    guideQueryId = q.id
    guideRoomName = roomName
    guideLastCue = null
    lastCapture = null
    guideThumb.setImageBitmap(thumb)
    guideTitle.text = q.record.name
    guideCue.text = UiModel.cue(q.mode, roomName, q.routeM, q.straightM, renderer.uiState?.located == true, null)
    val hasLocation = q.inputs.positionSource != PositionSource.NONE
    val decision = q.decision
    if (q.studyMode) {
      // Study mode keeps the v0.5 policy line with its reasons.
      guidePolicy.text =
        when {
          !hasLocation -> "Policy: no location at query start, so no verification decision"
          decision.verify -> "Verification recommended: " + decision.reasons.joinToString("; ")
          else -> "Policy: no verification needed (" + decision.reasons.joinToString("; ") + ")"
        }
      guidePolicy.setTextColor(if (decision.verify) AMBER else GREY)
      guidePolicy.visibility = View.VISIBLE
    } else {
      guidePolicy.visibility = View.GONE
    }
    refresh()
  }

  private fun drawGuide(q: Query, f: UiFrame, located: Boolean, hasSpots: Boolean) {
    if (guideQueryId != q.id) return // enterGuideMode has not run yet for this query
    guideSubtitle.text = UiModel.guideSubtitle(guideRoomName, InventoryText.ago(q.record.lastSeenMs, System.currentTimeMillis()))
    if (f.queryId == q.id && f.cue != null) guideLastCue = f.cue
    guideLastCue?.let { if (guideCue.text.toString() != it) guideCue.text = it }
    val status =
      UiModel.guideStatus(
        GuideInput(
          studyMode = q.studyMode,
          measuring = q.measuring,
          displaced = q.truth == GroundTruth.DISPLACED,
          lastOutcome = q.lastOutcome,
          lastSimilarity = q.lastSimilarity,
          located = located,
          resolvable = q.resolvable,
          near = q.mode == GuidanceMode.NEAR,
          hasSpots = hasSpots,
          recommend = q.decision.verify && q.inputs.positionSource != PositionSource.NONE,
        )
      )
    guideStatus.text = status.text
    guideStatus.setTextColor(toneColor(status.tone))
    guideStatus.visibility = if (status.text.isEmpty()) View.GONE else View.VISIBLE
    guideOpensResult = status.opensResult
    guideLocate.visibility = if (status.showLocate && !q.measuring) View.VISIBLE else View.GONE
  }

  /** Everyday mode: keep the capture for a tap on the status line (instead of a modal). */
  fun setLastCapture(queryId: String, stored: Bitmap?, fresh: Bitmap, similarity: Float, confirmed: Boolean) {
    if (renderer.query?.id != queryId) return
    lastCapture = LastCapture(queryId, stored, fresh, similarity, confirmed)
    refresh()
  }

  private fun showLastCapture() {
    if (!guideOpensResult) return
    val c = lastCapture ?: return
    if (renderer.query?.id != c.queryId) return
    showVerifyResult(c.stored, c.fresh, c.similarity, c.confirmed, study = false)
  }

  fun exitGuideMode() {
    guideQueryId = null
    guideLastCue = null
    lastCapture = null
    guideOpensResult = false
    guideThumb.setImageBitmap(null)
    guidePolicy.visibility = View.GONE
    guideLocate.visibility = View.GONE
    overlay.setTarget(null)
    refresh()
  }

  /** "Moved it" (v0.4 Update location): the guide card shows the item and asks for a tap on it where it is now. */
  fun enterUpdateMode(rec: ObjectRecord, thumb: Bitmap?, roomName: String?) {
    guideQueryId = null
    lastCapture = null
    guideOpensResult = false
    guideThumb.setImageBitmap(thumb)
    guideTitle.text = rec.name
    guideSubtitle.text =
      "Old place: " + (roomName?.let { "$it · " } ?: "") + "saved " + InventoryText.ago(rec.lastSeenMs, System.currentTimeMillis())
    guideCue.text = "Tap it where it is now"
    guidePolicy.visibility = View.GONE
    refresh()
  }

  private fun drawUpdate(rec: ObjectRecord, located: Boolean, hasSpots: Boolean) {
    if (guideQueryId != null) return
    if (guideTitle.text.toString() != rec.name) guideTitle.text = rec.name
    val locate = !located && hasSpots
    guideStatus.text =
      if (locate) "Not located: press Locate first, or the new place is kept for this session only"
      else "Saving replaces the old place. Cancel keeps it."
    guideStatus.setTextColor(if (locate) AMBER else GREY)
    guideStatus.visibility = View.VISIBLE // a search before may have hidden it
    guideLocate.visibility = if (locate) View.VISIBLE else View.GONE
  }

  fun exitUpdateMode() = exitGuideMode()

  // ------------------------------------------------------------------ dialogs

  /** Home sheet → Settings → Picture width: set the measured width of the reference picture. */
  fun showReferenceSettings() {
    val current = activity.refWidthM * 100f
    val edit =
      EditText(activity).apply {
        inputType = android.text.InputType.TYPE_CLASS_NUMBER or android.text.InputType.TYPE_NUMBER_FLAG_DECIMAL
        setText("%.1f".format(current))
        selectAll()
      }
    val box =
      LinearLayout(activity).apply {
        orientation = LinearLayout.VERTICAL
        setPadding(dp(20), dp(8), dp(20), 0)
        addView(
          TextView(activity).apply {
            text =
              "Width of the Earth picture as it appears in the room, in cm.\n\n" +
                "Printed A4 sheet: 20.0\n" +
                "iPad mini 6/7, full screen: about 11.6\n" +
                "iPad mini 5, full screen: about 12.0\n\n" +
                "Measure with a ruler if you can. Set this before saving objects and keep the " +
                "picture in the same place; the app restarts to apply it."
          }
        )
        addView(edit)
      }
    uiTrace.dialog(activity, "picture_width")
      .setTitle("Picture width")
      .setView(box)
      .setPositiveButton("Save") { _, _ ->
        val cm = edit.text.toString().toFloatOrNull()
        if (cm == null || cm < 3f || cm > 100f) {
          toast("Enter a width between 3 and 100 cm.")
        } else if (abs(cm - current) > 0.05f) {
          activity.setRefWidthM(cm / 100f)
        }
      }
      .setNegativeButton("Cancel", null)
      .show()
  }

  /**
   * Section 6.6: the photo, warnings in plain words, the name (pre-filled with [defaultName]; a tap
   * selects it, so typing replaces it) and the room. Study mode adds the v0.5 debug line and the
   * mobility choice; in everyday mode the mobility is the record's value on an update, else
   * [Mobility.DEFAULT].
   *
   * [onSave] gets the room chosen in the dialog, whether the user picked it explicitly (an explicit
   * pick also sets the current room and relabels nearby trail), a room created here, if any, and
   * the mobility class. `typed` is true only when the saved name differs from [defaultName] (a new
   * item) or from the record's name ([update], "Moved it").
   */
  fun showRegisterDialog(
    d: RegistrationDraft,
    defaultName: String,
    rooms: List<Room>,
    update: ObjectRecord? = null,
    studyMode: Boolean,
    onSave: (name: String, typed: Boolean, roomId: String?, picked: Boolean, newRoom: Room?, mobility: String) -> Unit,
    onCancel: () -> Unit
  ) {
    val pad = dp(16)
    val layout =
      LinearLayout(activity).apply {
        orientation = LinearLayout.VERTICAL
        setPadding(pad, pad / 2, pad, 0)
        // Takes the dialog's first focus, so the first tap on the name selects all of it.
        isFocusableInTouchMode = true
      }
    layout.addView(
      ImageView(activity).apply {
        setImageBitmap(d.crop)
        adjustViewBounds = true
        maxHeight = dp(220)
      }
    )
    if (d.captureDistance > 1.5f) {
      layout.addView(
        TextView(activity).apply {
          textSize = 13f
          setTextColor(AMBER)
          setPadding(0, dp(6), 0, 0)
          text = "%.1f m away: stand 0.5–1 m from the object for a reliable photo".format(d.captureDistance)
        }
      )
    }
    if (d.mapPosition == null) {
      layout.addView(
        TextView(activity).apply {
          textSize = 13f
          setTextColor(RED)
          setPadding(0, dp(6), 0, 0)
          text = "Not located: this item is saved for this session only. Press Locate first to put it on the home map."
        }
      )
    }
    val baseline = update?.name ?: defaultName
    val edit =
      EditText(activity).apply {
        hint = "Name"
        setSingleLine()
        setText(baseline)
        setSelectAllOnFocus(true)
      }
    layout.addView(edit)
    // Room row (only once a home map exists).
    var roomId = d.defaultRoomId ?: update?.roomId?.takeIf { old -> rooms.any { it.id == old } }
    var picked = false
    var newRoom: Room? = null
    val allRooms = rooms.toMutableList()
    val roomRow =
      TextView(activity).apply {
        textSize = 15f
        minHeight = dp(44)
        gravity = Gravity.CENTER_VERTICAL
        setPadding(0, dp(6), 0, dp(4))
        setTextColor(Color.rgb(0x1E, 0x88, 0xE5))
      }
    fun showRoom() {
      roomRow.text = "Room: " + (allRooms.firstOrNull { it.id == roomId }?.name ?: "none") + " (tap to change)"
    }
    if (d.hasMap) {
      showRoom()
      roomRow.setOnClickListener {
        uiTrace.action("dialog:register", "room")
        dialogs.pickRoom(allRooms, roomId) { id, created ->
          roomId = id
          picked = true
          if (created != null) {
            newRoom = created
            allRooms.add(created)
          }
          showRoom()
        }
      }
      layout.addView(roomRow)
    }
    var mobilityGroup: RadioGroup? = null
    val mobilityOf = HashMap<Int, String>()
    val initialMobility = Mobility.orDefault(update?.mobility)
    if (studyMode) {
      val detectorText =
        if (d.rawLabel.isEmpty()) "no category" else "%s (%.2f)".format(d.rawLabel, d.rawConfidence)
      layout.addView(
        TextView(activity).apply {
          textSize = 12f
          setTextColor(GREY)
          setPadding(0, dp(6), 0, 0)
          text =
            "Detector: $detectorText · crop: ${d.boxSource}\n" +
              "%.2f m away · %s · map: %s".format(
                d.captureDistance,
                d.hitType,
                if (d.mapPosition != null) "yes" else "NO (this session only)"
              )
        }
      )
      // v0.4 (S6): mobility class, a pre-capture signal for the policies compared offline.
      layout.addView(
        TextView(activity).apply {
          textSize = 13f
          setPadding(0, dp(8), 0, 0)
          text = "How often does it move?"
        }
      )
      val group = RadioGroup(activity).apply { orientation = RadioGroup.HORIZONTAL }
      for ((value, label) in Mobility.CHOICES) {
        val button =
          RadioButton(activity).apply {
            id = View.generateViewId()
            text = label
          }
        mobilityOf[button.id] = value
        group.addView(button)
        if (value == initialMobility) group.check(button.id)
      }
      layout.addView(group)
      mobilityGroup = group
    }

    uiTrace.dialog(activity, if (update != null) "register_update" else "register")
      .setTitle(if (update != null) "New place for \"" + update.name + "\"" else "Remember this object?")
      .setView(ScrollView(activity).apply { addView(layout) }) // the name field stays reachable above the keyboard
      .setPositiveButton("Save") { _, _ ->
        val saved = edit.text.toString().trim().ifEmpty { defaultName }
        onSave(
          saved,
          saved != baseline,
          roomId,
          picked,
          newRoom?.takeIf { it.id == roomId },
          mobilityGroup?.let { mobilityOf[it.checkedRadioButtonId] } ?: initialMobility,
        )
      }
      .setNegativeButton("Cancel") { _, _ -> onCancel() }
      .setOnCancelListener { onCancel() }
      .show()
    layout.requestFocus()
  }

  /** Section 6.7: the counts of a finished scan. [onItems] opens the Items panel. */
  fun showScanSummary(end: ScanEnd, onItems: () -> Unit) {
    val s = end.summary
    val box =
      LinearLayout(activity).apply {
        orientation = LinearLayout.VERTICAL
        setPadding(dp(24), dp(8), dp(24), 0)
      }
    fun line(text: String, size: Float, color: Int) =
      box.addView(
        TextView(activity).apply {
          this.text = text
          textSize = size
          setTextColor(color)
          setPadding(0, dp(3), 0, dp(3))
        }
      )
    val dark = Color.rgb(0x21, 0x21, 0x21)
    if (s == null) {
      line("The scan could not be summarised" + (end.error?.let { ": $it" } ?: "") + ".", 16f, dark)
    } else {
      for (t in UiModel.scanSummaryLines(s.newCount, s.seenAgain, s.missed, s.tentativeDropped)) line(t, 16f, dark)
    }
    if (end.reason == "paused") line("The scan ended because the app left the screen.", 14f, dark)
    line(UiModel.scanSummaryDetails(end.durationMs, end.metersWalked, end.keyframes, end.depthCheckMedianM), 12f, GREY)
    uiTrace.dialog(activity, "scan_summary")
      .setTitle("Scan finished")
      .setView(box)
      .setPositiveButton("OK", null)
      .setNeutralButton("See items") { _, _ -> onItems() }
      .show()
  }

  /**
   * Both photos of a capture. [study]: the v0.5 modal with the similarity (shown after every
   * capture in study mode; [onDismiss] runs once it is closed, for the BENCHMARK label). Everyday
   * mode shows it from the guide card's status line, in plain words.
   */
  fun showVerifyResult(
    stored: Bitmap?,
    fresh: Bitmap,
    similarity: Float,
    confirmed: Boolean,
    study: Boolean = true,
    onDismiss: (() -> Unit)? = null,
  ) {
    val pad = dp(12)
    val row = LinearLayout(activity).apply { orientation = LinearLayout.HORIZONTAL; gravity = Gravity.CENTER }
    for ((bmp, caption) in listOf(stored to "Remembered", fresh to "Now")) {
      val col =
        LinearLayout(activity).apply {
          orientation = LinearLayout.VERTICAL
          gravity = Gravity.CENTER_HORIZONTAL
          setPadding(pad, pad, pad, 0)
        }
      col.addView(
        ImageView(activity).apply {
          setImageBitmap(bmp)
          layoutParams = LinearLayout.LayoutParams(dp(120), dp(120))
          scaleType = ImageView.ScaleType.CENTER_CROP
        }
      )
      col.addView(TextView(activity).apply { text = caption })
      row.addView(col)
    }
    val title: String
    val message: String
    if (study) {
      title = if (confirmed) "Verified" else "Not verified"
      message =
        if (confirmed) "Seen here now. Similarity %.2f ≥ %.2f.".format(similarity, VerificationPolicy.MATCH_THRESHOLD)
        else
          ("Not confirmed: insufficient evidence (similarity %.2f < %.2f). " +
              "The item may still be here; showing the last-seen location.")
            .format(similarity, VerificationPolicy.MATCH_THRESHOLD)
    } else {
      title = if (confirmed) "Still here" else "Could not confirm"
      message =
        if (confirmed) "The view at the ring matches the remembered photo."
        else "The view at the ring does not match the remembered photo well enough. The item may still be here: " +
          "the ring shows where it was last seen."
    }
    uiTrace.dialog(activity, if (study) "verify_result" else "last_capture")
      .setTitle(title)
      .setView(row)
      .setMessage(message)
      .setPositiveButton("OK", null)
      .setOnDismissListener { onDismiss?.invoke() }
      .show()
  }

  /**
   * Section 6.3, everyday mode: "Did you find it?". [onAnswer] gets the answer; Back or "Keep
   * looking" returns to the search with nothing logged.
   */
  fun askFoundIt(onAnswer: (FoundAnswer) -> Unit) {
    val answers = FoundAnswer.values()
    uiTrace.dialog(activity, "found_it")
      .setTitle("Did you find it?")
      .setItems(answers.map { it.label }.toTypedArray()) { _, i -> onAnswer(answers[i]) }
      .setNegativeButton("Keep looking", null)
      .show()
  }

  /**
   * v0.4 (S4), study mode: the five ground-truth values of the study spec, section 2. Not
   * cancelable, so the query always ends with exactly one answer ("Skip" is the way out).
   */
  fun askGroundTruth(name: String, onAnswer: (String) -> Unit) {
    uiTrace.dialog(activity, "ground_truth")
      .setTitle("Ground truth: \"$name\"")
      .setItems(GroundTruth.CHOICES.map { it.second }.toTypedArray()) { _, i -> onAnswer(GroundTruth.CHOICES[i].first) }
      .setCancelable(false)
      .show()
  }

  /**
   * v0.4 PAIRED (S2): Done without a capture while a capture is possible. "Back", or dismissing the
   * dialog, returns to the search with nothing changed; [onFinish] goes on to the ground truth.
   */
  fun askCaptureFirst(name: String, onFinish: () -> Unit) {
    uiTrace.dialog(activity, "capture_first")
      .setTitle("Capture once before finishing?")
      .setMessage(
        "PAIRED mode takes one capture per query, and \"$name\" has none yet. " +
          "Go back and press Verify with the ring in view, or finish without a capture."
      )
      .setPositiveButton("Back", null)
      .setNegativeButton("Finish without capture") { _, _ -> onFinish() }
      .show()
  }

  /** v0.4 BENCHMARK (S3): the capture condition stays as last chosen until the app restarts. */
  private var benchmarkCondition = VerifyLabel.DEFAULT_CONDITION

  /**
   * v0.4 BENCHMARK (S3): what is at the marker, asked after each verification result. [onAnswer] is
   * called exactly once: with the label, or with null when the dialog is dismissed without one.
   */
  fun askVerifyLabel(name: String, onAnswer: (label: String?, condition: String) -> Unit) {
    var answered = false
    var dialog: AlertDialog? = null
    val spinner =
      Spinner(activity).apply {
        adapter = ArrayAdapter(activity, android.R.layout.simple_spinner_dropdown_item, VerifyLabel.CONDITIONS)
        setSelection(VerifyLabel.CONDITIONS.indexOf(benchmarkCondition).coerceAtLeast(0))
      }
    fun answer(label: String?) {
      if (answered) return
      answered = true
      uiTrace.action("dialog:verify_label", label ?: "dismissed")
      benchmarkCondition = VerifyLabel.CONDITIONS.getOrElse(spinner.selectedItemPosition) { benchmarkCondition }
      onAnswer(label, benchmarkCondition)
    }
    val box =
      LinearLayout(activity).apply {
        orientation = LinearLayout.VERTICAL
        setPadding(dp(20), dp(8), dp(20), 0)
        addView(
          TextView(activity).apply {
            textSize = 13f
            text = "Target: $name"
          }
        )
        addView(
          LinearLayout(activity).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
            addView(
              TextView(activity).apply {
                textSize = 15f
                text = "Condition:"
              }
            )
            addView(spinner)
          }
        )
        for ((value, label) in VerifyLabel.CHOICES) {
          addView(
            Button(activity).apply {
              text = label
              isAllCaps = false
              setOnClickListener {
                answer(value)
                dialog?.dismiss()
              }
            }
          )
        }
      }
    dialog =
      AlertDialog.Builder(activity)
        .setTitle("What is at the marker right now?")
        .setView(box)
        .setOnDismissListener { answer(null) }
        .show()
  }

  private fun dp(v: Int) = (v * activity.resources.displayMetrics.density).toInt()
}
