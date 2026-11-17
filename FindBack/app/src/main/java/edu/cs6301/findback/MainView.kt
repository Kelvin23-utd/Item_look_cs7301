package edu.cs6301.findback

import android.annotation.SuppressLint
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.Typeface
import android.opengl.GLSurfaceView
import android.text.Editable
import android.text.InputType
import android.text.SpannableStringBuilder
import android.text.Spanned
import android.text.TextWatcher
import android.text.style.ForegroundColorSpan
import android.text.style.RelativeSizeSpan
import android.text.style.StyleSpan
import android.view.GestureDetector
import android.view.Gravity
import android.view.MotionEvent
import android.view.View
import android.view.ViewGroup
import android.view.WindowManager
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
import androidx.appcompat.app.AlertDialog
import androidx.lifecycle.DefaultLifecycleObserver
import androidx.lifecycle.LifecycleOwner
import com.google.ar.core.examples.java.common.helpers.SnackbarHelper
import com.google.ar.core.examples.java.common.samplerender.SampleRender
import edu.cs6301.findback.map.GhostGeometry
import edu.cs6301.findback.map.GuidancePlanner
import edu.cs6301.findback.map.Localizer
import edu.cs6301.findback.map.Room
import edu.cs6301.findback.memory.ObjectRecord
import edu.cs6301.findback.policy.PolicyDecision
import edu.cs6301.findback.policy.VerificationPolicy
import edu.cs6301.findback.scan.InventoryEntry
import edu.cs6301.findback.scan.InventoryText
import edu.cs6301.findback.scan.ScanEnd
import edu.cs6301.findback.study.GroundTruth
import edu.cs6301.findback.study.Mobility
import edu.cs6301.findback.study.VerifyLabel
import java.io.File
import kotlin.math.abs

/** All Android UI: overlays, buttons and dialogs. The renderer owns the AR state. */
@SuppressLint("ClickableViewAccessibility")
class MainView(val activity: MainActivity, private val renderer: FindBackRenderer) : DefaultLifecycleObserver {
  companion object {
    val GREEN = Color.rgb(0x34, 0xD3, 0x99)
    val AMBER = Color.rgb(0xFF, 0xC1, 0x07)
    val RED = Color.rgb(0xEF, 0x53, 0x50)
    val GREY = Color.rgb(0x9E, 0x9E, 0x9E)

    fun ago(ms: Long): String {
      val s = (System.currentTimeMillis() - ms) / 1000
      return when {
        s < 60 -> "${s}s ago"
        s < 3600 -> "${s / 60} min ago"
        s < 86400 -> "${s / 3600} h ago"
        else -> "${s / 86400} d ago"
      }
    }
  }

  /** Second status dot: is the home map located in this session? */
  enum class LocState { NO_MAP, NOT_LOCATED, LOCATED }

  /** v0.5: the bottom-bar scan control. */
  enum class ScanControl { HIDDEN, START, STOP }

  /** v0.5: what the Items list does with a scanned entry. */
  class EntryActions(
    val onFind: (InventoryEntry) -> Unit,
    val onRename: (InventoryEntry, String) -> Unit,
    val onDelete: (InventoryEntry) -> Unit,
  )

  /** One row of the Items list: a tapped record or a scanned entry. */
  private class ItemRow(val record: ObjectRecord?, val entry: InventoryEntry?) {
    val lastSeenMs: Long
      get() = record?.lastSeenMs ?: entry!!.lastSeenMs
  }

  val root: View = View.inflate(activity, R.layout.activity_main, null)
  private val surfaceView: GLSurfaceView =
    root.findViewById<GLSurfaceView>(R.id.surfaceview).apply {
      SampleRender(this, renderer, activity.assets)
    }
  val overlay: GuidanceOverlay = root.findViewById(R.id.guidanceOverlay)
  private val statusText: TextView = root.findViewById(R.id.statusText)
  private val hintText: TextView = root.findViewById(R.id.hintText)
  private val guideCard: View = root.findViewById(R.id.guideCard)
  private val guideThumb: ImageView = root.findViewById(R.id.guideThumb)
  private val guideTitle: TextView = root.findViewById(R.id.guideTitle)
  private val guideSubtitle: TextView = root.findViewById(R.id.guideSubtitle)
  private val guideCue: TextView = root.findViewById(R.id.guideCue)
  private val guidePolicy: TextView = root.findViewById(R.id.guidePolicy)
  private val guideVerify: TextView = root.findViewById(R.id.guideVerify)
  val itemsButton: Button = root.findViewById(R.id.itemsButton)
  val verifyButton: Button = root.findViewById(R.id.verifyButton)
  val doneButton: Button = root.findViewById(R.id.doneButton)
  val scanButton: Button = root.findViewById(R.id.scanButton)
  val scanOverlay: ScanOverlay = root.findViewById(R.id.scanOverlay)
  val miniMap: MiniMapView = root.findViewById(R.id.miniMap)
  private val roomChip: TextView = root.findViewById(R.id.roomChip)
  private val ghostImage: ImageView = root.findViewById(R.id.ghostImage)
  private val lockBar: View = root.findViewById(R.id.lockBar)
  private val lockText: TextView = root.findViewById(R.id.lockText)
  private val lockButton: Button = root.findViewById(R.id.lockButton)
  private val lockCancel: Button = root.findViewById(R.id.lockCancel)
  val dialogs = LocationDialogs(activity, renderer, this)

  /** True while the spot lock bar (and maybe the ghost) is shown; camera taps are ignored then. */
  var lockModeActive = false
    private set
  private var lockInstructions = ""

  private val snackbar =
    SnackbarHelper().apply {
      setParentView(root.findViewById(R.id.coordinatorLayout))
      setMaxLines(4)
    }

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
    // A single tap or a long-press on the status panel opens the location sheet.
    val statusPanel = root.findViewById<View>(R.id.statusPanel)
    statusPanel.setOnClickListener { dialogs.showSheet(renderer.uiState) }
    statusPanel.setOnLongClickListener {
      dialogs.showSheet(renderer.uiState)
      true
    }
    roomChip.setOnClickListener {
      val st = renderer.uiState
      val map = st?.map ?: return@setOnClickListener toast("Set home first: rooms belong to the home map.")
      dialogs.pickRoom(map.rooms, st.currentRoomId) { id, newRoom -> renderer.requestPickRoom(id, newRoom, "chip") }
    }
  }

  /** Location sheet → Picture width: set the measured width of the reference picture. */
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
    AlertDialog.Builder(activity)
      .setTitle("Room reference size")
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

  override fun onResume(owner: LifecycleOwner) = surfaceView.onResume()

  override fun onPause(owner: LifecycleOwner) = surfaceView.onPause()

  fun post(action: () -> Unit) {
    root.post { action() }
  }

  fun toast(message: String) = snackbar.showMessageForShortDuration(activity, message)

  // ------------------------------------------------------------------ status panel

  fun renderStatus(
    tracking: Boolean,
    trackingDetail: String,
    loc: LocState,
    locLine: String,
    items: Int,
    hint: String,
    roomChipText: String?,
    /** v0.4: capture mode and study tag, or null when neither is set (FREE, no tag). */
    studyLine: String? = null,
  ) {
    val s = SpannableStringBuilder()
    fun dot(color: Int) {
      val start = s.length
      s.append("● ")
      s.setSpan(ForegroundColorSpan(color), start, s.length, Spanned.SPAN_EXCLUSIVE_EXCLUSIVE)
    }
    dot(if (tracking) GREEN else RED)
    s.append(if (tracking) "Tracking" else "Tracking: $trackingDetail")
    s.append("    ")
    dot(
      when (loc) {
        LocState.LOCATED -> GREEN
        LocState.NOT_LOCATED -> AMBER
        LocState.NO_MAP -> GREY
      }
    )
    s.append(locLine)
    s.append("    $items items")
    if (studyLine != null) s.append("\n").append(studyLine)
    statusText.text = s
    hintText.text = hint
    if (roomChipText == null) roomChip.visibility = View.GONE
    else {
      roomChip.visibility = View.VISIBLE
      roomChip.text = "Room: $roomChipText ▾"
    }
  }

  // ------------------------------------------------------------------ spot lock / ghost

  /**
   * Show the lock bar (and the ghost image when given), hide the guide card meanwhile. Lock and
   * Cancel go to the renderer; the localizer ends the mode through [exitLockMode].
   */
  fun enterLockMode(text: String, ghost: Bitmap?, onLock: () -> Unit, onCancel: () -> Unit) {
    lockModeActive = true
    lockInstructions = text
    lockText.text = text
    guideCard.visibility = View.GONE
    lockBar.visibility = View.VISIBLE
    lockButton.setOnClickListener {
      lockText.text = "Hold still…"
      onLock()
    }
    lockCancel.setOnClickListener {
      onCancel()
      exitLockMode()
    }
    if (ghost != null) {
      ghostImage.setImageBitmap(ghost)
      placeGhost(ghost)
      ghostImage.visibility = View.VISIBLE
    } else {
      // Entering "Add a spot here" from an open "Use a spot" must not keep the old ghost.
      ghostImage.visibility = View.GONE
      ghostImage.setImageBitmap(null)
    }
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
    lockBar.visibility = View.GONE
    ghostImage.visibility = View.GONE
    ghostImage.setImageBitmap(null)
    // A search, or "Update location", is still running.
    if (renderer.query != null || renderer.updateTarget != null) guideCard.visibility = View.VISIBLE
  }

  // ------------------------------------------------------------------ guide card

  private var outcomeShown = false
  private var guideLastSeenMs = 0L

  fun enterGuideMode(rec: ObjectRecord, thumb: Bitmap?, decision: PolicyDecision, hasLocation: Boolean, roomName: String?) {
    guideThumb.setImageBitmap(thumb)
    guideTitle.text = rec.name
    guideLastSeenMs = rec.lastSeenMs
    guideSubtitle.text = "Last seen ${ago(rec.lastSeenMs)} · historical location"
    guideCue.text = GuidancePlanner.roomPhrase(roomName)
    guidePolicy.text =
      when {
        !hasLocation -> "Policy: no location at query start, so no verification decision"
        decision.verify -> "Verification recommended: " + decision.reasons.joinToString("; ")
        else -> "Policy: no verification needed (" + decision.reasons.joinToString("; ") + ")"
      }
    guidePolicy.setTextColor(if (decision.verify) AMBER else GREY)
    setVerifyLine(null, null)
    guideCard.visibility = if (lockModeActive) View.GONE else View.VISIBLE
    verifyButton.isEnabled = true
    verifyButton.text = if (decision.verify) "Verify ★" else "Verify"
    doneButton.isEnabled = true
    doneButton.text = "Done"
    itemsButton.isEnabled = false
  }

  fun setGuideCue(cue: String, hasLocation: Boolean, near: Boolean = true) {
    guideCue.text = cue
    guideSubtitle.text = "Last seen ${ago(guideLastSeenMs)} · historical location"
    if (!outcomeShown && doneButton.text != "Skip") {
      guideVerify.text =
        when {
          !hasLocation -> "Verification needs a location: go to a fix point (picture or spot)"
          near -> "Not verified yet: aim at the ring and tap Verify"
          else -> "Not verified yet: walk closer; Verify once the ring is on the item"
        }
      guideVerify.setTextColor(if (hasLocation) Color.WHITE else GREY)
    }
  }

  fun setVerifyLine(outcome: String?, similarity: Float?) {
    outcomeShown = outcome != null
    when (outcome) {
      null -> {
        guideVerify.text = "Not verified yet: aim at the ring and tap Verify"
        guideVerify.setTextColor(Color.WHITE)
      }
      "CONFIRMED" -> {
        guideVerify.text = "✓ Seen here now (similarity %.2f)".format(similarity ?: 0f)
        guideVerify.setTextColor(GREEN)
      }
      else -> {
        guideVerify.text =
          "? Not confirmed (%.2f): insufficient evidence, may still be here".format(similarity ?: 0f)
        guideVerify.setTextColor(AMBER)
      }
    }
  }

  /** After ground truth "present" (marker error) or "displaced" (how far the item moved). */
  fun enterMeasureMode(displaced: Boolean = false) {
    guideVerify.text =
      if (displaced) "Tap the item's true centre where it is now to measure the displacement, or tap Skip"
      else "Tap the object's true centre to measure marker error, or tap Skip"
    guideVerify.setTextColor(Color.WHITE)
    verifyButton.isEnabled = false
    doneButton.text = "Skip"
  }

  fun exitGuideMode() {
    guideCard.visibility = View.GONE
    guideThumb.setImageBitmap(null)
    overlay.setTarget(null)
    verifyButton.isEnabled = false
    verifyButton.text = "Verify"
    doneButton.isEnabled = false
    doneButton.text = "Done"
    itemsButton.isEnabled = true
  }

  /**
   * v0.4 "Update location": the guide card shows the item and asks for a tap on it where it is now.
   * Done becomes Cancel; Items and Verify are off, as during a search.
   */
  fun enterUpdateMode(rec: ObjectRecord, thumb: Bitmap?) {
    guideThumb.setImageBitmap(thumb)
    guideTitle.text = "Update location of " + rec.name
    guideLastSeenMs = rec.lastSeenMs
    guideSubtitle.text = "Last seen ${ago(rec.lastSeenMs)} at the old location"
    guideCue.text = "Tap it where it is now"
    guidePolicy.text = "Saving replaces the old location. The old photo and logs are kept."
    guidePolicy.setTextColor(GREY)
    outcomeShown = false
    guideVerify.text = "Cancel keeps the old location."
    guideVerify.setTextColor(Color.WHITE)
    guideCard.visibility = if (lockModeActive) View.GONE else View.VISIBLE
    verifyButton.isEnabled = false
    verifyButton.text = "Verify"
    doneButton.isEnabled = true
    doneButton.text = "Cancel"
    itemsButton.isEnabled = false
  }

  fun exitUpdateMode() = exitGuideMode()

  // ------------------------------------------------------------------ dialogs

  /**
   * [onSave] gets the room chosen in the dialog, whether the user picked it explicitly (an explicit
   * pick also sets the current room and relabels nearby trail), a room created here, if any, and
   * the mobility class (v0.4: "rare", "some" or "often").
   *
   * With [update] (v0.4 "Update location") the dialog is titled "Update location of X" and starts
   * with that record's name and mobility; `typed` then means the name was changed. The room starts
   * as the room of the capture, or the record's old room when the capture has none.
   */
  fun showRegisterDialog(
    d: RegistrationDraft,
    defaultName: String,
    rooms: List<Room>,
    update: ObjectRecord? = null,
    onSave: (name: String, typed: Boolean, roomId: String?, picked: Boolean, newRoom: Room?, mobility: String) -> Unit,
    onCancel: () -> Unit
  ) {
    val pad = dp(16)
    val layout =
      LinearLayout(activity).apply {
        orientation = LinearLayout.VERTICAL
        setPadding(pad, pad / 2, pad, 0)
      }
    layout.addView(
      ImageView(activity).apply {
        setImageBitmap(d.crop)
        adjustViewBounds = true
        maxHeight = dp(220)
      }
    )
    val detectorText =
      if (d.rawLabel.isEmpty()) "no category" else "%s (%.2f)".format(d.rawLabel, d.rawConfidence)
    layout.addView(
      TextView(activity).apply {
        textSize = 12f
        text =
          "Detector: $detectorText · crop: ${d.boxSource}\n" +
            "%.2f m away · %s · map: %s".format(
              d.captureDistance,
              d.hitType,
              if (d.mapPosition != null) "yes" else "NO (this session only)"
            )
      }
    )
    if (d.captureDistance > 1.5f) {
      layout.addView(
        TextView(activity).apply {
          textSize = 13f
          setTextColor(AMBER)
          text = "%.1f m away: for a reliable photo, stand 0.5–1 m from the object.".format(d.captureDistance)
        }
      )
    }
    if (d.mapPosition == null) {
      layout.addView(
        TextView(activity).apply {
          textSize = 13f
          setTextColor(RED)
          text =
            "Not located: this item is valid for this session only. Look at the picture or use a spot " +
              "first to save it on the home map."
        }
      )
    }
    // Room row (only once a home map exists).
    var roomId = d.defaultRoomId ?: update?.roomId?.takeIf { old -> rooms.any { it.id == old } }
    var picked = false
    var newRoom: Room? = null
    val allRooms = rooms.toMutableList()
    val roomRow =
      TextView(activity).apply {
        textSize = 15f
        setPadding(0, dp(10), 0, dp(4))
        setTextColor(Color.rgb(0x1E, 0x88, 0xE5))
      }
    fun showRoom() {
      roomRow.text = "Room: ${allRooms.firstOrNull { it.id == roomId }?.name ?: "none"} (tap to change)"
    }
    if (d.hasMap) {
      showRoom()
      roomRow.setOnClickListener {
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
    // v0.4 (S6): mobility class, a pre-capture signal for the policies compared offline.
    layout.addView(
      TextView(activity).apply {
        textSize = 13f
        setPadding(0, dp(8), 0, 0)
        text = "How often does it move?"
      }
    )
    val mobilityGroup = RadioGroup(activity).apply { orientation = RadioGroup.HORIZONTAL }
    val mobilityOf = HashMap<Int, String>()
    val initialMobility = Mobility.orDefault(update?.mobility)
    for ((value, label) in Mobility.CHOICES) {
      val button =
        RadioButton(activity).apply {
          id = View.generateViewId()
          text = label
        }
      mobilityOf[button.id] = value
      mobilityGroup.addView(button)
      if (value == initialMobility) mobilityGroup.check(button.id)
    }
    layout.addView(mobilityGroup)
    val edit =
      EditText(activity).apply {
        hint = "Name (default: $defaultName)"
        setSingleLine()
        if (update != null) setText(update.name)
      }
    layout.addView(edit)

    AlertDialog.Builder(activity)
      .setTitle(if (update != null) "Update location of " + update.name else "Remember this object?")
      .setView(ScrollView(activity).apply { addView(layout) }) // the name field stays reachable above the keyboard
      .setPositiveButton("Save") { _, _ ->
        val typed = edit.text.toString().trim()
        onSave(
          typed.ifEmpty { defaultName },
          typed.isNotEmpty() && typed != update?.name,
          roomId,
          picked,
          newRoom?.takeIf { it.id == roomId },
          mobilityOf[mobilityGroup.checkedRadioButtonId] ?: Mobility.DEFAULT,
        )
      }
      .setNegativeButton("Cancel") { _, _ -> onCancel() }
      .setOnCancelListener { onCancel() }
      .show()
  }

  /**
   * The Items list (v0.5): tapped records and scanned entries in one list, newest first, with a
   * search over names and labels. A tapped record opens the same dialog as before (Find, Update
   * location, Rename, Delete); a scanned entry opens [showEntry]. Rows are marked with their kind
   * only when the inventory has entries, so without a scan the rows read as before.
   */
  fun showItems(
    records: List<ObjectRecord>,
    filesRoot: File,
    state: Localizer.State?,
    onFind: (ObjectRecord) -> Unit,
    onUpdate: (ObjectRecord) -> Unit,
    onRename: (ObjectRecord, String) -> Unit,
    onDelete: (ObjectRecord) -> Unit,
    entries: List<InventoryEntry> = emptyList(),
    lastScanId: String? = null,
    entryActions: EntryActions? = null,
  ) {
    if (records.isEmpty() && entries.isEmpty()) {
      toast("No items yet. Tap an object in the camera view to remember it.")
      return
    }
    val all = (records.map { ItemRow(it, null) } + entries.map { ItemRow(null, it) }).sortedByDescending { it.lastSeenMs }
    val shown = ArrayList(all)
    val markKinds = entries.isNotEmpty()
    val thumbs = HashMap<String, Bitmap?>()
    fun thumb(path: String): Bitmap? = thumbs.getOrPut(path) { loadThumb(File(filesRoot, path)) }
    val adapter =
      object : ArrayAdapter<ItemRow>(activity, 0, shown) {
        override fun getView(position: Int, convertView: View?, parent: ViewGroup): View {
          val item = getItem(position)!!
          val row =
            LinearLayout(context).apply {
              orientation = LinearLayout.HORIZONTAL
              gravity = Gravity.CENTER_VERTICAL
              setPadding(dp(16), dp(8), dp(16), dp(8))
            }
          row.addView(
            ImageView(context).apply {
              layoutParams = LinearLayout.LayoutParams(dp(56), dp(56))
              scaleType = ImageView.ScaleType.CENTER_CROP
              setImageBitmap(thumb(item.record?.cropFile ?: item.entry!!.cropFile))
            }
          )
          row.addView(
            TextView(context).apply {
              layoutParams =
                LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f).apply {
                  marginStart = dp(12)
                }
              val r = item.record
              val e = item.entry
              val sb = SpannableStringBuilder(r?.name ?: e!!.name)
              sb.setSpan(StyleSpan(Typeface.BOLD), 0, sb.length, Spanned.SPAN_EXCLUSIVE_EXCLUSIVE)
              if (markKinds) {
                val start = sb.length
                sb.append(if (r != null) "  tapped" else "  scanned")
                sb.setSpan(ForegroundColorSpan(GREY), start, sb.length, Spanned.SPAN_EXCLUSIVE_EXCLUSIVE)
                sb.setSpan(RelativeSizeSpan(0.8f), start, sb.length, Spanned.SPAN_EXCLUSIVE_EXCLUSIVE)
              }
              if (r != null) {
                val mapId = state?.map?.mapId
                val where =
                  when {
                    r.mapId != null && r.mapId == mapId && r.mapPosition != null ->
                      " · " + (state?.roomName(r.roomId) ?: "room not named")
                    r.sessionId == activity.sessionId -> " · this session only"
                    else -> " · photo only (saved before this map)"
                  }
                sb.append("\nlast seen ${ago(r.lastSeenMs)}$where")
              } else {
                sb.append("\n").append(entryLine(e!!, state, lastScanId))
              }
              text = sb
              textSize = 15f
            }
          )
          return row
        }
      }
    val search =
      EditText(activity).apply {
        setSingleLine()
        inputType = InputType.TYPE_CLASS_TEXT or InputType.TYPE_TEXT_FLAG_NO_SUGGESTIONS
        hint = "Search names and labels"
        addTextChangedListener(
          object : TextWatcher {
            override fun beforeTextChanged(s: CharSequence?, start: Int, count: Int, after: Int) {}

            override fun onTextChanged(s: CharSequence?, start: Int, before: Int, count: Int) {}

            override fun afterTextChanged(s: Editable?) {
              val q = s?.toString() ?: ""
              val filtered =
                all.filter { row ->
                  row.record?.let { InventoryText.matches(it, q) } ?: InventoryText.matches(row.entry!!, q)
                }
              adapter.setNotifyOnChange(false)
              adapter.clear()
              adapter.addAll(filtered)
              adapter.notifyDataSetChanged()
            }
          }
        )
      }
    val titleBox =
      LinearLayout(activity).apply {
        orientation = LinearLayout.VERTICAL
        setPadding(dp(24), dp(20), dp(24), 0)
        addView(
          TextView(activity).apply {
            text = "Remembered items"
            textSize = 20f
            setTypeface(typeface, Typeface.BOLD)
          }
        )
        addView(search)
      }
    AlertDialog.Builder(activity)
      .setCustomTitle(titleBox)
      .setAdapter(adapter) { _, i ->
        val item = shown.getOrNull(i) ?: return@setAdapter
        val e = item.entry
        if (e != null) {
          if (entryActions != null) showEntry(e, filesRoot, state, lastScanId, entryActions)
          return@setAdapter
        }
        val r = item.record!!
        // v0.4 (S7): a fourth action next to Find, Rename and Delete, as a button in the body.
        var itemDialog: AlertDialog? = null
        val updateRow =
          LinearLayout(activity).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(20), 0, dp(20), 0)
            addView(
              Button(activity).apply {
                text = "Update location"
                setOnClickListener {
                  itemDialog?.dismiss()
                  onUpdate(r)
                }
              }
            )
            addView(
              TextView(activity).apply {
                textSize = 12f
                text = "Found it somewhere else? Tap it there to replace the remembered location."
              }
            )
          }
        itemDialog =
          AlertDialog.Builder(activity)
            .setTitle(r.name)
            .setMessage("Last seen ${ago(r.lastSeenMs)}. The location is historical until verified.")
            .setView(updateRow)
            .setPositiveButton("Find") { _, _ -> onFind(r) }
            .setNeutralButton("Delete") { _, _ ->
              AlertDialog.Builder(activity)
                .setTitle("Delete \"${r.name}\"?")
                .setMessage("This removes the item from the list. Its photo and study logs are kept for analysis.")
                .setPositiveButton("Delete") { _, _ -> onDelete(r) }
                .setNegativeButton("Cancel", null)
                .show()
            }
            .setNegativeButton("Rename") { _, _ -> showRename(r, onRename) }
            .show()
      }
      .setNegativeButton("Close", null)
      .show()
      // The search field is in the custom title, which the dialog does not count as text input:
      // without this the soft keyboard would never open for it.
      .window
      ?.clearFlags(WindowManager.LayoutParams.FLAG_NOT_FOCUSABLE or WindowManager.LayoutParams.FLAG_ALT_FOCUSABLE_IM)
  }

  /**
   * Second line of a scanned entry: location and evidence, "Bedroom · seen 2 d ago", "Kitchen ·
   * looked for in the last scan, not seen" or "Office · not checked since Nov 17".
   */
  private fun entryLine(e: InventoryEntry, state: Localizer.State?, lastScanId: String?): String {
    val inMap = state?.map?.mapId != null && e.mapId == state.map.mapId
    return InventoryText.location(e, state?.roomName(e.roomId), inMap) + " · " +
      InventoryText.evidence(e, System.currentTimeMillis(), lastScanId)
  }

  /** A scanned entry: Find, Rename or Delete. Back or a tap outside changes nothing. */
  private fun showEntry(e: InventoryEntry, filesRoot: File, state: Localizer.State?, lastScanId: String?, actions: EntryActions) {
    val box =
      LinearLayout(activity).apply {
        orientation = LinearLayout.VERTICAL
        setPadding(dp(20), dp(8), dp(20), 0)
        BitmapFactory.decodeFile(File(filesRoot, e.cropFile).path)?.let { bmp ->
          addView(
            ImageView(activity).apply {
              setImageBitmap(bmp)
              adjustViewBounds = true
              maxHeight = dp(180)
            }
          )
        }
        addView(
          TextView(activity).apply {
            textSize = 14f
            setPadding(0, dp(8), 0, 0)
            text =
              entryLine(e, state, lastScanId) + "\n" +
                "Detected as " + e.label + " · " + e.sightingCount + " sightings · first seen " + InventoryText.day(e.firstSeenMs) +
                (if (e.nameEdited) "" else "\nThe name is the detector's label until you rename it.") +
                "\nThe location is historical until verified."
          }
        )
      }
    AlertDialog.Builder(activity)
      .setTitle(e.name)
      .setView(box)
      .setPositiveButton("Find") { _, _ -> actions.onFind(e) }
      .setNeutralButton("Delete") { _, _ ->
        AlertDialog.Builder(activity)
          .setTitle("Delete \"" + e.name + "\"?")
          .setMessage("This removes the scanned item from the list. Its photo and logs are kept for analysis.")
          .setPositiveButton("Delete") { _, _ -> actions.onDelete(e) }
          .setNegativeButton("Cancel", null)
          .show()
      }
      .setNegativeButton("Rename") { _, _ -> showRenameEntry(e, actions) }
      .show()
  }

  private fun showRenameEntry(e: InventoryEntry, actions: EntryActions) {
    val edit =
      EditText(activity).apply {
        setText(e.name)
        setSingleLine()
        selectAll()
      }
    val box = LinearLayout(activity).apply { setPadding(dp(20), dp(8), dp(20), 0); addView(edit) }
    AlertDialog.Builder(activity)
      .setTitle("Rename")
      .setView(box)
      .setPositiveButton("Save") { _, _ ->
        val n = edit.text.toString().trim()
        if (n.isNotEmpty() && n != e.name) actions.onRename(e, n)
      }
      .setNegativeButton("Cancel", null)
      .show()
  }

  /** v0.5: Scan when idle and located, Stop during a scan, else hidden. */
  fun setScanControl(c: ScanControl) {
    when (c) {
      ScanControl.HIDDEN -> scanButton.visibility = View.GONE
      ScanControl.START -> {
        scanButton.text = "Scan"
        scanButton.visibility = View.VISIBLE
      }
      ScanControl.STOP -> {
        scanButton.text = "Stop"
        scanButton.visibility = View.VISIBLE
      }
    }
  }

  /** v0.5: the counts of a finished scan. [onItems] opens the Items list. */
  fun showScanSummary(end: ScanEnd, onItems: () -> Unit) {
    val s = end.summary
    val text = StringBuilder()
    if (s == null) {
      text.append("The scan could not be summarised").append(end.error?.let { ": $it" } ?: "").append(".\n")
    } else {
      text.append("New items: ").append(s.newCount).append('\n')
      text.append("Seen again: ").append(s.seenAgain).append('\n')
      text.append("Looked for, not seen: ").append(s.missed).append('\n')
      text.append("Not observed: ").append(s.notObserved).append('\n')
      text.append("Seen in one keyframe only (dropped): ").append(s.tentativeDropped).append('\n')
    }
    text.append('\n')
    text.append(end.keyframes).append(" keyframes · ").append("%.0f m walked · %.0f s".format(end.metersWalked, end.durationMs / 1000.0))
    end.depthCheckMedianM?.let { text.append("\nDepth check: median %.1f cm over %d keyframes".format(it * 100, end.depthChecks)) }
    if (end.reason == "paused") text.append("\n\nThe scan ended because the app left the screen.")
    AlertDialog.Builder(activity)
      .setTitle("Scan finished")
      .setMessage(text.toString())
      .setPositiveButton("OK", null)
      .setNeutralButton("Items") { _, _ -> onItems() }
      .show()
  }

  private fun showRename(r: ObjectRecord, onRename: (ObjectRecord, String) -> Unit) {
    val edit =
      EditText(activity).apply {
        setText(r.name)
        setSingleLine()
        selectAll()
      }
    val box = LinearLayout(activity).apply { setPadding(dp(20), dp(8), dp(20), 0); addView(edit) }
    AlertDialog.Builder(activity)
      .setTitle("Rename")
      .setView(box)
      .setPositiveButton("Save") { _, _ ->
        val n = edit.text.toString().trim()
        if (n.isNotEmpty() && n != r.name) onRename(r, n)
      }
      .setNegativeButton("Cancel", null)
      .show()
  }

  private fun loadThumb(f: File): Bitmap? =
    BitmapFactory.decodeFile(f.path, BitmapFactory.Options().apply { inSampleSize = 4 })

  /** [onDismiss] runs once the result is closed, by OK, Back or a tap outside (v0.4 BENCHMARK). */
  fun showVerifyResult(
    stored: Bitmap?,
    fresh: Bitmap,
    similarity: Float,
    confirmed: Boolean,
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
    val message =
      if (confirmed) "Seen here now. Similarity %.2f ≥ %.2f.".format(similarity, VerificationPolicy.MATCH_THRESHOLD)
      else
        ("Not confirmed: insufficient evidence (similarity %.2f < %.2f). " +
            "The item may still be here; showing the last-seen location.")
          .format(similarity, VerificationPolicy.MATCH_THRESHOLD)
    AlertDialog.Builder(activity)
      .setTitle(if (confirmed) "Verified" else "Not verified")
      .setView(row)
      .setMessage(message)
      .setPositiveButton("OK", null)
      .setOnDismissListener { onDismiss?.invoke() }
      .show()
  }

  /**
   * v0.4 (S4): the five ground-truth values of the study spec, section 2. Not cancelable, so the
   * query always ends with exactly one answer ("Skip" is the way out).
   */
  fun askGroundTruth(name: String, onAnswer: (String) -> Unit) {
    AlertDialog.Builder(activity)
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
    AlertDialog.Builder(activity)
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
