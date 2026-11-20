package edu.cs6301.findback

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Color
import android.text.InputFilter
import android.text.InputType
import android.view.Gravity
import android.view.View
import android.widget.CheckBox
import android.widget.EditText
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.RadioButton
import android.widget.RadioGroup
import android.widget.TextView
import androidx.appcompat.app.AlertDialog
import edu.cs6301.findback.map.FixPoint
import edu.cs6301.findback.map.FixType
import edu.cs6301.findback.map.Localizer
import edu.cs6301.findback.map.Room
import edu.cs6301.findback.study.CaptureMode
import edu.cs6301.findback.study.StudySettings
import edu.cs6301.findback.trace.TraceSession
import java.io.File
import java.util.Locale

/**
 * Location dialogs (UI thread): the Home sheet opened from the menu button (v0.6), Locate, the room
 * picker, and the prompts the localizer raises. Every prompt delivers exactly one answer, also when
 * the dialog is dismissed with Back (as "no"), so the localizer never waits forever.
 */
class LocationDialogs(
  private val activity: MainActivity,
  private val renderer: FindBackRenderer,
  private val main: MainView,
) {
  private fun dp(v: Int) = (v * activity.resources.displayMetrics.density).toInt()

  /** v0.7: a dialog whose answers are logged as `ui_action` (a plain builder when the trace is off). */
  private fun traced(name: String): AlertDialog.Builder = main.uiTrace.dialog(activity, name)

  // ------------------------------------------------------------------ Home sheet

  /** Section 6.4: status, details and the entries that work now. */
  fun showSheet(st: Localizer.State?) {
    val map = st?.map
    val spots = spotsOf(st)
    val located = st?.located == true
    val fixName = st?.fixName ?: ""
    val status =
      when {
        map == null -> "No home yet"
        located && st?.fixType == FixType.SPOT -> "Located at the spot \"$fixName\""
        located -> "Located at \"$fixName\""
        else -> "Not located"
      }
    val details = StringBuilder()
    if (map == null) {
      details.append("Press Set home here, or show the reference picture to the camera.")
    } else {
      val pictures = map.fixPoints.count { it.type == FixType.IMAGE }
      details.append("Map ").append(map.mapId)
      details.append(" · ").append(plural(map.rooms.size, "room", "rooms"))
      details.append(" · ").append(plural(spots.size, "spot", "spots"))
      if (pictures > 0) details.append(" · ").append(plural(pictures, "picture", "pictures"))
      details.append(" · ").append(map.nodes.size).append(" trail points")
      if (located) details.append(" · %.0f m walked since locating".format(Locale.US, st?.metersSinceFix ?: 0.0))
    }
    st?.imageFixNote?.let { details.append("\n\n").append(it) }
    StudySettings.statusLine(activity.studyTag, activity.captureMode)?.let { details.append("\n\n").append(it) }

    val entries = ArrayList<Pair<String, () -> Unit>>()
    if (spots.isNotEmpty()) entries += "Locate" to { locate(st) }
    if (map == null) entries += "Set home here" to { startAddSpot(home = true) }
    else if (located) entries += "Add a spot here" to { startAddSpot(home = false) }
    if (map != null && map.rooms.isNotEmpty()) entries += "Rooms…" to { chooseRoomToRename(map.rooms) }
    entries += "Settings…" to { showSettings() }
    entries += "Study settings…" to { showStudySettings() }
    if (map != null) entries += "Start a new map…" to { confirmNewMap(map.mapId, activity.store.snapshot().size) }

    val box =
      LinearLayout(activity).apply {
        orientation = LinearLayout.VERTICAL
        setPadding(dp(24), dp(20), dp(24), dp(8))
        addView(
          TextView(activity).apply {
            text = "Home"
            textSize = 20f
            setTypeface(typeface, android.graphics.Typeface.BOLD)
          }
        )
        addView(
          TextView(activity).apply {
            text = status
            textSize = 16f
            setPadding(0, dp(8), 0, 0)
          }
        )
        addView(
          TextView(activity).apply {
            text = details
            textSize = 12f
            setTextColor(MainView.GREY)
            setPadding(0, dp(4), 0, 0)
          }
        )
      }
    traced("home_sheet")
      .setCustomTitle(box)
      .setItems(entries.map { it.first }.toTypedArray()) { _, i -> entries[i].second() }
      .setNegativeButton("Close", null)
      .show()
  }

  private fun plural(n: Int, one: String, many: String) = if (n == 1) "1 $one" else "$n $many"

  /** The map's spots, the home spot (the oldest) first. */
  private fun spotsOf(st: Localizer.State?): List<FixPoint> =
    st?.map?.fixPoints?.filter { it.type == FixType.SPOT }?.sortedBy { it.createdMs } ?: emptyList()

  /**
   * Section 6.4, Settings: picture width, and "Save scan frames" (moved from Study settings).
   * v0.7: "Detailed trace" (applies at the next start) and the size of the trace data.
   * v0.8: "Study logging" (applies from the next scan or search).
   * v0.9: "Extra names" (applies when the Items panel next opens).
   */
  private fun showSettings() {
    var dialog: AlertDialog? = null
    val saveFrames =
      CheckBox(activity).apply {
        text = "Save scan frames (image, depth and detections of every scan keyframe)"
        isChecked = activity.scanSaveFrames
      }
    val studyLogging =
      CheckBox(activity).apply {
        text = "Study logging (every scan detection with its pose and embedding, a few close-up views per place, never a view that touches a person)"
        isChecked = activity.studyLogging
      }
    val extraNames =
      CheckBox(activity).apply {
        text = "Extra names (Items search also uses the names and descriptions in names.json, for items whose photo has not changed since)"
        isChecked = activity.extraNamesOn
      }
    val detailedTrace =
      CheckBox(activity).apply {
        text = "Detailed trace (motion, depth, sensors, battery and what was pressed; no images or sound). Applies at the next start."
        isChecked = activity.traceOnNextStart
      }
    val traceSize =
      TextView(activity).apply {
        textSize = 12f
        setTextColor(MainView.GREY)
        setPadding(dp(32), 0, 0, dp(4))
        text = "Trace data on this phone: …"
      }
    // The folder walk runs off the UI thread.
    val filesRoot = activity.filesRootDir
    Thread {
      val mb = TraceSession.storageBytes(filesRoot) / (1024.0 * 1024.0)
      traceSize.post { traceSize.text = "Trace data on this phone: %.1f MB".format(Locale.US, mb) }
    }.start()
    val box =
      LinearLayout(activity).apply {
        orientation = LinearLayout.VERTICAL
        setPadding(dp(20), dp(8), dp(20), 0)
        addView(
          TextView(activity).apply {
            text = "Picture width: %.1f cm (tap to change)".format(Locale.US, activity.refWidthM * 100)
            textSize = 15f
            minHeight = dp(48)
            gravity = Gravity.CENTER_VERTICAL
            setTextColor(Color.rgb(0x1E, 0x88, 0xE5))
            setOnClickListener {
              main.uiTrace.action("dialog:settings", "Picture width")
              dialog?.dismiss()
              main.showReferenceSettings()
            }
          }
        )
        addView(saveFrames)
        addView(studyLogging)
        addView(extraNames)
        addView(detailedTrace)
        addView(traceSize)
      }
    dialog =
      traced("settings")
        .setTitle("Settings")
        .setView(box)
        .setPositiveButton("Save") { _, _ ->
          // v0.5 (plan section 8): read when a scan starts; logged as `scan_start.saveFrames`.
          if (saveFrames.isChecked != activity.scanSaveFrames) activity.setScanSaveFrames(saveFrames.isChecked)
          // v0.8: read when a scan or a search starts; logged as `scan_start.studyLogging`.
          if (studyLogging.isChecked != activity.studyLogging) {
            main.uiTrace.action("dialog:settings", if (studyLogging.isChecked) "study_logging_on" else "study_logging_off")
            activity.setStudyLogging(studyLogging.isChecked)
          }
          // v0.9: read when the Items panel opens; logged as `ui_search.extraNames`.
          if (extraNames.isChecked != activity.extraNamesOn) {
            main.uiTrace.action("dialog:settings", if (extraNames.isChecked) "extra_names_on" else "extra_names_off")
            activity.setExtraNamesOn(extraNames.isChecked)
          }
          // v0.7: read at the next start; logged as `session_start.traceOn`.
          if (detailedTrace.isChecked != activity.traceOnNextStart) {
            main.uiTrace.action("dialog:settings", if (detailedTrace.isChecked) "trace_on" else "trace_off")
            activity.setTraceOnNextStart(detailedTrace.isChecked)
            main.toast(
              if (detailedTrace.isChecked) "Detailed trace turns on at the next start." else "Detailed trace turns off at the next start."
            )
          }
        }
        .setNegativeButton("Cancel", null)
        .show()
  }

  /**
   * v0.4 (S1): study tag and capture mode. Stored in preferences and logged as `study_settings`
   * when either changes. Not offered during a search, so every event of a query carries the same
   * settings. Cancel or Back changes nothing.
   */
  private fun showStudySettings() {
    if (renderer.query != null) {
      main.toast("Finish this search (Done) before changing the study settings.")
      return
    }
    val tagEdit =
      EditText(activity).apply {
        setSingleLine()
        // No suggestions: autocorrect would rewrite key=value pairs.
        inputType = InputType.TYPE_CLASS_TEXT or InputType.TYPE_TEXT_FLAG_NO_SUGGESTIONS
        filters = arrayOf(InputFilter.LengthFilter(StudySettings.TAG_MAX))
        setText(activity.studyTag)
        hint = "home=A;exp=E3;round=2"
      }
    val modeGroup = RadioGroup(activity)
    val modeOf = HashMap<Int, CaptureMode>()
    val labels =
      mapOf(
        CaptureMode.FREE to "FREE: any number of captures (default)",
        CaptureMode.PAIRED to "PAIRED: one capture per query",
        CaptureMode.BENCHMARK to "BENCHMARK: label every capture",
      )
    for (mode in CaptureMode.values()) {
      val button =
        RadioButton(activity).apply {
          id = View.generateViewId()
          text = labels[mode] ?: mode.name
        }
      modeOf[button.id] = mode
      modeGroup.addView(button)
      if (mode == activity.captureMode) modeGroup.check(button.id)
    }
    val box =
      LinearLayout(activity).apply {
        orientation = LinearLayout.VERTICAL
        setPadding(dp(20), dp(8), dp(20), 0)
        addView(
          TextView(activity).apply {
            textSize = 13f
            text =
              "Study tag (up to ${StudySettings.TAG_MAX} characters): key=value pairs separated by ;\n" +
                "for example home=A;exp=E3;round=2;light=evening;split=dev"
          }
        )
        addView(tagEdit)
        addView(
          TextView(activity).apply {
            textSize = 13f
            setPadding(0, dp(12), 0, 0)
            text = "Capture mode"
          }
        )
        addView(modeGroup)
        addView(
          TextView(activity).apply {
            textSize = 12f
            setPadding(0, dp(12), 0, 0)
            text = "With a tag or a mode other than FREE, the app asks the study questions (ground truth, mobility)."
          }
        )
      }
    traced("study_settings")
      .setTitle("Study settings")
      .setView(box)
      .setPositiveButton("Save") { _, _ ->
        if (renderer.query != null) {
          main.toast("A search started meanwhile: study settings not changed.")
        } else {
          val mode = modeOf[modeGroup.checkedRadioButtonId] ?: activity.captureMode
          activity.setStudySettings(tagEdit.text.toString(), mode)
          main.toast(StudySettings.statusLine(activity.studyTag, activity.captureMode) ?: "Study: FREE, no tag")
          main.refresh()
        }
      }
      .setNegativeButton("Cancel", null)
      .show()
  }

  // ------------------------------------------------------------------ Locate and spots

  /**
   * Section 6.1: one spot → its lock bar with the ghost at once; several → "Locate at which spot?",
   * the home spot first; none → the banner says so.
   */
  fun locate(st: Localizer.State?) {
    val spots = spotsOf(st)
    when {
      spots.isEmpty() ->
        main.toast(
          if (st?.map == null) "No home yet: press Set home here first."
          else "No spots saved yet. Once located, add one in the menu (⋮): Add a spot here."
        )
      spots.size == 1 -> useSpot(spots[0])
      else ->
        traced("locate_which")
          .setTitle("Locate at which spot?")
          .setItems(spots.map { it.name }.toTypedArray()) { _, i -> useSpot(spots[i]) }
          .setNegativeButton("Cancel", null)
          .show()
    }
  }

  /** Set home here (no map) or Add a spot here (located): the lock bar without a ghost. */
  fun startAddSpot(home: Boolean) {
    val text =
      (if (home) "This spot becomes home. " else "") +
        "Hold the phone in a pose you can repeat later, for example against a door frame or the edge of a " +
        "shelf (upright or on its side). Then press Lock and hold still for a moment."
    main.enterLockMode(
      if (home) "Set home here" else "Add a spot here",
      text,
      null,
      onLock = { renderer.requestAddSpot() },
      onCancel = { renderer.requestCancelLock() },
    )
  }

  private fun useSpot(spot: FixPoint) {
    val ghost = spot.ghostFile?.let { BitmapFactory.decodeFile(File(activity.store.root, it).path) }
    val text =
      if (ghost != null) "Line up the camera with the faint picture, hold still, then press Lock."
      else "No saved picture for this spot. Hold the phone in the saved pose and still, then press Lock."
    main.enterLockMode(
      "Locate at \"" + spot.name + "\"",
      text,
      ghost,
      onLock = { renderer.requestUseSpot(spot.id) },
      onCancel = { renderer.requestCancelLock() },
    )
  }

  private fun chooseRoomToRename(rooms: List<Room>) {
    traced("rename_which_room")
      .setTitle("Rename which room?")
      .setItems(rooms.map { it.name }.toTypedArray()) { _, i ->
        val r = rooms[i]
        askName("Rename room", r.name) { n -> if (n != r.name) renderer.requestRenameRoom(r.id, n) }
      }
      .setNegativeButton("Cancel", null)
      .show()
  }

  private fun confirmNewMap(mapId: String, records: Int) {
    traced("new_map")
      .setTitle("Start a new map?")
      .setMessage(
        "The current map and its $records item(s) move to an archive folder on the phone; nothing is deleted. " +
          "Photos and logs stay where they are. The app then restarts with an empty map."
      )
      .setPositiveButton("New map") { _, _ -> activity.startNewMap(mapId) }
      .setNegativeButton("Cancel", null)
      .show()
  }

  // ------------------------------------------------------------------ rooms

  /** Pick the current room: existing rooms, "New room…" or "No room". */
  fun pickRoom(rooms: List<Room>, currentId: String?, onPicked: (roomId: String?, newRoom: Room?) -> Unit) {
    val labels = rooms.map { (if (it.id == currentId) "● " else "") + it.name } + listOf("New room…", "No room")
    traced("pick_room")
      .setTitle("Which room are you in?")
      .setItems(labels.toTypedArray()) { _, i ->
        when {
          i < rooms.size -> onPicked(rooms[i].id, null)
          i == rooms.size ->
            askName("New room", "") { n ->
              val r = Room(Localizer.newId(), n, System.currentTimeMillis())
              onPicked(r.id, r)
            }
          else -> onPicked(null, null)
        }
      }
      .setNegativeButton("Cancel", null)
      .show()
  }

  private fun askName(title: String, initial: String, onName: (String) -> Unit) {
    val name = if (title == "New room") "new_room" else "rename_room"
    val edit =
      EditText(activity).apply {
        setText(initial)
        setSingleLine()
        hint = "Name"
        selectAll()
      }
    val box = LinearLayout(activity).apply { setPadding(dp(20), dp(8), dp(20), 0); addView(edit) }
    traced(name)
      .setTitle(title)
      .setView(box)
      .setPositiveButton("Save") { _, _ ->
        val n = edit.text.toString().trim()
        if (n.isEmpty()) main.toast("Enter a name.") else onName(n)
      }
      .setNegativeButton("Cancel", null)
      .show()
  }

  // ------------------------------------------------------------------ localizer prompts

  private fun yesNo(name: String, title: String, message: String, yes: String, no: String, onAnswer: (Boolean) -> Unit) {
    var answered = false
    fun answer(v: Boolean) {
      if (answered) return
      answered = true
      onAnswer(v)
    }
    traced(name)
      .setTitle(title)
      .setMessage(message)
      .setPositiveButton(yes) { _, _ -> answer(true) }
      .setNegativeButton(no) { _, _ -> answer(false) }
      .setOnCancelListener { answer(false) }
      .setOnDismissListener { answer(false) }
      .show()
  }

  fun askSetHome(onAnswer: (Boolean) -> Unit) =
    yesNo(
      "set_home",
      "Set home here?",
      "The picture becomes the home of a new map. Items you save are stored relative to it, " +
        "so keep the picture in this place.",
      "Set home",
      "Not now",
      onAnswer,
    )

  fun askUseNewFix(jumpM: Double, jumpDeg: Double, fixName: String, onAnswer: (Boolean) -> Unit) =
    yesNo(
      "use_new_fix",
      "Position jumped %.1f m. Use the new location?".format(jumpM),
      "\"%s\" disagrees with the current map position by %.1f m and %.0f°. ".format(fixName, jumpM, jumpDeg) +
        "If the picture was moved, keep the current one.",
      "Use new",
      "Keep current",
      onAnswer,
    )

  fun askAddPicture(onAnswer: (Boolean) -> Unit) =
    yesNo(
      "add_picture",
      "Locate with the picture too?",
      "You are located. Saving the picture here lets later sessions locate from it as well. Keep it in this place.",
      "Save",
      "No",
      onAnswer,
    )

  fun askSpotName(defaultName: String, isHome: Boolean, ghost: Bitmap?, onSave: (String) -> Unit, onCancel: () -> Unit) {
    var answered = false
    val edit =
      EditText(activity).apply {
        setSingleLine()
        hint = "Name (default: $defaultName)"
      }
    val box =
      LinearLayout(activity).apply {
        orientation = LinearLayout.VERTICAL
        setPadding(dp(20), dp(8), dp(20), 0)
        if (ghost != null) {
          addView(
            ImageView(activity).apply {
              setImageBitmap(ghost)
              adjustViewBounds = true
              maxHeight = dp(200)
            }
          )
        }
        addView(
          TextView(activity).apply {
            textSize = 13f
            text =
              (if (isHome) "This spot becomes home. " else "") +
                "Later, Locate shows this view faintly over the camera so you can line it up again."
          }
        )
        addView(edit)
      }
    traced("spot_name")
      .setTitle(if (isHome) "Name the home spot" else "Name this spot")
      .setView(box)
      .setPositiveButton("Save") { _, _ ->
        answered = true
        onSave(edit.text.toString().trim().ifEmpty { defaultName })
      }
      .setNegativeButton("Cancel") { _, _ ->
        answered = true
        onCancel()
      }
      .setOnDismissListener { if (!answered) onCancel() }
      .show()
  }
}
