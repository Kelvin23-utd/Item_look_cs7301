package edu.cs6301.findback

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.text.InputFilter
import android.text.InputType
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
import java.io.File

/**
 * Location dialogs (UI thread): the location sheet opened from the status panel, the room picker,
 * and the prompts the localizer raises. Every prompt delivers exactly one answer, also when the
 * dialog is dismissed with Back (as "no"), so the localizer never waits forever.
 */
class LocationDialogs(
  private val activity: MainActivity,
  private val renderer: FindBackRenderer,
  private val main: MainView,
) {
  private fun dp(v: Int) = (v * activity.resources.displayMetrics.density).toInt()

  // ------------------------------------------------------------------ location sheet

  fun showSheet(st: Localizer.State?) {
    val map = st?.map
    val status = StringBuilder()
    when {
      map == null -> status.append("No home map yet.\nShow the picture to the camera, or add a spot, to set home.")
      else -> {
        status.append(
          "Map ${map.mapId} · ${map.fixPoints.size} fix point(s) · ${map.rooms.size} room(s) · ${map.nodes.size} trail points\n"
        )
        status.append(
          if (st?.located == true) "Located via %s · %.0f m walked since the fix".format(st.fixName, st.metersSinceFix ?: 0.0)
          else "Not located: look at the picture, or use a spot."
        )
      }
    }
    st?.imageFixNote?.let { status.append("\n\n").append(it) }
    StudySettings.statusLine(activity.studyTag, activity.captureMode)?.let { status.append("\n\n").append(it) }

    val spots = map?.fixPoints?.filter { it.type == FixType.SPOT } ?: emptyList()
    val entries = ArrayList<Pair<String, () -> Unit>>()
    entries += "Picture width (%.1f cm)".format(activity.refWidthM * 100) to { main.showReferenceSettings() }
    entries +=
      if (map == null) "Set home here with a spot" to { startAddSpot(home = true) }
      else "Add a spot here" to {
        if (st?.located == true) startAddSpot(home = false)
        else main.toast("Locate first (look at the picture or use a spot), then add a spot.")
      }
    entries += "Use a spot" to {
      if (spots.isEmpty()) main.toast("No spots saved yet. Use \"Add a spot here\" first.") else chooseSpot(spots)
    }
    if (map != null && map.rooms.isNotEmpty()) entries += "Rename a room" to { chooseRoomToRename(map.rooms) }
    entries += "Study settings…" to { showStudySettings() }
    if (map != null) entries += "New map…" to { confirmNewMap(map.mapId, activity.store.snapshot().size) }

    val header =
      TextView(activity).apply {
        text = status
        textSize = 14f
        setPadding(dp(24), dp(20), dp(24), dp(8))
      }
    val box =
      LinearLayout(activity).apply {
        orientation = LinearLayout.VERTICAL
        addView(TextView(activity).apply {
          text = "Home map"
          textSize = 20f
          setTypeface(typeface, android.graphics.Typeface.BOLD)
          setPadding(dp(24), dp(20), dp(24), 0)
        })
        addView(header)
      }
    AlertDialog.Builder(activity)
      .setCustomTitle(box)
      .setItems(entries.map { it.first }.toTypedArray()) { _, i -> entries[i].second() }
      .setNegativeButton("Close", null)
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
    // v0.5 (plan section 8): read when a scan starts; logged as `scan_start.saveFrames`.
    val saveFrames =
      CheckBox(activity).apply {
        text = "Save scan frames (image, depth and detections of every scan keyframe)"
        isChecked = activity.scanSaveFrames
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
            textSize = 13f
            setPadding(0, dp(12), 0, 0)
            text = "Scan mode"
          }
        )
        addView(saveFrames)
      }
    AlertDialog.Builder(activity)
      .setTitle("Study settings")
      .setView(box)
      .setPositiveButton("Save") { _, _ ->
        if (saveFrames.isChecked != activity.scanSaveFrames) activity.setScanSaveFrames(saveFrames.isChecked)
        if (renderer.query != null) {
          main.toast("A search started meanwhile: study settings not changed.")
        } else {
          val mode = modeOf[modeGroup.checkedRadioButtonId] ?: activity.captureMode
          activity.setStudySettings(tagEdit.text.toString(), mode)
          main.toast(StudySettings.statusLine(activity.studyTag, activity.captureMode) ?: "Study: FREE, no tag")
        }
      }
      .setNegativeButton("Cancel", null)
      .show()
  }

  private fun startAddSpot(home: Boolean) {
    val text =
      (if (home) "This spot becomes home. " else "") +
        "Hold the phone in a pose you can repeat later, for example pressed against a door frame or the edge " +
        "of a shelf. Upright or on its side both work. Then press Lock and hold still for a moment."
    main.enterLockMode(text, null, onLock = { renderer.requestAddSpot() }, onCancel = { renderer.requestCancelLock() })
  }

  private fun chooseSpot(spots: List<FixPoint>) {
    if (spots.size == 1) return useSpot(spots[0])
    AlertDialog.Builder(activity)
      .setTitle("Use which spot?")
      .setItems(spots.map { it.name }.toTypedArray()) { _, i -> useSpot(spots[i]) }
      .setNegativeButton("Cancel", null)
      .show()
  }

  private fun useSpot(spot: FixPoint) {
    val ghost = spot.ghostFile?.let { BitmapFactory.decodeFile(File(activity.store.root, it).path) }
    val text =
      if (ghost != null) "Move the phone until the live view lines up with the faint picture of \"${spot.name}\", " +
        "hold it still, then press Lock."
      else "No ghost image for \"${spot.name}\". Hold the phone in the saved pose and still, then press Lock."
    main.enterLockMode(text, ghost, onLock = { renderer.requestUseSpot(spot.id) }, onCancel = { renderer.requestCancelLock() })
  }

  private fun chooseRoomToRename(rooms: List<Room>) {
    AlertDialog.Builder(activity)
      .setTitle("Rename which room?")
      .setItems(rooms.map { it.name }.toTypedArray()) { _, i ->
        val r = rooms[i]
        askName("Rename room", r.name) { n -> if (n != r.name) renderer.requestRenameRoom(r.id, n) }
      }
      .setNegativeButton("Cancel", null)
      .show()
  }

  private fun confirmNewMap(mapId: String, records: Int) {
    AlertDialog.Builder(activity)
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
    AlertDialog.Builder(activity)
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
    val edit =
      EditText(activity).apply {
        setText(initial)
        setSingleLine()
        hint = "Name"
        selectAll()
      }
    val box = LinearLayout(activity).apply { setPadding(dp(20), dp(8), dp(20), 0); addView(edit) }
    AlertDialog.Builder(activity)
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

  private fun yesNo(title: String, message: String, yes: String, no: String, onAnswer: (Boolean) -> Unit) {
    var answered = false
    fun answer(v: Boolean) {
      if (answered) return
      answered = true
      onAnswer(v)
    }
    AlertDialog.Builder(activity)
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
      "Set home here?",
      "The picture becomes the home fix point of a new map. Items you save are stored relative to it, " +
        "so keep the picture in this place.",
      "Set home",
      "Not now",
      onAnswer,
    )

  fun askUseNewFix(jumpM: Double, jumpDeg: Double, fixName: String, onAnswer: (Boolean) -> Unit) =
    yesNo(
      "Position jumped %.1f m. Use the new fix?".format(jumpM),
      "\"%s\" disagrees with the current map position by %.1f m and %.0f°. ".format(fixName, jumpM, jumpDeg) +
        "If the picture was moved, keep the current fix.",
      "Use new fix",
      "Keep current",
      onAnswer,
    )

  fun askAddPicture(onAnswer: (Boolean) -> Unit) =
    yesNo(
      "Save the picture as a fix point?",
      "You are located. Saving the picture here lets later sessions locate from it too. Keep it in this place.",
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
                "Later, \"Use a spot\" shows this view faintly over the camera so you can line it up again."
          }
        )
        addView(edit)
      }
    AlertDialog.Builder(activity)
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
