package edu.cs6301.findback

import android.content.res.ColorStateList
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Color
import android.graphics.Typeface
import android.text.Editable
import android.text.SpannableStringBuilder
import android.text.Spanned
import android.text.TextWatcher
import android.text.style.ForegroundColorSpan
import android.text.style.RelativeSizeSpan
import android.text.style.StyleSpan
import android.util.TypedValue
import android.view.Gravity
import android.view.View
import android.view.ViewGroup
import android.view.WindowManager
import android.view.inputmethod.EditorInfo
import android.view.inputmethod.InputMethodManager
import android.widget.ArrayAdapter
import android.widget.EditText
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.ListView
import android.widget.ScrollView
import android.widget.TextView
import androidx.appcompat.app.AlertDialog
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import edu.cs6301.findback.map.Localizer
import edu.cs6301.findback.memory.ObjectRecord
import edu.cs6301.findback.scan.InventoryEntry
import edu.cs6301.findback.scan.InventoryText
import edu.cs6301.findback.ui.UiModel
import java.io.File
import kotlin.math.max

/**
 * v0.6 (docs/PLAN_v0.6_ui.md, section 6.5): the Items screen, a full-screen panel inside the
 * activity layout. Tapped records and scanned entries in one list, newest first, with a search over
 * names, labels and room names, and room filter chips. A row opens the item sheet. UI thread only.
 */
class ItemsPanel(private val activity: MainActivity, root: View) {
  /** What the panel lists, read when it opens and after every change. */
  class Data(
    val records: List<ObjectRecord>,
    val entries: List<InventoryEntry>,
    val lastScanId: String?,
    val state: Localizer.State?,
    val filesRoot: File,
  )

  /** What the item sheet's buttons do. */
  class Actions(
    val onFind: (ObjectRecord) -> Unit,
    val onMoved: (ObjectRecord) -> Unit,
    val onRename: (ObjectRecord, String) -> Unit,
    val onDelete: (ObjectRecord) -> Unit,
    val onFindEntry: (InventoryEntry) -> Unit,
    val onRenameEntry: (InventoryEntry, String) -> Unit,
    val onDeleteEntry: (InventoryEntry) -> Unit,
  )

  /** One row: a tapped record or a scanned entry. */
  private class Row(val record: ObjectRecord?, val entry: InventoryEntry?) {
    val lastSeenMs: Long
      get() = record?.lastSeenMs ?: entry!!.lastSeenMs
    val name: String
      get() = record?.name ?: entry!!.name
    val cropFile: String
      get() = record?.cropFile ?: entry!!.cropFile
  }

  private val panel: View = root.findViewById(R.id.itemsPanel)
  private val search: EditText = root.findViewById(R.id.itemsSearch)
  private val chipsScroll: View = root.findViewById(R.id.itemsChipsScroll)
  private val chips: LinearLayout = root.findViewById(R.id.itemsChips)
  private val empty: TextView = root.findViewById(R.id.itemsEmpty)
  private val list: ListView = root.findViewById(R.id.itemsList)
  private val basePaddingBottom = panel.paddingBottom

  private var data: Data? = null
  private var actions: Actions? = null
  private var all: List<Row> = emptyList()
  private val shown = ArrayList<Row>()
  /** Room id of the selected chip, or null for "All". */
  private var roomFilter: String? = null
  private val thumbs = HashMap<String, Bitmap?>()

  private val adapter =
    object : ArrayAdapter<Row>(activity, 0, shown) {
      override fun getView(position: Int, convertView: View?, parent: ViewGroup): View = rowView(getItem(position)!!)
    }

  val isOpen: Boolean
    get() = panel.visibility == View.VISIBLE

  init {
    list.adapter = adapter
    list.setOnItemClickListener { _, _, i, _ -> shown.getOrNull(i)?.let { showSheet(it) } }
    root.findViewById<View>(R.id.itemsClose).setOnClickListener { close() }
    search.addTextChangedListener(
      object : TextWatcher {
        override fun beforeTextChanged(s: CharSequence?, start: Int, count: Int, after: Int) {}

        override fun onTextChanged(s: CharSequence?, start: Int, before: Int, count: Int) {}

        override fun afterTextChanged(s: Editable?) = applyFilter()
      }
    )
    search.setOnEditorActionListener { _, id, _ ->
      if (id == EditorInfo.IME_ACTION_SEARCH) {
        hideKeyboard()
        true
      } else false
    }
    // The activity is edge-to-edge: keep the list above the keyboard by padding the panel.
    ViewCompat.setOnApplyWindowInsetsListener(panel) { v, insets ->
      val ime = insets.getInsets(WindowInsetsCompat.Type.ime()).bottom
      v.setPadding(v.paddingLeft, v.paddingTop, v.paddingRight, max(basePaddingBottom, ime))
      insets
    }
  }

  fun open(d: Data, a: Actions) {
    data = d
    actions = a
    roomFilter = null
    thumbs.clear()
    search.setText("")
    panel.visibility = View.VISIBLE
    rebuild()
  }

  fun close() {
    if (!isOpen) return
    hideKeyboard()
    search.clearFocus()
    panel.visibility = View.GONE
    data = null
    actions = null
    thumbs.clear()
    shown.clear()
    adapter.notifyDataSetChanged()
  }

  /** New data after a rename or delete; ignored when the panel is closed. */
  fun refresh(d: Data) {
    if (!isOpen) return
    data = d
    rebuild()
  }

  private fun rebuild() {
    val d = data ?: return
    all = (d.records.map { Row(it, null) } + d.entries.map { Row(null, it) }).sortedByDescending { it.lastSeenMs }
    val rooms = d.state?.map?.rooms ?: emptyList()
    if (roomFilter != null && rooms.none { it.id == roomFilter }) roomFilter = null
    chips.removeAllViews()
    if (rooms.size >= 2) {
      chips.addView(chip("All", null))
      for (r in rooms) chips.addView(chip(r.name, r.id))
      chipsScroll.visibility = View.VISIBLE
    } else {
      chipsScroll.visibility = View.GONE
    }
    applyFilter()
  }

  private fun chip(label: String, roomId: String?): TextView =
    TextView(activity).apply {
      text = label
      textSize = 14f
      gravity = Gravity.CENTER
      minHeight = dp(36)
      setPadding(dp(14), dp(6), dp(14), dp(6))
      setBackgroundResource(R.drawable.chip_light)
      isSelected = roomId == roomFilter
      setTextColor(
        ColorStateList(arrayOf(intArrayOf(android.R.attr.state_selected), intArrayOf()), intArrayOf(Color.WHITE, Color.rgb(0x42, 0x42, 0x42)))
      )
      layoutParams =
        LinearLayout.LayoutParams(ViewGroup.LayoutParams.WRAP_CONTENT, ViewGroup.LayoutParams.WRAP_CONTENT).apply {
          setMargins(dp(4), dp(4), dp(4), dp(4))
        }
      setOnClickListener {
        roomFilter = roomId
        for (i in 0 until chips.childCount) chips.getChildAt(i).isSelected = chips.getChildAt(i) === this
        applyFilter()
      }
    }

  /** The row's room in the current map, or null (no room, or another map). */
  private fun roomIdOf(row: Row, d: Data): String? {
    val mapId = d.state?.map?.mapId ?: return null
    val r = row.record
    if (r != null) return if (r.mapId == mapId) r.roomId else null
    val e = row.entry!!
    return if (e.mapId == mapId) e.roomId else null
  }

  private fun applyFilter() {
    val d = data ?: return
    val q = search.text?.toString() ?: ""
    val filtered =
      all.filter { row ->
        val roomId = roomIdOf(row, d)
        if (roomFilter != null && roomId != roomFilter) return@filter false
        val text = row.record?.let { InventoryText.matches(it, q) } ?: InventoryText.matches(row.entry!!, q)
        text || UiModel.roomMatches(d.state?.roomName(roomId), q)
      }
    adapter.setNotifyOnChange(false)
    adapter.clear()
    adapter.addAll(filtered)
    adapter.notifyDataSetChanged()
    val msg = UiModel.itemsEmptyText(all.size, filtered.size, q, roomFilter?.let { d.state?.roomName(it) })
    empty.text = msg
    empty.visibility = if (msg == null) View.GONE else View.VISIBLE
  }

  // ------------------------------------------------------------------ rows

  /** "Kitchen", "this session only" or "photo only" for a tapped record. */
  private fun recordPlace(r: ObjectRecord, d: Data): String {
    val mapId = d.state?.map?.mapId
    return when {
      r.mapId != null && r.mapId == mapId && r.mapPosition != null -> d.state?.roomName(r.roomId) ?: "room not named"
      r.sessionId == activity.sessionId -> "this session only"
      else -> "photo only"
    }
  }

  private fun entryPlace(e: InventoryEntry, d: Data): String {
    val mapId = d.state?.map?.mapId
    val inMap = mapId != null && e.mapId == mapId
    return InventoryText.location(e, d.state?.roomName(e.roomId), inMap)
  }

  private fun rowView(item: Row): View {
    val d = data
    val row =
      LinearLayout(activity).apply {
        orientation = LinearLayout.HORIZONTAL
        gravity = Gravity.CENTER_VERTICAL
        minimumHeight = dp(72)
        setPadding(dp(16), dp(8), dp(16), dp(8))
      }
    row.addView(
      ImageView(activity).apply {
        layoutParams = LinearLayout.LayoutParams(dp(56), dp(56))
        scaleType = ImageView.ScaleType.CENTER_CROP
        if (d != null) setImageBitmap(thumb(d.filesRoot, item.cropFile))
      }
    )
    val sb = SpannableStringBuilder(item.name)
    sb.setSpan(StyleSpan(Typeface.BOLD), 0, sb.length, Spanned.SPAN_EXCLUSIVE_EXCLUSIVE)
    if (d != null) {
      val now = System.currentTimeMillis()
      val r = item.record
      val e = item.entry
      sb.append('\n')
      if (r != null) {
        sb.append(recordPlace(r, d)).append(" · saved ").append(InventoryText.ago(r.lastSeenMs, now))
      } else if (e != null) {
        sb.append(entryPlace(e, d)).append(" · ").append(InventoryText.evidence(e, now, d.lastScanId))
        val start = sb.length
        sb.append("  from a scan")
        sb.setSpan(ForegroundColorSpan(MainView.GREY), start, sb.length, Spanned.SPAN_EXCLUSIVE_EXCLUSIVE)
        sb.setSpan(RelativeSizeSpan(0.85f), start, sb.length, Spanned.SPAN_EXCLUSIVE_EXCLUSIVE)
      }
    }
    row.addView(
      TextView(activity).apply {
        layoutParams =
          LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f).apply { marginStart = dp(12) }
        text = sb
        textSize = 15f
        setTextColor(Color.rgb(0x21, 0x21, 0x21))
      }
    )
    return row
  }

  private fun thumb(root: File, path: String): Bitmap? =
    thumbs.getOrPut(path) { BitmapFactory.decodeFile(File(root, path).path, BitmapFactory.Options().apply { inSampleSize = 4 }) }

  // ------------------------------------------------------------------ item sheet

  /** Large photo, name, location and evidence, [Find] [Close], and Rename / Moved it / Delete. */
  private fun showSheet(row: Row) {
    val d = data ?: return
    val a = actions ?: return
    val now = System.currentTimeMillis()
    val r = row.record
    val e = row.entry
    val box =
      LinearLayout(activity).apply {
        orientation = LinearLayout.VERTICAL
        setPadding(dp(20), dp(8), dp(20), 0)
      }
    BitmapFactory.decodeFile(File(d.filesRoot, row.cropFile).path)?.let { bmp ->
      box.addView(
        ImageView(activity).apply {
          setImageBitmap(bmp)
          adjustViewBounds = true
          maxHeight = dp(260)
        }
      )
    }
    val place: String
    val evidence: String
    var details: String? = null
    if (r != null) {
      place =
        when (recordPlace(r, d)) {
          "this session only" -> "Saved for this session only (not on the home map)"
          "photo only" -> "Photo only (saved before this map)"
          else -> recordPlace(r, d)
        }
      evidence = "Saved " + InventoryText.ago(r.lastSeenMs, now) + ". The marker shows where it was then."
    } else {
      e!!
      place = entryPlace(e, d)
      evidence = InventoryText.evidence(e, now, d.lastScanId).replaceFirstChar { it.uppercase() } + ". Found by a scan."
      details =
        "Detected as " + e.label + " · seen in " + e.sightingCount + (if (e.sightingCount == 1) " frame" else " frames") +
          " · first seen " + InventoryText.day(e.firstSeenMs)
    }
    box.addView(
      TextView(activity).apply {
        textSize = 16f
        setTextColor(Color.rgb(0x21, 0x21, 0x21))
        setPadding(0, dp(10), 0, 0)
        text = place
      }
    )
    box.addView(
      TextView(activity).apply {
        textSize = 14f
        setTextColor(Color.rgb(0x61, 0x61, 0x61))
        setPadding(0, dp(4), 0, 0)
        text = evidence
      }
    )
    details?.let {
      box.addView(
        TextView(activity).apply {
          textSize = 12f
          setTextColor(MainView.GREY)
          setPadding(0, dp(4), 0, 0)
          text = it
        }
      )
    }
    var sheet: AlertDialog? = null
    val buttons =
      LinearLayout(activity).apply {
        orientation = LinearLayout.HORIZONTAL
        setPadding(0, dp(8), 0, 0)
      }
    fun textButton(label: String, onClick: () -> Unit) =
      buttons.addView(
        TextView(activity).apply {
          text = label
          textSize = 15f
          gravity = Gravity.CENTER
          minHeight = dp(48)
          setPadding(dp(12), 0, dp(12), 0)
          setTextColor(Color.rgb(0x1E, 0x88, 0xE5))
          val tv = TypedValue()
          activity.theme.resolveAttribute(android.R.attr.selectableItemBackground, tv, true)
          setBackgroundResource(tv.resourceId)
          setOnClickListener {
            sheet?.dismiss()
            onClick()
          }
        }
      )
    textButton("Rename") { askName(row.name) { n -> if (r != null) a.onRename(r, n) else a.onRenameEntry(e!!, n) } }
    if (r != null) {
      textButton("Moved it") {
        close()
        a.onMoved(r)
      }
    }
    textButton("Delete") { confirmDelete(row, a) }
    box.addView(buttons)
    sheet =
      AlertDialog.Builder(activity)
        .setTitle(row.name)
        .setView(ScrollView(activity).apply { addView(box) })
        .setPositiveButton("Find") { _, _ ->
          close()
          if (r != null) a.onFind(r) else a.onFindEntry(e!!)
        }
        .setNegativeButton("Close", null)
        .show()
  }

  private fun confirmDelete(row: Row, a: Actions) {
    val r = row.record
    val e = row.entry
    AlertDialog.Builder(activity)
      .setTitle("Delete \"" + row.name + "\"?")
      .setMessage(
        if (r != null) "This removes the item from the list. Its photo and study logs are kept for analysis."
        else "This removes the scanned item from the list. Its photo and logs are kept for analysis."
      )
      .setPositiveButton("Delete") { _, _ -> if (r != null) a.onDelete(r) else a.onDeleteEntry(e!!) }
      .setNegativeButton("Cancel", null)
      .show()
  }

  /** Rename: the new name, trimmed, when it is not empty and differs from [current]. */
  private fun askName(current: String, onName: (String) -> Unit) {
    val edit =
      EditText(activity).apply {
        setText(current)
        setSingleLine()
        setSelectAllOnFocus(true)
      }
    val box = LinearLayout(activity).apply { setPadding(dp(20), dp(8), dp(20), 0); addView(edit) }
    val dialog =
      AlertDialog.Builder(activity)
        .setTitle("Rename")
        .setView(box)
        .setPositiveButton("Save") { _, _ ->
          val n = edit.text.toString().trim()
          if (n.isNotEmpty() && n != current) onName(n)
        }
        .setNegativeButton("Cancel", null)
        .create()
    // Renaming means typing: open the keyboard with the old name selected.
    dialog.window?.setSoftInputMode(WindowManager.LayoutParams.SOFT_INPUT_STATE_ALWAYS_VISIBLE)
    dialog.show()
    edit.requestFocus()
  }

  private fun hideKeyboard() {
    val imm = activity.getSystemService(InputMethodManager::class.java) ?: return
    imm.hideSoftInputFromWindow(search.windowToken, 0)
  }

  private fun dp(v: Int) = (v * activity.resources.displayMetrics.density).toInt()
}
