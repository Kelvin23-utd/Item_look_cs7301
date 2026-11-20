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
import androidx.core.view.WindowCompat
import androidx.core.view.WindowInsetsCompat
import edu.cs6301.findback.map.Localizer
import edu.cs6301.findback.memory.ObjectRecord
import edu.cs6301.findback.scan.InventoryEntry
import edu.cs6301.findback.scan.InventoryText
import edu.cs6301.findback.search.ExtraNames
import edu.cs6301.findback.search.ExtraNamesStore
import edu.cs6301.findback.search.NameSearch
import edu.cs6301.findback.search.NameSet
import edu.cs6301.findback.search.SearchHit
import edu.cs6301.findback.search.SearchItem
import edu.cs6301.findback.trace.UiTrace
import edu.cs6301.findback.ui.UiModel
import java.io.File
import kotlin.math.max

/**
 * v0.6 (docs/PLAN_v0.6_ui.md, section 6.5): the Items screen, a full-screen panel inside the
 * activity layout. Tapped records and scanned entries in one list, newest first, and room filter
 * chips. A row opens the item sheet. UI thread only.
 *
 * v0.9 (docs/PLAN_v0.9_find_by_name.md): the search is [NameSearch] (results by match class, then
 * newest first) over names, labels, containers, rooms and, with "Extra names" on, the names and
 * descriptors names.json binds to the item's current photo. The binding is read off the UI thread
 * after each open and refresh. A "Last scan" chip keeps the entries seen in the most recent
 * finished scan.
 */
class ItemsPanel(private val activity: MainActivity, root: View, private val ui: UiTrace) {
  companion object {
    /** v0.7 `ui_search`: the search text counts as settled after this long without a change. */
    const val SEARCH_SETTLE_MS = 1000L

    /** [recordPlace] of a tapped record that is not on the current map. */
    private const val PLACE_SESSION_ONLY = "this session only"
    private const val PLACE_PHOTO_ONLY = "photo only"
  }

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

  /**
   * One row: a tapped record or a scanned entry. [roomId]: its room in the current map, or null (no
   * room, or another map). [inLastScan]: an entry seen in the most recent finished scan. [search]:
   * what the search sees of it.
   */
  private class Row(
    val record: ObjectRecord?,
    val entry: InventoryEntry?,
    val roomId: String?,
    val inLastScan: Boolean,
    val search: SearchItem,
  ) {
    val name: String
      get() = record?.name ?: entry!!.name
    val cropFile: String
      get() = record?.cropFile ?: entry!!.cropFile
    /** `items_row` detail and names.json kind: "record" or "entry". */
    val kind: String
      get() = if (record != null) ExtraNames.KIND_RECORD else ExtraNames.KIND_ENTRY
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
  private val shown = ArrayList<SearchHit<Row>>()
  /** Room id of the selected chip, or null for "All" and "Last scan". */
  private var roomFilter: String? = null
  /** v0.9: the "Last scan" chip is selected. */
  private var lastScanOnly = false
  /** v0.9: "Extra names" as it was when the panel opened. */
  private var extraOn = MainActivity.DEFAULT_EXTRA_NAMES
  /** v0.9: bound extra names by [ExtraNames.key], from the last finished [bindNames]. */
  private var bound: Map<String, NameSet> = emptyMap()
  /** v0.9: numbers each [bindNames]; a result of an older one, or after close, is dropped. */
  private var bindSeq = 0
  private val thumbs = HashMap<String, Bitmap?>()

  /** v0.7: the search text at the last `ui_search` (kept in memory only, never logged). */
  private var searchLogged = ""
  private val searchSettled = Runnable { logSearch() }

  private val adapter =
    object : ArrayAdapter<SearchHit<Row>>(activity, 0, shown) {
      override fun getView(position: Int, convertView: View?, parent: ViewGroup): View = rowView(getItem(position)!!)
    }

  val isOpen: Boolean
    get() = panel.visibility == View.VISIBLE

  init {
    list.adapter = adapter
    list.setOnItemClickListener { _, _, i, _ ->
      hideKeyboard() // the item sheet should not open over the keyboard
      shown.getOrNull(i)?.let { hit ->
        // v0.9: its rank, the rows shown, the query length (never the text) and how it matched.
        ui.itemsRow(hit.item.kind, i + 1, shown.size, search.text?.length ?: 0, hit.matchClass, hit.viaExtra)
        showSheet(hit.item)
      }
    }
    root.findViewById<View>(R.id.itemsClose).setOnClickListener { close("button") }
    search.addTextChangedListener(
      object : TextWatcher {
        override fun beforeTextChanged(s: CharSequence?, start: Int, count: Int, after: Int) {}

        override fun onTextChanged(s: CharSequence?, start: Int, before: Int, count: Int) {}

        override fun afterTextChanged(s: Editable?) {
          applyFilter()
          if (ui.on && isOpen) {
            search.removeCallbacks(searchSettled)
            search.postDelayed(searchSettled, SEARCH_SETTLE_MS)
          }
        }
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
    lastScanOnly = false
    extraOn = activity.extraNamesOn
    bound = emptyMap()
    thumbs.clear()
    search.setText("")
    search.removeCallbacks(searchSettled)
    searchLogged = ""
    ui.action("items_open")
    panel.visibility = View.VISIBLE
    lightStatusBar(true) // dark status-bar icons on the white panel
    rebuild()
    bindNames()
  }

  /** [reason] (v0.7 `ui_action.detail`): what closed it ("button", "back", "find", "moved", "scan"). */
  fun close(reason: String = "other") {
    if (!isOpen) return
    search.removeCallbacks(searchSettled)
    logSearch() // a search typed and left without waiting a second
    ui.action("items_close", reason)
    hideKeyboard()
    search.clearFocus()
    panel.visibility = View.GONE
    lightStatusBar(false)
    data = null
    actions = null
    bindSeq++
    bound = emptyMap()
    thumbs.clear()
    shown.clear()
    adapter.notifyDataSetChanged()
  }

  /** New data after a rename or delete; ignored when the panel is closed. */
  fun refresh(d: Data) {
    if (!isOpen) return
    data = d
    rebuild()
    bindNames()
  }

  /**
   * The rows and the chips. Room chips ("All" and one per room) need two rooms; "Last scan" (v0.9)
   * needs an entry seen in the most recent finished scan. The row of chips shows when either does.
   */
  private fun rebuild() {
    val d = data ?: return
    all = rows(d)
    val rooms = d.state?.map?.rooms ?: emptyList()
    val roomChips = rooms.size >= 2
    if (roomFilter != null && (!roomChips || rooms.none { it.id == roomFilter })) roomFilter = null
    val lastScanChip = all.any { it.inLastScan }
    if (!lastScanChip) lastScanOnly = false
    chips.removeAllViews()
    if (roomChips || lastScanChip) {
      chips.addView(
        chip("All", roomFilter == null && !lastScanOnly) {
          ui.action("items_room", "All")
          roomFilter = null
          lastScanOnly = false
        }
      )
      if (lastScanChip) {
        chips.addView(
          chip("Last scan", lastScanOnly) {
            ui.action("items_last_scan")
            roomFilter = null
            lastScanOnly = true
          }
        )
      }
      if (roomChips) {
        for (r in rooms) {
          chips.addView(
            chip(r.name, roomFilter == r.id) {
              ui.action("items_room", r.name)
              roomFilter = r.id
              lastScanOnly = false
            }
          )
        }
      }
      chipsScroll.visibility = View.VISIBLE
    } else {
      chipsScroll.visibility = View.GONE
    }
    applyFilter()
  }

  /** Every tapped record and scanned entry, with the extra names [bound] to it. */
  private fun rows(d: Data): List<Row> {
    val mapId = d.state?.map?.mapId
    val out = ArrayList<Row>(d.records.size + d.entries.size)
    for (r in d.records) {
      val roomId = if (mapId != null && r.mapId == mapId) r.roomId else null
      val extra = bound[ExtraNames.key(ExtraNames.KIND_RECORD, r.id)]
      val si =
        SearchItem(
          id = r.id,
          name = r.name,
          labels = listOf(r.rawLabel),
          container = null,
          room = d.state?.roomName(roomId),
          lastSeenMs = r.lastSeenMs,
          extraNames = extra?.names ?: emptyList(),
          descriptors = extra?.descriptors ?: emptyList(),
        )
      out.add(Row(r, null, roomId, false, si))
    }
    for (e in d.entries) {
      val roomId = if (mapId != null && e.mapId == mapId) e.roomId else null
      val extra = bound[ExtraNames.key(ExtraNames.KIND_ENTRY, e.id)]
      val si =
        SearchItem(
          id = e.id,
          name = e.name,
          labels = (listOf(e.label) + e.labelVotes.keys).distinct(),
          container = e.container,
          room = d.state?.roomName(roomId),
          lastSeenMs = e.lastSeenMs,
          extraNames = extra?.names ?: emptyList(),
          descriptors = extra?.descriptors ?: emptyList(),
        )
      out.add(Row(null, e, roomId, InventoryText.seenInScan(e, d.lastScanId), si))
    }
    return out
  }

  /**
   * v0.9: the extra names of the listed items, bound off the UI thread (photos not hashed yet are
   * read there), then the search runs again with them. Without names.json nothing is read.
   */
  private fun bindNames() {
    val d = data ?: return
    val seq = ++bindSeq
    activity.extraNames.bindAsync(ExtraNamesStore.refs(d.records, d.entries)) { result ->
      panel.post {
        if (seq != bindSeq || data == null || result == bound) return@post
        bound = result
        data?.let { all = rows(it) }
        applyFilter()
      }
    }
  }

  /** A filter chip; a tap runs [onSelect], selects it alone and filters again. */
  private fun chip(label: String, selected: Boolean, onSelect: () -> Unit): TextView =
    TextView(activity).apply {
      text = label
      textSize = 14f
      gravity = Gravity.CENTER
      minHeight = dp(36)
      setPadding(dp(14), dp(6), dp(14), dp(6))
      setBackgroundResource(R.drawable.chip_light)
      isSelected = selected
      setTextColor(
        ColorStateList(arrayOf(intArrayOf(android.R.attr.state_selected), intArrayOf()), intArrayOf(Color.WHITE, Color.rgb(0x42, 0x42, 0x42)))
      )
      layoutParams =
        LinearLayout.LayoutParams(ViewGroup.LayoutParams.WRAP_CONTENT, ViewGroup.LayoutParams.WRAP_CONTENT).apply {
          setMargins(dp(4), dp(4), dp(4), dp(4))
        }
      setOnClickListener {
        onSelect()
        for (i in 0 until chips.childCount) chips.getChildAt(i).isSelected = chips.getChildAt(i) === this
        applyFilter()
      }
    }

  /** The selected chip's rows, searched and ordered by [NameSearch]. */
  private fun applyFilter() {
    val d = data ?: return
    val q = search.text?.toString() ?: ""
    val pool =
      when {
        lastScanOnly -> all.filter { it.inLastScan }
        roomFilter != null -> all.filter { it.roomId == roomFilter }
        else -> all
      }
    val hits = NameSearch.search(pool, q, extraOn) { it.search }
    adapter.setNotifyOnChange(false)
    adapter.clear()
    adapter.addAll(hits)
    adapter.notifyDataSetChanged()
    val scope = if (lastScanOnly) UiModel.LAST_SCAN_SCOPE else roomFilter?.let { d.state?.roomName(it) }
    val msg = UiModel.itemsEmptyText(all.size, hits.size, q, scope)
    empty.text = msg
    empty.visibility = if (msg == null) View.GONE else View.VISIBLE
  }

  /**
   * v0.7 `ui_search`, when the text changed since the last one: its length and the rows shown.
   * v0.9: the "Extra names" setting, the items bound, the shown rows via extra, the "Last scan" chip.
   */
  private fun logSearch() {
    if (!ui.on || data == null) return
    val q = search.text?.toString() ?: ""
    if (q == searchLogged) return
    searchLogged = q
    ui.search(
      q.length,
      shown.size,
      all.size,
      roomFilter?.let { data?.state?.roomName(it) ?: it },
      extraOn,
      bound.size,
      shown.count { it.viaExtra },
      lastScanOnly,
    )
  }

  // ------------------------------------------------------------------ rows

  /** "Kitchen", "this session only" or "photo only" for a tapped record. */
  private fun recordPlace(r: ObjectRecord, d: Data): String {
    val mapId = d.state?.map?.mapId
    return when {
      r.mapId != null && r.mapId == mapId && r.mapPosition != null -> d.state?.roomName(r.roomId) ?: "room not named"
      r.sessionId == activity.sessionId -> PLACE_SESSION_ONLY
      else -> PLACE_PHOTO_ONLY
    }
  }

  /**
   * v0.9: the row's second line and the item sheet's first line. "Last seen: Kitchen · 2 d ago",
   * then an entry's miss note; a tapped record off the map: "Last seen: 5 min ago · photo only".
   */
  private fun lastSeenText(row: Row, d: Data, now: Long): String {
    val r = row.record
    if (r != null) {
      return when (val place = recordPlace(r, d)) {
        PLACE_SESSION_ONLY, PLACE_PHOTO_ONLY -> InventoryText.lastSeenLine(null, r.lastSeenMs, now, place)
        else -> InventoryText.lastSeenLine(place, r.lastSeenMs, now)
      }
    }
    val e = row.entry!!
    return InventoryText.lastSeenLine(entryPlace(e, d), e.lastSeenMs, now, InventoryText.missNote(e, now, d.lastScanId))
  }

  private fun entryPlace(e: InventoryEntry, d: Data): String {
    val mapId = d.state?.map?.mapId
    val inMap = mapId != null && e.mapId == mapId
    return InventoryText.location(e, d.state?.roomName(e.roomId), inMap)
  }

  private fun rowView(hit: SearchHit<Row>): View {
    val item = hit.item
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
    // v0.9: a result only because of extra names says which one matched.
    if (hit.viaExtra && hit.extraMatches.isNotEmpty()) greyNote(sb, "  also: " + hit.extraMatches.joinToString(", "))
    if (d != null) {
      sb.append('\n').append(lastSeenText(item, d, System.currentTimeMillis()))
      if (item.entry != null) greyNote(sb, "  from a scan")
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

  /** [text] appended in the small grey style. */
  private fun greyNote(sb: SpannableStringBuilder, text: String) {
    val start = sb.length
    sb.append(text)
    sb.setSpan(ForegroundColorSpan(MainView.GREY), start, sb.length, Spanned.SPAN_EXCLUSIVE_EXCLUSIVE)
    sb.setSpan(RelativeSizeSpan(0.85f), start, sb.length, Spanned.SPAN_EXCLUSIVE_EXCLUSIVE)
  }

  private fun thumb(root: File, path: String): Bitmap? =
    thumbs.getOrPut(path) { BitmapFactory.decodeFile(File(root, path).path, BitmapFactory.Options().apply { inSampleSize = 4 }) }

  // ------------------------------------------------------------------ item sheet

  /**
   * Large photo, name, the row's last-seen line, what the place means, [Find] [Close], and Rename /
   * Moved it / Delete.
   */
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
    val place = lastSeenText(row, d, now)
    val evidence: String
    var details: String? = null
    if (r != null) {
      evidence =
        when (recordPlace(r, d)) {
          PLACE_SESSION_ONLY -> "Saved for this session only (not on the home map). The marker shows where it was then."
          PLACE_PHOTO_ONLY -> "Photo only (saved before this map)."
          else -> "The marker shows where it was then."
        }
    } else {
      e!!
      evidence = "Found by a scan."
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
            ui.action("dialog:item_sheet", label)
            sheet?.dismiss()
            onClick()
          }
        }
      )
    textButton("Rename") { askName(row.name) { n -> if (r != null) a.onRename(r, n) else a.onRenameEntry(e!!, n) } }
    if (r != null) {
      textButton("Moved it") {
        close("moved")
        a.onMoved(r)
      }
    }
    textButton("Delete") { confirmDelete(row, a) }
    box.addView(buttons)
    sheet =
      ui.dialog(activity, "item_sheet")
        .setTitle(row.name)
        .setView(ScrollView(activity).apply { addView(box) })
        .setPositiveButton("Find") { _, _ ->
          close("find")
          if (r != null) a.onFind(r) else a.onFindEntry(e!!)
        }
        .setNegativeButton("Close", null)
        .show()
  }

  private fun confirmDelete(row: Row, a: Actions) {
    val r = row.record
    val e = row.entry
    ui.dialog(activity, "item_delete")
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
      ui.dialog(activity, "item_rename")
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

  private fun lightStatusBar(light: Boolean) {
    WindowCompat.getInsetsController(activity.window, activity.window.decorView).isAppearanceLightStatusBars = light
  }

  private fun hideKeyboard() {
    val imm = activity.getSystemService(InputMethodManager::class.java) ?: return
    imm.hideSoftInputFromWindow(search.windowToken, 0)
  }

  private fun dp(v: Int) = (v * activity.resources.displayMetrics.density).toInt()
}
