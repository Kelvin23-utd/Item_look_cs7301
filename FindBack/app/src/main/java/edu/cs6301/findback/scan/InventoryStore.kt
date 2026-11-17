package edu.cs6301.findback.scan

import java.io.File
import org.json.JSONArray
import org.json.JSONObject

/**
 * The scan inventory (plan section 6): inventory.json in the app's files root, next to records.json
 * and never mixed with it, so the v0.4 experiments are unaffected.
 *
 * ```
 * { "version": 1, "lastScan": { "id": "<scan id>", "endMs": 0 } | null, "entries": [ <entry>... ] }
 * ```
 *
 * Every change is also appended to [LOG_NAME], one JSON object per line:
 * ```
 * {"t": epoch ms, "session": id, "op": "add"|"update"|"rename"|"delete", "entry": { full entry }}
 * {"t": epoch ms, "session": id, "op": "evidence", "id": "<entry id>", "lastSeenMs": 0,
 *  "evidence": { ... }, "source": "scan"|"passive"|"verify", "sourceId": "..."}
 * ```
 * `add`: a new entry. `update`: the entry as it is after a scan that sighted it (position, votes,
 * best crop and evidence). `evidence`: only the evidence fields changed. `rename`, `delete`: by the
 * user; `delete` lines written when "New map" moves the file away carry `"reason": "map_archived"`.
 * Replaying the lines in order gives the inventory at any time.
 *
 * The in-memory list is an immutable snapshot replaced on every change, so any thread may read
 * [snapshot]. Changes are synchronized; the scan worker makes them. The file is rewritten by
 * [flush] (temp file, then rename), not on every change: a scan changes many entries per second.
 * No Android imports, so the file formats are unit-tested.
 */
class InventoryStore(val root: File, private val sessionId: String = "") {
  companion object {
    const val FILE_NAME = "inventory.json"
    const val LOG_NAME = "inventory_log.jsonl"
    const val VERSION = 1
  }

  private val file = File(root, FILE_NAME)
  private val logFile = File(root, LOG_NAME)

  @Volatile private var entries: List<InventoryEntry> = emptyList()

  /** The most recent finished scan, for "looked for in the last scan". */
  @Volatile
  var lastScanId: String? = null
    private set

  @Volatile
  var lastScanEndMs: Long? = null
    private set

  /** Set when inventory.json could not be read (it was copied aside and the list starts empty). */
  var loadError: Exception? = null
    private set

  private var dirty = false
  /** Set by [archiveTo]: nothing is written until the activity restarts with a fresh store. */
  private var closed = false

  init {
    load()
  }

  /** The entries right now. The list never changes; a later call may return another list. */
  fun snapshot(): List<InventoryEntry> = entries

  fun get(id: String): InventoryEntry? = entries.firstOrNull { it.id == id }

  /** A new entry: one `add` line. */
  @Synchronized
  fun add(entry: InventoryEntry) {
    if (closed) return
    entries = entries.filter { it.id != entry.id } + entry
    dirty = true
    appendEntryLine("add", entry)
  }

  /**
   * [entry] takes the place of the entry with its id, without a log line (a scan does this for
   * every sighting, then calls [logUpdate] once at its end). False when the entry no longer exists.
   */
  @Synchronized
  fun replace(entry: InventoryEntry): Boolean {
    if (closed) return false
    val i = entries.indexOfFirst { it.id == entry.id }
    if (i < 0) return false
    entries = entries.toMutableList().also { it[i] = entry }
    dirty = true
    return true
  }

  /**
   * Apply [f] to the entry with [id] as it is now, under the lock, without a log line. A change
   * another thread made meanwhile (the user's rename during a keyframe) is kept, which a
   * [replace] with a copy made earlier would undo. Null when the entry no longer exists.
   */
  @Synchronized
  fun update(id: String, f: (InventoryEntry) -> InventoryEntry): InventoryEntry? {
    if (closed) return null
    val i = entries.indexOfFirst { it.id == id }
    if (i < 0) return null
    val e = f(entries[i])
    entries = entries.toMutableList().also { it[i] = e }
    dirty = true
    return e
  }

  /** One `update` line with the entry as it is now. */
  @Synchronized
  fun logUpdate(id: String) {
    val e = get(id) ?: return
    appendEntryLine("update", e)
  }

  /** One `evidence` line with the evidence fields of the entry as it is now. */
  @Synchronized
  fun logEvidence(id: String): Boolean {
    val entry = get(id) ?: return false
    val o = JSONObject()
    o.put("t", System.currentTimeMillis())
    o.put("session", sessionId)
    o.put("op", "evidence")
    o.put("id", entry.id)
    o.put("lastSeenMs", entry.lastSeenMs)
    o.put("evidence", entry.evidenceJson())
    o.put("source", entry.evidence.lastSource ?: JSONObject.NULL)
    o.put("sourceId", entry.evidence.lastSourceId ?: JSONObject.NULL)
    appendLine(o)
    return true
  }

  /** The user's own name for an entry; it then no longer follows the label. Saved at once. */
  @Synchronized
  fun rename(id: String, name: String): InventoryEntry? {
    val old = get(id) ?: return null
    val e = old.copy(name = name, nameEdited = true)
    if (!replace(e)) return null
    appendEntryLine("rename", e)
    flush()
    return e
  }

  /** Remove an entry from the list. Its photo and log lines stay. Saved at once. */
  @Synchronized
  fun delete(id: String): InventoryEntry? {
    if (closed) return null
    val old = get(id) ?: return null
    entries = entries.filter { it.id != id }
    dirty = true
    appendEntryLine("delete", old)
    flush()
    return old
  }

  @Synchronized
  fun setLastScan(id: String, endMs: Long) {
    if (closed) return
    lastScanId = id
    lastScanEndMs = endMs
    dirty = true
  }

  @Synchronized fun isDirty(): Boolean = dirty

  /** Write inventory.json if anything changed. Returns true when a file was written. */
  @Synchronized
  fun flush(): Boolean {
    if (closed || !dirty) return false
    val o = JSONObject()
    o.put("version", VERSION)
    o.put(
      "lastScan",
      lastScanId?.let { id -> JSONObject().put("id", id).put("endMs", lastScanEndMs ?: 0L) } ?: JSONObject.NULL,
    )
    o.put("entries", JSONArray().apply { entries.forEach { put(it.toJson()) } })
    root.mkdirs()
    val tmp = File(root, "$FILE_NAME.tmp")
    tmp.writeText(o.toString(1))
    if (!tmp.renameTo(file)) {
      tmp.copyTo(file, overwrite = true)
      tmp.delete()
    }
    dirty = false
    return true
  }

  /**
   * "New map": save, move inventory.json into [dir] and stop writing. The photos and the log stay
   * where they are; nothing is deleted. Returns false when there was no file to move.
   */
  @Synchronized
  fun archiveTo(dir: File): Boolean {
    if (!closed) {
      flush()
      for (e in entries) appendEntryLine("delete", e, "reason" to "map_archived")
    }
    closed = true
    if (!file.exists()) return false
    dir.mkdirs()
    val dest = File(dir, FILE_NAME)
    if (file.renameTo(dest)) return true
    file.copyTo(dest, overwrite = false)
    return file.delete()
  }

  private fun load() {
    if (!file.exists()) return
    try {
      val o = JSONObject(file.readText())
      val arr = o.optJSONArray("entries") ?: JSONArray()
      entries = (0 until arr.length()).map { InventoryEntry.fromJson(arr.getJSONObject(it)) }
      if (!o.isNull("lastScan")) {
        val s = o.getJSONObject("lastScan")
        lastScanId = s.optString("id", "").ifEmpty { null }
        lastScanEndMs = s.optLong("endMs", 0L)
      }
    } catch (e: Exception) {
      // A corrupt file is copied aside, never deleted; the inventory starts empty.
      loadError = e
      entries = emptyList()
      try {
        file.copyTo(File(root, "inventory.corrupt.${System.currentTimeMillis()}.json"))
      } catch (e2: Exception) {
        // Nothing more to do here; the caller logs loadError.
      }
    }
  }

  /** Caller holds the lock, so lines keep the order of the changes. */
  private fun appendEntryLine(op: String, entry: InventoryEntry, extra: Pair<String, Any>? = null) {
    val o = JSONObject()
    o.put("t", System.currentTimeMillis())
    o.put("session", sessionId)
    o.put("op", op)
    o.put("entry", entry.toJson())
    if (extra != null) o.put(extra.first, extra.second)
    appendLine(o)
  }

  private fun appendLine(o: JSONObject) {
    if (closed) return
    root.mkdirs()
    logFile.appendText(o.toString() + "\n")
  }
}
