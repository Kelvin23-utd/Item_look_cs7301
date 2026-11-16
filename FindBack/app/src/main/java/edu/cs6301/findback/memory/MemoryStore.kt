package edu.cs6301.findback.memory

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.util.Log
import edu.cs6301.findback.ml.Embedder
import java.io.File
import org.json.JSONArray
import org.json.JSONObject

/**
 * Local object memory: records.json plus crop images, stored in the app's external files directory
 * so the whole study dataset can be pulled with
 *   adb pull /sdcard/Android/data/edu.cs6301.findback/files ./findback_data
 *
 * records.json holds the active records only. Every change is also appended to
 * [RECORDS_LOG] (v0.4, S5), one JSON object per line:
 *   {"t": epoch ms, "session": id, "op": "add"|"rename"|"delete"|"supersede", "record": {...}}
 * where `record` is the full record as in records.json, embedding included: the new record for
 * `add` and `supersede` (its `supersedes` names the record that left the active list), the record
 * under its new name for `rename`, and the removed record for `delete`. Replaying the lines in
 * order gives the active list at any time, and offline analysis gets every record ever stored.
 * Two kinds of line carry one more key. `"backfill": true` marks the `add` lines reconstructed,
 * once, for records that were already in records.json when the log was first created: their `t`
 * and `session` are the record's own `lastSeenMs` and `sessionId`, as the line would have been
 * written at registration. `"reason": "map_archived"` marks the `delete` lines written when
 * "New map" moves records.json away.
 */
class MemoryStore(val root: File, private val sessionId: String = "") {
  companion object {
    const val RECORDS_LOG = "records_log.jsonl"
  }

  private val recordsFile = File(root, "records.json")
  private val logFile = File(root, RECORDS_LOG)
  private val records = mutableListOf<ObjectRecord>()
  /** Set by [archiveTo]: nothing is written until the activity restarts with a fresh store. */
  private var closed = false

  init {
    File(root, "crops").mkdirs()
    File(root, "verify").mkdirs()
    load()
    // First start with a records log: the records already stored have no "add" line yet.
    // Reconstruct it as of their registration, so a replay by time is right for older sessions too.
    if (!logFile.exists()) {
      for (r in records) appendLog("add", r, "backfill" to true, tMs = r.lastSeenMs, session = r.sessionId)
    }
  }

  @Synchronized fun snapshot(): List<ObjectRecord> = records.toList()

  @Synchronized fun get(id: String): ObjectRecord? = records.firstOrNull { it.id == id }

  @Synchronized
  fun add(record: ObjectRecord) {
    records.add(record)
    save()
    appendLog("add", record)
  }

  /** Rename only; positions, crops and embeddings stay frozen for the study. */
  @Synchronized
  fun rename(id: String, name: String) {
    val i = records.indexOfFirst { it.id == id }
    if (i < 0) return
    records[i] = records[i].copy(name = name)
    save()
    appendLog("rename", records[i])
  }

  @Synchronized
  fun delete(id: String) {
    val removed = records.filter { it.id == id }
    if (removed.isEmpty()) return
    records.removeAll { it.id == id }
    save()
    for (r in removed) appendLog("delete", r)
  }

  /**
   * "Update location" (S7): [record] takes the place of the record [oldId] in the active list, in
   * one save. The old record's crop, verification files and log lines stay. Returns the record that
   * left the list, or null when [oldId] was not active (then [record] is simply added).
   */
  @Synchronized
  fun supersede(oldId: String, record: ObjectRecord): ObjectRecord? {
    val i = records.indexOfFirst { it.id == oldId }
    val old = if (i >= 0) records.removeAt(i) else null
    records.add(record)
    save()
    appendLog("supersede", record)
    return old
  }

  /** Highest appearance similarity between [record] and any other stored record (distractor risk). */
  @Synchronized
  fun maxSimilarityToOthers(record: ObjectRecord): Float =
    records.filter { it.id != record.id }.maxOfOrNull { Embedder.cosine(it.embedding, record.embedding) }
      ?: 0f

  fun saveJpeg(relativePath: String, bitmap: Bitmap, quality: Int = 90): Long {
    val f = File(root, relativePath)
    f.parentFile?.mkdirs()
    f.outputStream().use { bitmap.compress(Bitmap.CompressFormat.JPEG, quality, it) }
    return f.length()
  }

  fun loadBitmap(relativePath: String): Bitmap? = BitmapFactory.decodeFile(File(root, relativePath).path)

  /** Write a small text file under the files root (the embedding file of a verification). */
  fun saveText(relativePath: String, text: String): Long {
    val f = File(root, relativePath)
    f.parentFile?.mkdirs()
    f.writeText(text)
    return f.length()
  }

  fun bytesFor(record: ObjectRecord): Long =
    File(root, record.cropFile).length() + record.toJson().toString().length

  private fun load() {
    if (!recordsFile.exists()) return
    try {
      val arr = JSONArray(recordsFile.readText())
      for (i in 0 until arr.length()) records.add(ObjectRecord.fromJson(arr.getJSONObject(i)))
    } catch (e: Exception) {
      Log.e("MemoryStore", "Failed to load records; starting empty", e)
      recordsFile.copyTo(File(root, "records.corrupt.${System.currentTimeMillis()}.json"))
    }
  }

  /**
   * "New map": move records.json into [dir] and stop writing it. Crops, verification frames and the
   * event log stay where they are. Nothing is deleted. Returns false when there was no file.
   */
  @Synchronized
  fun archiveTo(dir: File): Boolean {
    if (!closed) for (r in records) appendLog("delete", r, "reason" to "map_archived")
    closed = true
    if (!recordsFile.exists()) return false
    dir.mkdirs()
    val dest = File(dir, "records.json")
    if (recordsFile.renameTo(dest)) return true
    recordsFile.copyTo(dest, overwrite = false)
    return recordsFile.delete()
  }

  private fun save() {
    if (closed) return
    val arr = JSONArray().apply { records.forEach { put(it.toJson()) } }
    val tmp = File(root, "records.json.tmp")
    tmp.writeText(arr.toString(2))
    tmp.renameTo(recordsFile)
  }

  /** One line of [RECORDS_LOG]. Caller holds the lock, so lines keep the order of the changes. */
  private fun appendLog(
    op: String,
    record: ObjectRecord,
    extra: Pair<String, Any>? = null,
    tMs: Long = System.currentTimeMillis(),
    session: String = sessionId,
  ) {
    if (closed) return
    val o = JSONObject()
    o.put("t", tMs)
    o.put("session", session)
    o.put("op", op)
    o.put("record", record.toJson())
    if (extra != null) o.put(extra.first, extra.second)
    logFile.appendText(o.toString() + "\n")
  }
}
