package edu.cs6301.findback.search

import edu.cs6301.findback.memory.ObjectRecord
import edu.cs6301.findback.scan.InventoryEntry
import java.io.File
import java.security.MessageDigest
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

/** An item whose photo may carry extra names: its kind ("entry", "record"), id and photo path relative to the files root. */
data class PhotoRef(val kind: String, val id: String, val cropFile: String)

/**
 * v0.9: names.json in the files root ([root], the folder of records.json and inventory.json), read
 * once at start, and the binding of its items to the current photos. A photo's SHA-256 is cached
 * per (path, length, last modified), so only a new or replaced photo is read again. No Android
 * imports. [bind] reads files: call it off the UI thread ([bindAsync]), or once at start.
 */
class ExtraNamesStore(private val root: File) {
  companion object {
    /** The photos of [records] and [entries], for [bind]. */
    fun refs(records: List<ObjectRecord>, entries: List<InventoryEntry>): List<PhotoRef> =
      records.map { PhotoRef(ExtraNames.KIND_RECORD, it.id, it.cropFile) } +
        entries.map { PhotoRef(ExtraNames.KIND_ENTRY, it.id, it.cropFile) }
  }

  private data class Stamp(val length: Long, val lastModified: Long)

  /** names.json as read, or null when it is missing or cannot be used ([loadError]). */
  val file: NamesFile?

  /** Why there are no extra names: "missing", "unreadable: ...", "not JSON", "version 2", "no items"; null when [file] is set. */
  val loadError: String?

  private val hashes = HashMap<String, Pair<Stamp, String>>()
  private val worker: ExecutorService by lazy {
    Executors.newSingleThreadExecutor { r -> Thread(r, "extra-names").apply { isDaemon = true } }
  }

  /** Photo files read and hashed so far (the cache's misses). */
  var photosHashed = 0
    private set

  init {
    val f = File(root, ExtraNames.FILE_NAME)
    var parsed: NamesFile? = null
    var error: String? = null
    if (!f.exists()) {
      error = "missing"
    } else {
      try {
        parsed = ExtraNames.parse(f.readText())
      } catch (e: ExtraNames.Unusable) {
        error = e.reason
      } catch (e: Exception) {
        error = "unreadable: $e"
      }
    }
    file = parsed
    loadError = error
  }

  /**
   * The bound names of each of [refs] that the file lists and whose current photo matches, by
   * [ExtraNames.key]. Empty without a file. Reads the photos not hashed yet.
   */
  @Synchronized
  fun bind(refs: List<PhotoRef>): Map<String, NameSet> {
    val names = file ?: return emptyMap()
    val out = HashMap<String, NameSet>()
    for (r in refs) {
      val item = names.item(r.kind, r.id) ?: continue
      ExtraNames.bind(item, photoSha256(File(root, r.cropFile)))?.let { out[ExtraNames.key(r.kind, r.id)] = it }
    }
    return out
  }

  /** [bind] on a background thread; [done] runs on that thread. */
  fun bindAsync(refs: List<PhotoRef>, done: (Map<String, NameSet>) -> Unit) {
    if (file == null) return done(emptyMap())
    worker.execute { done(bind(refs)) }
  }

  /** Lower-case hex SHA-256 of [f]'s bytes, or null when it is not a readable file. */
  @Synchronized
  fun photoSha256(f: File): String? {
    if (!f.isFile) return null
    val before = Stamp(f.length(), f.lastModified())
    hashes[f.path]?.let { (stamp, hex) -> if (stamp == before) return hex }
    val hex =
      try {
        val md = MessageDigest.getInstance("SHA-256")
        f.inputStream().use { input ->
          val buf = ByteArray(64 * 1024)
          while (true) {
            val n = input.read(buf)
            if (n < 0) break
            md.update(buf, 0, n)
          }
        }
        ExtraNames.hex(md.digest())
      } catch (e: Exception) {
        return null
      }
    photosHashed++
    // A photo replaced while it was read is hashed again next time.
    if (Stamp(f.length(), f.lastModified()) == before) hashes[f.path] = before to hex
    return hex
  }
}
