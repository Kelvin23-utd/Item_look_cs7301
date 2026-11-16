package edu.cs6301.findback.map

import java.io.File
import java.util.concurrent.Executors
import java.util.concurrent.ScheduledExecutorService
import java.util.concurrent.TimeUnit
import org.json.JSONArray
import org.json.JSONObject

/** map.json encoding (format in docs/PLAN_v0.3_home_map.md, section 5). */
object HomeMapJson {
  fun toJson(m: HomeMap): JSONObject =
    JSONObject().apply {
      put("version", m.version)
      put("mapId", m.mapId)
      put("createdMs", m.createdMs)
      put("fixPoints", JSONArray().apply { m.fixPoints.forEach { put(fixToJson(it)) } })
      put("rooms", JSONArray().apply { m.rooms.forEach { put(roomToJson(it)) } })
      put("nodes", JSONArray().apply { m.nodes.forEach { put(nodeToJson(it)) } })
      put("edges", JSONArray().apply { m.edges.forEach { (a, b) -> put(JSONArray().put(a).put(b)) } })
    }

  fun fromJson(o: JSONObject): HomeMap {
    val fixes = o.optJSONArray("fixPoints") ?: JSONArray()
    val rooms = o.optJSONArray("rooms") ?: JSONArray()
    val nodes = o.optJSONArray("nodes") ?: JSONArray()
    val edges = o.optJSONArray("edges") ?: JSONArray()
    return HomeMap(
      version = o.optInt("version", HomeMap.VERSION),
      mapId = o.getString("mapId"),
      createdMs = o.optLong("createdMs", 0L),
      fixPoints = (0 until fixes.length()).map { fixFromJson(fixes.getJSONObject(it)) },
      rooms = (0 until rooms.length()).map { roomFromJson(rooms.getJSONObject(it)) },
      nodes = (0 until nodes.length()).map { nodeFromJson(nodes.getJSONObject(it)) },
      edges = (0 until edges.length()).map { edges.getJSONArray(it).let { e -> e.getInt(0) to e.getInt(1) } },
    )
  }

  fun fixToJson(f: FixPoint): JSONObject =
    JSONObject().apply {
      put("id", f.id)
      put("name", f.name)
      put("type", f.type.name)
      put("x", f.x)
      put("y", f.y)
      put("z", f.z)
      put("yaw", f.yaw)
      put("yawAxis", f.yawAxis ?: JSONObject.NULL)
      put("widthM", f.widthM ?: JSONObject.NULL)
      put("ghostFile", f.ghostFile ?: JSONObject.NULL)
      put("createdMs", f.createdMs)
      put("sessionId", f.sessionId)
    }

  fun fixFromJson(o: JSONObject) =
    FixPoint(
      id = o.getString("id"),
      name = o.optString("name", ""),
      type = FixType.valueOf(o.getString("type")),
      x = o.getDouble("x"),
      y = o.getDouble("y"),
      z = o.getDouble("z"),
      yaw = o.getDouble("yaw"),
      yawAxis = optStringOrNull(o, "yawAxis"),
      widthM = if (o.isNull("widthM")) null else o.getDouble("widthM"),
      ghostFile = optStringOrNull(o, "ghostFile"),
      createdMs = o.optLong("createdMs", 0L),
      sessionId = o.optString("sessionId", ""),
    )

  private fun roomToJson(r: Room) =
    JSONObject().apply {
      put("id", r.id)
      put("name", r.name)
      put("createdMs", r.createdMs)
    }

  private fun roomFromJson(o: JSONObject) = Room(o.getString("id"), o.optString("name", ""), o.optLong("createdMs", 0L))

  private fun nodeToJson(n: MapNode) =
    JSONObject().apply {
      put("id", n.id)
      put("x", n.x)
      put("y", n.y)
      put("z", n.z)
      put("roomId", n.roomId ?: JSONObject.NULL)
      put("sessionId", n.sessionId)
      put("tMs", n.tMs)
    }

  private fun nodeFromJson(o: JSONObject) =
    MapNode(
      id = o.getInt("id"),
      x = o.getDouble("x"),
      y = o.getDouble("y"),
      z = o.getDouble("z"),
      roomId = optStringOrNull(o, "roomId"),
      sessionId = o.optString("sessionId", ""),
      tMs = o.optLong("tMs", 0L),
    )

  private fun optStringOrNull(o: JSONObject, key: String): String? = if (o.isNull(key)) null else o.getString(key)
}

/**
 * Persistent home map (map.json in the app's files root). Follows MemoryStore's pattern: state is
 * synchronized and a save writes a temp file, then renames it. The GL thread [publish]es
 * immutable snapshots; they reach the disk at most every [SAVE_INTERVAL_MS] while dirty, and on
 * [flush] (called on pause). A corrupt file is copied aside, never deleted.
 */
class MapStore(
  val root: File,
  private val onSaved: (HomeMap) -> Unit = {},
  private val onError: (String, Throwable) -> Unit = { _, _ -> },
) {
  companion object {
    const val FILE_NAME = "map.json"
    /** Autosave interval while the map has unsaved changes. */
    const val SAVE_INTERVAL_MS = 5000L
  }

  private val file = File(root, FILE_NAME)
  private var latest: HomeMap? = null
  private var dirty = false
  private var closed = false
  private var autosave: ScheduledExecutorService? = null
  /** Held while writing or archiving the file; the snapshot fields use the object's own lock. */
  private val writeLock = Any()

  /** Read map.json; null when absent or corrupt (a corrupt file is copied aside first). */
  @Synchronized
  fun load(): HomeMap? {
    if (!file.exists()) return null
    return try {
      HomeMapJson.fromJson(JSONObject(file.readText())).also { latest = it }
    } catch (e: Exception) {
      onError("Failed to load map.json; starting without a map", e)
      try {
        file.copyTo(File(root, "map.corrupt.${System.currentTimeMillis()}.json"))
      } catch (e2: Exception) {
        onError("Could not copy the corrupt map aside", e2)
      }
      null
    }
  }

  /** Latest snapshot from the GL thread; written by the autosave or [flush]. */
  @Synchronized
  fun publish(map: HomeMap) {
    if (closed) return
    latest = map
    dirty = true
  }

  @Synchronized fun isDirty() = dirty

  /**
   * Save now if there are unsaved changes. Returns true when a file was written. Serialising and
   * writing happen under [writeLock] only, so the GL thread's [publish] never waits for the disk.
   */
  fun flush(): Boolean = synchronized(writeLock) { flushLocked() }

  /** Caller holds [writeLock]. */
  private fun flushLocked(): Boolean {
    val m: HomeMap
    synchronized(this) {
      val l = latest
      if (closed || !dirty || l == null) return false
      m = l
      dirty = false
    }
    return try {
      val tmp = File(root, "$FILE_NAME.tmp")
      tmp.writeText(HomeMapJson.toJson(m).toString(1))
      if (!tmp.renameTo(file)) {
        tmp.copyTo(file, overwrite = true)
        tmp.delete()
      }
      onSaved(m)
      true
    } catch (e: Exception) {
      synchronized(this) { if (latest === m) dirty = true }
      onError("Failed to save map.json", e)
      false
    }
  }

  fun startAutosave() {
    synchronized(this) {
      if (autosave != null || closed) return
      autosave = Executors.newSingleThreadScheduledExecutor()
    }
    autosave?.scheduleWithFixedDelay({ flush() }, SAVE_INTERVAL_MS, SAVE_INTERVAL_MS, TimeUnit.MILLISECONDS)
  }

  /** Flush and stop the autosave (activity destroyed). */
  fun close() {
    flush()
    synchronized(this) {
      autosave?.shutdown()
      autosave = null
    }
  }

  /**
   * "New map": save the latest state, move map.json into [dir], and stop writing. Nothing is
   * deleted. Returns false when there was no map file to move.
   */
  fun archiveTo(dir: File): Boolean = synchronized(writeLock) {
    flushLocked()
    synchronized(this) {
      closed = true
      autosave?.shutdown()
      autosave = null
    }
    if (!file.exists()) return false
    dir.mkdirs()
    val dest = File(dir, FILE_NAME)
    if (file.renameTo(dest)) return true
    file.copyTo(dest, overwrite = false)
    return file.delete()
  }
}
