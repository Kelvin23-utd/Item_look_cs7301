package edu.cs6301.findback.memory

import org.json.JSONArray
import org.json.JSONObject

/**
 * One user-confirmed object memory. Raw model output ([rawLabel], [rawConfidence], [boxSource]) is
 * kept separate from the user's correction ([name]).
 *
 * Records are a snapshot of the registration moment: verification never rewrites them, so the three
 * study conditions all query identical memories. The only way an item's memory changes is "Update
 * location" (v0.4), which replaces the record with a new one of the same lineage (see [Lineage]).
 */
data class ObjectRecord(
  val id: String,
  val name: String,
  val rawLabel: String,
  val rawConfidence: Float,
  /** "detector" when an ML Kit box contained the tap, "tap_crop" for the fixed-size fallback. */
  val boxSource: String,
  /** Crop image path relative to the app's files directory. */
  val cropFile: String,
  val embedding: FloatArray,
  /**
   * v0.2: position in the reference image's coordinate frame (metres), or null if it wasn't tracked.
   * v0.3 still reads (and re-saves) it for old records but no longer writes or uses it: new records
   * have null here and a [mapPosition] instead.
   */
  val refPosition: FloatArray?,
  /** Position in the capture session's world frame (only meaningful within [sessionId]). */
  val worldPosition: FloatArray,
  val sessionId: String,
  /** Camera-to-object distance at capture (metres), used to rescale verification crops. */
  val captureDistance: Float,
  val cropWidthPx: Float,
  val cropHeightPx: Float,
  val imageWidth: Int,
  val imageHeight: Int,
  /** ARCore trackable type the position came from: Plane, DepthPoint or Point. */
  val hitType: String,
  val lastSeenMs: Long,
  // v0.3 home map. All nullable and absent in older files; null when registered while not located.
  /** Id of the home map [mapPosition] and [viewPosition] belong to. */
  val mapId: String? = null,
  /** Item position in the map frame M (metres). */
  val mapPosition: FloatArray? = null,
  /** Camera position in M at capture: a walkable point next to the item, the route's end. */
  val viewPosition: FloatArray? = null,
  /** Room picked at registration (an id in the map's room list), or null for "No room". */
  val roomId: String? = null,
  // v0.4 study fields. All absent in older files, which read as an unmoved item of unknown mobility.
  /** How often the item moves, asked at registration: "rare", "some" or "often" (see Mobility). */
  val mobility: String? = null,
  /**
   * Id shared by every record of the same physical item across "Update location": the id of the
   * item's first record. An ordinary record is its own lineage.
   */
  val lineageId: String = id,
  /** Id of the record this one replaced through "Update location", or null. */
  val supersedes: String? = null,
  /** How many times the item's location was updated before this record (0 for a first record). */
  val moveCount: Int = 0,
  /** When the lineage's first record was registered (epoch ms); [lastSeenMs] for a first record. */
  val lineageStartMs: Long = lastSeenMs,
) {
  fun toJson(): JSONObject =
    JSONObject().apply {
      put("id", id)
      put("name", name)
      put("rawLabel", rawLabel)
      put("rawConfidence", rawConfidence.toDouble())
      put("boxSource", boxSource)
      put("cropFile", cropFile)
      put("embedding", floats(embedding))
      put("refPosition", refPosition?.let { floats(it) } ?: JSONObject.NULL)
      put("worldPosition", floats(worldPosition))
      put("sessionId", sessionId)
      put("captureDistance", captureDistance.toDouble())
      put("cropWidthPx", cropWidthPx.toDouble())
      put("cropHeightPx", cropHeightPx.toDouble())
      put("imageWidth", imageWidth)
      put("imageHeight", imageHeight)
      put("hitType", hitType)
      put("lastSeenMs", lastSeenMs)
      put("mapId", mapId ?: JSONObject.NULL)
      put("mapPosition", mapPosition?.let { floats(it) } ?: JSONObject.NULL)
      put("viewPosition", viewPosition?.let { floats(it) } ?: JSONObject.NULL)
      put("roomId", roomId ?: JSONObject.NULL)
      put("mobility", mobility ?: JSONObject.NULL)
      put("lineageId", lineageId)
      put("supersedes", supersedes ?: JSONObject.NULL)
      put("moveCount", moveCount)
      put("lineageStartMs", lineageStartMs)
    }

  companion object {
    fun fromJson(o: JSONObject): ObjectRecord {
      val id = o.getString("id")
      val lastSeenMs = o.getLong("lastSeenMs")
      return ObjectRecord(
        id = id,
        name = o.getString("name"),
        rawLabel = o.optString("rawLabel", ""),
        rawConfidence = o.optDouble("rawConfidence", 0.0).toFloat(),
        boxSource = o.optString("boxSource", ""),
        cropFile = o.getString("cropFile"),
        embedding = toFloats(o.getJSONArray("embedding")),
        refPosition = if (o.isNull("refPosition")) null else toFloats(o.getJSONArray("refPosition")),
        worldPosition = toFloats(o.getJSONArray("worldPosition")),
        sessionId = o.getString("sessionId"),
        captureDistance = o.getDouble("captureDistance").toFloat(),
        cropWidthPx = o.getDouble("cropWidthPx").toFloat(),
        cropHeightPx = o.getDouble("cropHeightPx").toFloat(),
        imageWidth = o.getInt("imageWidth"),
        imageHeight = o.getInt("imageHeight"),
        hitType = o.optString("hitType", ""),
        lastSeenMs = lastSeenMs,
        mapId = if (o.isNull("mapId")) null else o.getString("mapId"),
        mapPosition = if (o.isNull("mapPosition")) null else toFloats(o.getJSONArray("mapPosition")),
        viewPosition = if (o.isNull("viewPosition")) null else toFloats(o.getJSONArray("viewPosition")),
        roomId = if (o.isNull("roomId")) null else o.getString("roomId"),
        // v0.4: a file without these keys (or with nulls) reads as a first record of its own lineage.
        mobility = if (o.isNull("mobility")) null else o.getString("mobility"),
        lineageId = if (o.isNull("lineageId")) id else o.getString("lineageId").ifEmpty { id },
        supersedes = if (o.isNull("supersedes")) null else o.getString("supersedes"),
        moveCount = if (o.isNull("moveCount")) 0 else o.getInt("moveCount"),
        lineageStartMs = if (o.isNull("lineageStartMs")) lastSeenMs else o.getLong("lineageStartMs"),
      )
    }

    fun floats(a: FloatArray) = JSONArray().apply { a.forEach { put(it.toDouble()) } }

    fun toFloats(a: JSONArray) = FloatArray(a.length()) { a.getDouble(it).toFloat() }
  }
}

/** The lineage fields of a new record (spec v0.4, S7). */
data class LineageFields(val lineageId: String, val supersedes: String?, val moveCount: Int, val lineageStartMs: Long)

/**
 * "Update location": the user found an item somewhere else and registered it there. The new record
 * replaces the old one in the active list and continues its lineage. Pure functions, unit-tested.
 */
object Lineage {
  /** An ordinary registration: the record starts its own lineage. */
  fun first(id: String, registeredMs: Long) = LineageFields(id, null, 0, registeredMs)

  /** The record that replaces [old]: same lineage, one more move, same first-registration time. */
  fun successor(old: ObjectRecord) =
    LineageFields(old.lineageId, old.id, old.moveCount + 1, old.lineageStartMs)

  /** Minutes since the lineage's first registration (`query_start.lineageAgeMin`). */
  fun ageMinutes(rec: ObjectRecord, nowMs: Long): Double = (nowMs - rec.lineageStartMs) / 60000.0
}
