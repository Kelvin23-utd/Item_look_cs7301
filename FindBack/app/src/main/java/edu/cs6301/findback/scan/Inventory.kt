package edu.cs6301.findback.scan

import edu.cs6301.findback.map.Vec3
import edu.cs6301.findback.memory.EmbeddingFile
import edu.cs6301.findback.memory.ObjectRecord
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import org.json.JSONArray
import org.json.JSONObject

/** One sighting of an entry by a scan keyframe: where the detection was placed in the map frame. */
data class Sighting(
  val tMs: Long,
  val position: Vec3,
  val sizeM: Double,
  val label: String,
  val score: Float,
  val scanId: String,
  val keyframe: Int,
)

/**
 * What the app knows about whether an entry is still where it was last seen (plan section 7). Each
 * scan, each one-minute passive window and each confirmed verification gives one outcome. The time
 * of the last `seen` is the entry's own [InventoryEntry.lastSeenMs].
 */
data class EvidenceState(
  val seenCount: Int = 0,
  val missedCount: Int = 0,
  val notObservedCount: Int = 0,
  val lastMissedMs: Long? = null,
  val lastNotObservedMs: Long? = null,
  /** "seen", "missed" or "not_observed"; null until the entry's first scan has ended. */
  val lastOutcome: String? = null,
  val lastOutcomeMs: Long? = null,
  /** "scan", "passive" or "verify". */
  val lastSource: String? = null,
  /** The scan id, passive window id or query id of the last outcome. */
  val lastSourceId: String? = null,
)

/** "May be <name>, last seen in <room>": an older entry that was looked for and not seen. */
data class MovedFrom(val entryId: String, val name: String, val roomId: String?, val similarity: Float, val scanId: String)

/**
 * v0.8: where the item was last known to be for certain. Set at the end of the scan that created
 * the entry (`source` "created"); afterwards only an explicit confirmation by the user changes it
 * ("user_verify"). A later scan's merge never does. Map frame, metres; times epoch ms.
 */
data class ConfirmedState(
  val mapPosition: Vec3,
  /** Where the camera stood (map frame), or null when unknown. */
  val viewPosition: Vec3?,
  val roomId: String?,
  /** Time of the latest observation behind this state. */
  val tMs: Long,
  /** The creating scan; null for "user_verify". */
  val scan: String?,
  /** "created" or "user_verify". */
  val source: String,
  /** Observation ids of the reference views (`obs/`). */
  val refs: List<String>,
  /** "user_verify": the query and the verification similarity of the capture. */
  val query: String? = null,
  val similarity: Float? = null,
)

/**
 * v0.8: what one scan said about an entry, from that scan's own sightings only. One item per scan
 * that gave the entry an outcome.
 */
data class HistoryItem(
  val scan: String,
  /** Scan end, epoch ms. */
  val tMs: Long,
  /** "seen", "missed" or "not_observed", as the scan's evidence outcome. */
  val outcome: String,
  /** The entry took part in the scan (its map and target). */
  val eligible: Boolean,
  /** Mean map point of the detections the matcher linked to the entry in this scan, or null. */
  val position: Vec3?,
  /** Number of those detections. */
  val sightings: Int,
  /** Capture time of the latest of them, or null. */
  val lastObsMs: Long?,
  /** Their observation ids. */
  val obs: List<String>,
  /** "created" (this scan created the entry) or "baseline_unconfirmed" (a later scan's matcher). */
  val link: String,
)

/**
 * One entry of the scan inventory (inventory.json): an item found by scan mode, merged from
 * repeated sightings. Immutable; every change makes a copy.
 */
data class InventoryEntry(
  val id: String,
  /** Display name: the winning label until the user renames the entry. */
  val name: String,
  val nameEdited: Boolean,
  /** The label with the highest summed score. */
  val label: String,
  /** Label to summed detection score, over all sightings. */
  val labelVotes: Map<String, Double>,
  val mapId: String,
  /** Item position in the map frame: the mean of the latest sightings. */
  val position: Vec3,
  /** Where the camera stood at the best sighting (map frame): the end of the route. */
  val viewPosition: Vec3,
  val roomId: String?,
  /** Name of the container the entry was scanned in ("desk drawer"), or null for the open area. */
  val container: String?,
  /** Physical size (metres): the longer box side of the best sighting. */
  val sizeM: Double,
  // The best crop and everything needed to build an ObjectRecord for Find.
  val cropFile: String,
  val embedding: FloatArray,
  val embeddingModel: String,
  val captureDistance: Float,
  val cropWidthPx: Float,
  val cropHeightPx: Float,
  val imageWidth: Int,
  val imageHeight: Int,
  /** Quality of the best crop: box area (px²) × score. */
  val bestQuality: Float,
  val bestScore: Float,
  /** Item position in the world frame of the session of the best sighting. */
  val worldPosition: Vec3,
  val sessionId: String,
  /** The latest sightings (at most [ScanConfig.POSITION_SIGHTINGS]); [sightingCount] counts all. */
  val sightings: List<Sighting>,
  val sightingCount: Int,
  val firstSeenMs: Long,
  /** Last time a detection, an appearance match or a confirmed verification saw the item. */
  val lastSeenMs: Long,
  /** Detector that created the entry. */
  val detector: String,
  /** Scan that created the entry. */
  val createdScan: String,
  val evidence: EvidenceState = EvidenceState(),
  val movedFrom: MovedFrom? = null,
  /** v0.8 study logging; null until set (entries from before v0.8 or made with study logging off). */
  val confirmed: ConfirmedState? = null,
  /** v0.8 study logging: one item per scan, oldest first. */
  val history: List<HistoryItem> = emptyList(),
) {
  /** The entry after one more evidence outcome at [atMs]. */
  fun withOutcome(outcome: String, atMs: Long, source: String, sourceId: String?): InventoryEntry {
    val ev = evidence
    val next =
      when (outcome) {
        ScanConfig.OUTCOME_SEEN -> ev.copy(seenCount = ev.seenCount + 1)
        ScanConfig.OUTCOME_MISSED -> ev.copy(missedCount = ev.missedCount + 1, lastMissedMs = atMs)
        else -> ev.copy(notObservedCount = ev.notObservedCount + 1, lastNotObservedMs = atMs)
      }.copy(lastOutcome = outcome, lastOutcomeMs = atMs, lastSource = source, lastSourceId = sourceId)
    return copy(
      evidence = next,
      lastSeenMs = if (outcome == ScanConfig.OUTCOME_SEEN && atMs > lastSeenMs) atMs else lastSeenMs,
    )
  }

  /**
   * The record the existing query flow (guidance, verification) works on. The entry's best crop
   * plays the part of the registration crop; the position source is always the map.
   */
  fun toObjectRecord(): ObjectRecord =
    ObjectRecord(
      id = id,
      name = name,
      rawLabel = label,
      rawConfidence = bestScore,
      boxSource = BOX_SOURCE,
      cropFile = cropFile,
      embedding = embedding,
      refPosition = null,
      worldPosition = worldPosition.toFloatArray(),
      sessionId = sessionId,
      captureDistance = captureDistance,
      cropWidthPx = cropWidthPx,
      cropHeightPx = cropHeightPx,
      imageWidth = imageWidth,
      imageHeight = imageHeight,
      hitType = HIT_TYPE,
      lastSeenMs = lastSeenMs,
      mapId = mapId,
      mapPosition = position.toFloatArray(),
      viewPosition = viewPosition.toFloatArray(),
      roomId = roomId,
      lineageStartMs = firstSeenMs,
    )

  /** inventory.json form. [withEmbedding] false gives the compact form used in events.jsonl. */
  fun toJson(withEmbedding: Boolean = true): JSONObject =
    JSONObject().apply {
      put("id", id)
      put("name", name)
      put("nameEdited", nameEdited)
      put("label", label)
      put("labelVotes", JSONObject().apply { for ((k, v) in labelVotes) put(k, v) })
      put("mapId", mapId)
      put("mapPosition", vec(position))
      put("viewPosition", vec(viewPosition))
      put("roomId", roomId ?: JSONObject.NULL)
      put("container", container ?: JSONObject.NULL)
      put("sizeM", sizeM)
      put("cropFile", cropFile)
      if (withEmbedding) put("embedding", EmbeddingFile.encode(embedding))
      put("embeddingDim", embedding.size)
      put("embeddingEncoding", EmbeddingFile.ENCODING)
      put("embeddingModel", embeddingModel)
      put("captureDistance", captureDistance.toDouble())
      put("cropWidthPx", cropWidthPx.toDouble())
      put("cropHeightPx", cropHeightPx.toDouble())
      put("imageWidth", imageWidth)
      put("imageHeight", imageHeight)
      put("bestQuality", bestQuality.toDouble())
      put("bestScore", bestScore.toDouble())
      put("worldPosition", vec(worldPosition))
      put("sessionId", sessionId)
      put(
        "sightings",
        JSONArray().apply {
          for (s in sightings) {
            put(
              JSONObject().apply {
                put("t", s.tMs)
                put("p", vec(s.position))
                put("sizeM", s.sizeM)
                put("label", s.label)
                put("score", s.score.toDouble())
                put("scan", s.scanId)
                put("kf", s.keyframe)
              }
            )
          }
        },
      )
      put("sightingCount", sightingCount)
      put("firstSeenMs", firstSeenMs)
      put("lastSeenMs", lastSeenMs)
      put("detector", detector)
      put("createdScan", createdScan)
      put("evidence", evidenceJson())
      put(
        "movedFrom",
        movedFrom?.let { m ->
          JSONObject().apply {
            put("entry", m.entryId)
            put("name", m.name)
            put("roomId", m.roomId ?: JSONObject.NULL)
            put("similarity", m.similarity.toDouble())
            put("scan", m.scanId)
          }
        } ?: JSONObject.NULL,
      )
      // v0.8: written only when set, so an entry without them is written exactly as before.
      confirmed?.let { put("confirmed", confirmedJson(it)) }
      if (history.isNotEmpty()) put("history", JSONArray().apply { history.forEach { put(historyJson(it)) } })
    }

  /** The evidence fields as written in inventory.json and in `evidence` lines of the log. */
  fun evidenceJson(): JSONObject =
    JSONObject().apply {
      put("seen", evidence.seenCount)
      put("missed", evidence.missedCount)
      put("notObserved", evidence.notObservedCount)
      put("lastSeenMs", lastSeenMs) // the entry's own lastSeenMs, repeated for the reader
      put("lastMissedMs", evidence.lastMissedMs ?: JSONObject.NULL)
      put("lastNotObservedMs", evidence.lastNotObservedMs ?: JSONObject.NULL)
      put("lastOutcome", evidence.lastOutcome ?: JSONObject.NULL)
      put("lastOutcomeMs", evidence.lastOutcomeMs ?: JSONObject.NULL)
      put("lastSource", evidence.lastSource ?: JSONObject.NULL)
      put("lastSourceId", evidence.lastSourceId ?: JSONObject.NULL)
    }

  companion object {
    /** `ObjectRecord.boxSource` and `hitType` of a record built from a scan entry. */
    const val BOX_SOURCE = "scan"
    const val HIT_TYPE = "ScanDepth"

    /** Path of an entry's photo, relative to the files root. */
    fun cropPath(id: String) = "inventory/$id.jpg"

    fun vec(v: Vec3): JSONArray = JSONArray().put(v.x).put(v.y).put(v.z)

    fun toVec(a: JSONArray): Vec3 = Vec3(a.getDouble(0), a.getDouble(1), a.getDouble(2))

    private fun optVec(o: JSONObject, key: String): Vec3? = if (o.isNull(key)) null else toVec(o.getJSONArray(key))

    private fun strings(a: JSONArray?): List<String> = if (a == null) emptyList() else (0 until a.length()).map { a.getString(it) }

    fun confirmedJson(c: ConfirmedState): JSONObject =
      JSONObject().apply {
        put("mapPosition", vec(c.mapPosition))
        put("viewPosition", c.viewPosition?.let { vec(it) } ?: JSONObject.NULL)
        put("roomId", c.roomId ?: JSONObject.NULL)
        put("tMs", c.tMs)
        put("scan", c.scan ?: JSONObject.NULL)
        put("source", c.source)
        put("refs", JSONArray(c.refs))
        put("query", c.query ?: JSONObject.NULL)
        put("similarity", c.similarity?.toDouble() ?: JSONObject.NULL)
      }

    fun confirmedFromJson(o: JSONObject): ConfirmedState =
      ConfirmedState(
        mapPosition = toVec(o.getJSONArray("mapPosition")),
        viewPosition = optVec(o, "viewPosition"),
        roomId = optString(o, "roomId"),
        tMs = o.getLong("tMs"),
        scan = optString(o, "scan"),
        source = o.getString("source"),
        refs = strings(o.optJSONArray("refs")),
        query = optString(o, "query"),
        similarity = if (o.isNull("similarity")) null else o.getDouble("similarity").toFloat(),
      )

    fun historyJson(h: HistoryItem): JSONObject =
      JSONObject().apply {
        put("scan", h.scan)
        put("tMs", h.tMs)
        put("outcome", h.outcome)
        put("eligible", h.eligible)
        put("position", h.position?.let { vec(it) } ?: JSONObject.NULL)
        put("sightings", h.sightings)
        put("lastObsMs", h.lastObsMs ?: JSONObject.NULL)
        put("obs", JSONArray(h.obs))
        put("link", h.link)
      }

    fun historyFromJson(o: JSONObject): HistoryItem =
      HistoryItem(
        scan = o.getString("scan"),
        tMs = o.getLong("tMs"),
        outcome = o.getString("outcome"),
        eligible = o.optBoolean("eligible", true),
        position = optVec(o, "position"),
        sightings = o.optInt("sightings", 0),
        lastObsMs = optLong(o, "lastObsMs"),
        obs = strings(o.optJSONArray("obs")),
        link = o.getString("link"),
      )

    fun fromJson(o: JSONObject): InventoryEntry {
      val votes = LinkedHashMap<String, Double>()
      o.optJSONObject("labelVotes")?.let { v -> for (k in v.keys()) votes[k] = v.getDouble(k) }
      val sightings = ArrayList<Sighting>()
      o.optJSONArray("sightings")?.let { a ->
        for (i in 0 until a.length()) {
          val s = a.getJSONObject(i)
          sightings.add(
            Sighting(
              tMs = s.getLong("t"),
              position = toVec(s.getJSONArray("p")),
              sizeM = s.optDouble("sizeM", 0.0),
              label = s.optString("label", ""),
              score = s.optDouble("score", 0.0).toFloat(),
              scanId = s.optString("scan", ""),
              keyframe = s.optInt("kf", 0),
            )
          )
        }
      }
      val ev = o.optJSONObject("evidence") ?: JSONObject()
      val moved = if (o.isNull("movedFrom")) null else o.getJSONObject("movedFrom")
      // v0.8 fields: absent in older files.
      val confirmed = if (o.isNull("confirmed")) null else confirmedFromJson(o.getJSONObject("confirmed"))
      val history = o.optJSONArray("history")?.let { a -> (0 until a.length()).map { historyFromJson(a.getJSONObject(it)) } } ?: emptyList()
      val lastSeenMs = o.getLong("lastSeenMs")
      return InventoryEntry(
        id = o.getString("id"),
        name = o.getString("name"),
        nameEdited = o.optBoolean("nameEdited", false),
        label = o.optString("label", ""),
        labelVotes = votes,
        mapId = o.getString("mapId"),
        position = toVec(o.getJSONArray("mapPosition")),
        viewPosition = toVec(o.getJSONArray("viewPosition")),
        roomId = optString(o, "roomId"),
        container = optString(o, "container"),
        sizeM = o.optDouble("sizeM", 0.0),
        cropFile = o.getString("cropFile"),
        embedding = EmbeddingFile.decode(o.getString("embedding")),
        embeddingModel = o.optString("embeddingModel", ""),
        captureDistance = o.getDouble("captureDistance").toFloat(),
        cropWidthPx = o.getDouble("cropWidthPx").toFloat(),
        cropHeightPx = o.getDouble("cropHeightPx").toFloat(),
        imageWidth = o.getInt("imageWidth"),
        imageHeight = o.getInt("imageHeight"),
        bestQuality = o.optDouble("bestQuality", 0.0).toFloat(),
        bestScore = o.optDouble("bestScore", 0.0).toFloat(),
        worldPosition = if (o.isNull("worldPosition")) Vec3.ZERO else toVec(o.getJSONArray("worldPosition")),
        sessionId = o.optString("sessionId", ""),
        sightings = sightings,
        sightingCount = o.optInt("sightingCount", sightings.size),
        firstSeenMs = o.optLong("firstSeenMs", lastSeenMs),
        lastSeenMs = lastSeenMs,
        detector = o.optString("detector", ""),
        createdScan = o.optString("createdScan", ""),
        evidence =
          EvidenceState(
            seenCount = ev.optInt("seen", 0),
            missedCount = ev.optInt("missed", 0),
            notObservedCount = ev.optInt("notObserved", 0),
            lastMissedMs = optLong(ev, "lastMissedMs"),
            lastNotObservedMs = optLong(ev, "lastNotObservedMs"),
            lastOutcome = optString(ev, "lastOutcome"),
            lastOutcomeMs = optLong(ev, "lastOutcomeMs"),
            lastSource = optString(ev, "lastSource"),
            lastSourceId = optString(ev, "lastSourceId"),
          ),
        movedFrom =
          moved?.let { m ->
            MovedFrom(
              entryId = m.getString("entry"),
              name = m.optString("name", ""),
              roomId = optString(m, "roomId"),
              similarity = m.optDouble("similarity", 0.0).toFloat(),
              scanId = m.optString("scan", ""),
            )
          },
        confirmed = confirmed,
        history = history,
      )
    }

    private fun optString(o: JSONObject, key: String): String? = if (o.isNull(key)) null else o.getString(key)

    private fun optLong(o: JSONObject, key: String): Long? = if (o.isNull(key)) null else o.getLong(key)
  }
}

/** The words the Items list shows for a scanned entry, and its search. Pure, unit-tested. */
object InventoryText {
  /** "5 min ago", in the same steps as the rest of the app. */
  fun ago(ms: Long, nowMs: Long): String {
    val s = ((nowMs - ms) / 1000).coerceAtLeast(0)
    return when {
      s < 60 -> "${s}s ago"
      s < 3600 -> "${s / 60} min ago"
      s < 86400 -> "${s / 3600} h ago"
      else -> "${s / 86400} d ago"
    }
  }

  /** "Nov 17". */
  fun day(ms: Long): String = SimpleDateFormat("MMM d", Locale.US).format(Date(ms))

  /** "Bedroom", or "in desk drawer, Office" for an entry scanned in a container. */
  fun location(entry: InventoryEntry, roomName: String?, inCurrentMap: Boolean): String {
    if (!inCurrentMap) return "another map (scanned before this map)"
    val room = roomName ?: "room not named"
    return if (entry.container != null) "in ${entry.container}, $room" else room
  }

  /**
   * "seen 2 d ago", "looked for in the last scan, not seen" or "not checked since Nov 17".
   * [lastScanId] is the most recent finished scan.
   */
  fun evidence(entry: InventoryEntry, nowMs: Long, lastScanId: String?): String {
    val ev = entry.evidence
    return when (ev.lastOutcome) {
      ScanConfig.OUTCOME_MISSED ->
        if (ev.lastSource == ScanConfig.SOURCE_SCAN && ev.lastSourceId != null && ev.lastSourceId == lastScanId)
          "looked for in the last scan, not seen"
        else "looked for ${ago(ev.lastMissedMs ?: ev.lastOutcomeMs ?: nowMs, nowMs)}, not seen"
      ScanConfig.OUTCOME_NOT_OBSERVED -> {
        val missed = ev.lastMissedMs
        if (missed != null && missed > entry.lastSeenMs) "not seen on ${day(missed)}, not checked since"
        else "not checked since ${day(entry.lastSeenMs)}"
      }
      else -> "seen ${ago(entry.lastSeenMs, nowMs)}"
    }
  }

  /** "may be mug, last seen in Kitchen", or null. */
  fun movedHint(entry: InventoryEntry, roomName: (String?) -> String?): String? {
    val m = entry.movedFrom ?: return null
    val room = roomName(m.roomId)
    return "may be ${m.name}" + if (room != null) ", last seen in $room" else ""
  }

  /** Case-insensitive search over the name, every label the entry was detected as, and the container. */
  fun matches(entry: InventoryEntry, query: String): Boolean {
    val q = query.trim().lowercase(Locale.ROOT)
    if (q.isEmpty()) return true
    if (entry.name.lowercase(Locale.ROOT).contains(q)) return true
    if (entry.label.lowercase(Locale.ROOT).contains(q)) return true
    if (entry.labelVotes.keys.any { it.lowercase(Locale.ROOT).contains(q) }) return true
    return entry.container?.lowercase(Locale.ROOT)?.contains(q) == true
  }

  /** The same search for a tapped record: its name and the detector's raw label. */
  fun matches(record: ObjectRecord, query: String): Boolean {
    val q = query.trim().lowercase(Locale.ROOT)
    if (q.isEmpty()) return true
    return record.name.lowercase(Locale.ROOT).contains(q) || record.rawLabel.lowercase(Locale.ROOT).contains(q)
  }

  /** Two container names are the same container when they differ only in case and outer spaces. */
  fun sameContainer(a: String?, b: String?): Boolean {
    if (a == null || b == null) return a == null && b == null
    return a.trim().equals(b.trim(), ignoreCase = true)
  }
}
