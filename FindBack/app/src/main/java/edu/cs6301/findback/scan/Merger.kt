package edu.cs6301.findback.scan

import edu.cs6301.findback.map.Vec3
import edu.cs6301.findback.map.distance
import kotlin.math.max
import kotlin.math.sqrt

/** What a keyframe contributes to every sighting made in it. */
class KeyframeInfo(
  val index: Int,
  /** Capture time, epoch ms. */
  val tMs: Long,
  /** The user's room at capture, or null. */
  val roomId: String?,
  /** Camera position in the map frame. */
  val cameraMap: Vec3,
  val imageWidth: Int,
  val imageHeight: Int,
)

/** The entry a detection merges into, with how it qualified. */
class Match(val entry: InventoryEntry, val distanceM: Double, val similarity: Float, val byLabel: Boolean)

/** v0.8 study log: one entry [Merger.match] looked at for a detection, and how it compared. */
class MergeCandidate(
  val entryId: String,
  val distanceM: Double,
  val similarity: Float,
  val sameLabel: Boolean,
  /** Within the merge radius of [Merger.radius]. */
  val withinRadius: Boolean,
  /** A tentative entry of the running scan. */
  val tentative: Boolean,
)

/**
 * Plan section 6: merging repeated sightings into entries. Pure Kotlin, unit-tested.
 */
object Merger {
  /** Cosine similarity; the same computation as `Embedder.cosine`, without its Android imports. */
  fun cosine(a: FloatArray, b: FloatArray): Float {
    if (a.size != b.size || a.isEmpty()) return 0f
    var dot = 0f
    var na = 0f
    var nb = 0f
    for (i in a.indices) {
      dot += a[i] * b[i]
      na += a[i] * a[i]
      nb += b[i] * b[i]
    }
    val denom = sqrt(na) * sqrt(nb)
    return if (denom == 0f) 0f else dot / denom
  }

  /** Two sightings can be the same item within clamp(0.5 × max(sizes) + 0.15, 0.20, 0.50) metres. */
  fun radius(sizeA: Double, sizeB: Double): Double =
    (ScanConfig.MERGE_RADIUS_FACTOR * max(sizeA, sizeB) + ScanConfig.MERGE_RADIUS_OFFSET_M)
      .coerceIn(ScanConfig.MERGE_RADIUS_MIN_M, ScanConfig.MERGE_RADIUS_MAX_M)

  /**
   * Among [candidates] within the merge radius of [obs] and not in [used], the nearest whose label
   * equals the detection's or whose embedding is at least [ScanConfig.MERGE_SIMILARITY] similar.
   * The caller passes only entries of the same map (and of the same scan target).
   */
  fun match(obs: Observation, candidates: List<InventoryEntry>, used: Set<String>): Match? {
    var best: Match? = null
    for (e in candidates) {
      if (e.id in used) continue
      val d = distance(e.position, obs.mapPoint)
      if (d > radius(e.sizeM, obs.sizeM)) continue
      val byLabel = e.label == obs.label
      val sim = cosine(e.embedding, obs.embedding)
      if (!byLabel && sim < ScanConfig.MERGE_SIMILARITY) continue
      if (best == null || d < best.distanceM) best = Match(e, d, sim, byLabel)
    }
    return best
  }

  /**
   * v0.8 study log, read only: the [n] entries of [candidates] (not in [used]) nearest to [obs],
   * nearest first, with the quantities [match] compares. It decides nothing.
   */
  fun nearest(obs: Observation, candidates: List<InventoryEntry>, used: Set<String>, tentativeIds: Set<String>, n: Int): List<MergeCandidate> =
    candidates
      .filter { it.id !in used }
      .map { it to distance(it.position, obs.mapPoint) }
      .sortedBy { it.second }
      .take(n)
      .map { (e, d) ->
        MergeCandidate(e.id, d, cosine(e.embedding, obs.embedding), e.label == obs.label, d <= radius(e.sizeM, obs.sizeM), e.id in tentativeIds)
      }

  /** A first sighting as an entry (tentative until its second sighting). */
  fun newEntry(
    id: String,
    obs: Observation,
    kf: KeyframeInfo,
    mapId: String,
    container: String?,
    scanId: String,
    detectorId: String,
    sessionId: String,
    embeddingModel: String,
  ): InventoryEntry =
    InventoryEntry(
      id = id,
      name = obs.label,
      nameEdited = false,
      label = obs.label,
      labelVotes = mapOf(obs.label to obs.score.toDouble()),
      mapId = mapId,
      position = obs.mapPoint,
      viewPosition = kf.cameraMap,
      roomId = kf.roomId,
      container = container,
      sizeM = obs.sizeM,
      cropFile = InventoryEntry.cropPath(id),
      embedding = obs.embedding,
      embeddingModel = embeddingModel,
      captureDistance = obs.distanceM.toFloat(),
      cropWidthPx = obs.cropWidthPx,
      cropHeightPx = obs.cropHeightPx,
      imageWidth = kf.imageWidth,
      imageHeight = kf.imageHeight,
      bestQuality = obs.quality,
      bestScore = obs.score,
      worldPosition = obs.worldPoint,
      sessionId = sessionId,
      sightings = listOf(Sighting(kf.tMs, obs.mapPoint, obs.sizeM, obs.label, obs.score, scanId, kf.index)),
      sightingCount = 1,
      firstSeenMs = kf.tMs,
      lastSeenMs = kf.tMs,
      detector = detectorId,
      createdScan = scanId,
    )

  /**
   * [entry] after one more sighting: the position becomes the mean of the latest sightings, the
   * label vote is added, last seen moves on, and the best crop (with its embedding and everything
   * verification needs) is replaced when the new crop's quality is higher. Second: whether the
   * best crop was replaced.
   */
  fun addSighting(
    entry: InventoryEntry,
    obs: Observation,
    kf: KeyframeInfo,
    scanId: String,
    sessionId: String,
  ): Pair<InventoryEntry, Boolean> {
    val sightings =
      (entry.sightings + Sighting(kf.tMs, obs.mapPoint, obs.sizeM, obs.label, obs.score, scanId, kf.index))
        .takeLast(ScanConfig.POSITION_SIGHTINGS)
    var sx = 0.0
    var sy = 0.0
    var sz = 0.0
    for (s in sightings) {
      sx += s.position.x
      sy += s.position.y
      sz += s.position.z
    }
    val n = sightings.size
    val votes = LinkedHashMap(entry.labelVotes)
    votes[obs.label] = (votes[obs.label] ?: 0.0) + obs.score
    // The current label stays on a tie, so the name does not flicker.
    val current = votes[entry.label] ?: 0.0
    val top = votes.maxByOrNull { it.value }
    val label = if (top != null && top.value > current) top.key else entry.label
    val better = obs.quality > entry.bestQuality
    var e =
      entry.copy(
        name = if (entry.nameEdited) entry.name else label,
        label = label,
        labelVotes = votes,
        position = Vec3(sx / n, sy / n, sz / n),
        roomId = entry.roomId ?: kf.roomId,
        sightings = sightings,
        sightingCount = entry.sightingCount + 1,
        lastSeenMs = max(entry.lastSeenMs, kf.tMs),
      )
    if (better) {
      e =
        e.copy(
          viewPosition = kf.cameraMap,
          sizeM = obs.sizeM,
          embedding = obs.embedding,
          captureDistance = obs.distanceM.toFloat(),
          cropWidthPx = obs.cropWidthPx,
          cropHeightPx = obs.cropHeightPx,
          imageWidth = kf.imageWidth,
          imageHeight = kf.imageHeight,
          bestQuality = obs.quality,
          bestScore = obs.score,
          worldPosition = obs.worldPoint,
          sessionId = sessionId,
        )
    }
    return e to better
  }
}
