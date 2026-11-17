package edu.cs6301.findback.scan

import edu.cs6301.findback.map.distance
import java.util.UUID

/** What became of one kept detection in a keyframe. */
class ObservationResult(
  val observation: Observation,
  /** One of the KIND_ values of [ScanSession]. */
  val kind: String,
  /** The entry (real or tentative) the detection went to; null when unmatched in passive mode. */
  val entryId: String?,
  /** Similarity to that entry's embedding before this sighting; null for a new entry. */
  val similarity: Float?,
  /** Distance to that entry's position before this sighting; null for a new entry. */
  val distanceM: Double?,
)

class MergeResult(
  val results: List<ObservationResult>,
  /** Real entries a detection matched in this keyframe (entries promoted in it included). */
  val matchedIds: Set<String>,
  /** Entries that became real in this keyframe. */
  val added: List<InventoryEntry>,
  /** Entry id and crop handle of every real entry whose photo must be (re)written. */
  val cropUpdates: List<Pair<String, Any?>>,
)

/** One in-view entry in one keyframe, for the log. */
class EvidenceCheck(val entryId: String, val view: ViewCheck, val similarity: Float?)

/** One entry's outcome when a scan or a passive window ends. [entry] is the entry afterwards. */
class EvidenceResult(
  val entry: InventoryEntry,
  val outcome: String,
  val eligible: Boolean,
  val isNew: Boolean,
  val tally: Tally?,
)

/** A new entry that looks like an older one that was looked for and not seen. */
class MovedCandidate(val newEntry: InventoryEntry, val oldEntry: InventoryEntry, val similarity: Float)

class SessionSummary(
  val results: List<EvidenceResult>,
  val moved: List<MovedCandidate>,
  /** Entries created by this scan. */
  val newCount: Int,
  /** Older entries seen again. */
  val seenAgain: Int,
  val missed: Int,
  val notObserved: Int,
  /** Detections seen in one keyframe only: never became entries. */
  val tentativeDropped: Int,
)

/**
 * The state of one scan, or of one passive window: tentative entries, what each keyframe said about
 * each entry, and at the end one outcome per entry (plan sections 6 and 7). Pure Kotlin; the device
 * side ([ScanController]) feeds it keyframes from its worker thread, the only thread that uses it.
 *
 * A SCAN session creates and updates entries. A PASSIVE session only gathers evidence: detections
 * that match an entry count as seen, nothing is created, and no entry's position, votes or photo
 * change.
 */
class ScanSession(
  val id: String,
  val mode: Mode,
  val mapId: String,
  /** Scan target: a container name, or null for the open area (always null in passive mode). */
  val container: String?,
  val detectorId: String,
  private val appSessionId: String,
  private val embeddingModel: String,
  val startMs: Long,
  private val store: InventoryStore,
  /** Block B (not wired in yet): compare new entries with `missed` ones when the scan ends. */
  private val movedCandidates: Boolean = false,
  private val newId: () -> String = { UUID.randomUUID().toString().substring(0, 8) },
) {
  enum class Mode { SCAN, PASSIVE }

  companion object {
    /** A detection matched an entry that already existed. */
    const val KIND_KNOWN = "known"
    /** A tentative entry got its second sighting and became real. */
    const val KIND_PROMOTED = "promoted"
    /** A second sighting of a tentative entry, too soon to promote it. */
    const val KIND_TENTATIVE = "tentative"
    /** No match: a tentative entry was created. */
    const val KIND_NEW = "new"
    /** Passive mode: no match, nothing done. */
    const val KIND_UNMATCHED = "unmatched"
  }

  private class Tentative(var entry: InventoryEntry, val firstMs: Long, var crop: Any?)

  private val tentative = ArrayList<Tentative>()
  private val tallies = HashMap<String, Tally>()
  private val lastChecked = HashMap<String, Int>()
  private var checkSeq = 0

  /** Entries this scan created, and all entries it sighted (created ones included). */
  private val created = LinkedHashSet<String>()
  private val sighted = LinkedHashSet<String>()

  var keyframes = 0
    private set

  val tentativeCount: Int
    get() = tentative.size

  /** Real entries this scan has detected so far: the count in the status hint. */
  val itemCount: Int
    get() = sighted.size

  /** Entries with at least one informative keyframe so far (passive windows are logged only then). */
  val talliedCount: Int
    get() = tallies.size

  /** An entry takes part in this session when it is in this map and in the scanned target. */
  fun eligible(e: InventoryEntry): Boolean = e.mapId == mapId && InventoryText.sameContainer(e.container, container)

  private fun tally(id: String): Tally = tallies.getOrPut(id) { Tally() }

  /**
   * Merge one keyframe's kept detections. Within a keyframe an entry takes at most one detection
   * (two detections of one frame are two objects), detections in descending score order.
   */
  fun merge(kf: KeyframeInfo, observations: List<Observation>): MergeResult {
    keyframes++
    val results = ArrayList<ObservationResult>()
    val matched = LinkedHashSet<String>()
    val added = ArrayList<InventoryEntry>()
    val crops = ArrayList<Pair<String, Any?>>()
    val used = HashSet<String>()
    for (obs in observations.sortedByDescending { it.score }) {
      val real = store.snapshot().filter { eligible(it) }
      val candidates = if (mode == Mode.SCAN) real + tentative.map { it.entry } else real
      val m = Merger.match(obs, candidates, used)
      if (m == null) {
        if (mode == Mode.SCAN) {
          val e = Merger.newEntry(newId(), obs, kf, mapId, container, id, detectorId, appSessionId, embeddingModel)
          tentative.add(Tentative(e, kf.tMs, obs.crop))
          used.add(e.id)
          results.add(ObservationResult(obs, KIND_NEW, e.id, null, null))
        } else {
          results.add(ObservationResult(obs, KIND_UNMATCHED, null, null, null))
        }
        continue
      }
      used.add(m.entry.id)
      val t = tentative.firstOrNull { it.entry.id == m.entry.id }
      if (t != null) {
        val (e, better) = Merger.addSighting(t.entry, obs, kf, id, appSessionId)
        t.entry = e
        if (better) t.crop = obs.crop
        if (kf.tMs - t.firstMs >= ScanConfig.TENTATIVE_CONFIRM_MS) {
          tentative.remove(t)
          store.add(e)
          created.add(e.id)
          sighted.add(e.id)
          matched.add(e.id)
          added.add(e)
          crops.add(e.id to t.crop)
          tally(e.id).let {
            it.matchedKeyframes++
            it.lastSeenMs = kf.tMs
          }
          results.add(ObservationResult(obs, KIND_PROMOTED, e.id, m.similarity, m.distanceM))
        } else {
          results.add(ObservationResult(obs, KIND_TENTATIVE, e.id, m.similarity, m.distanceM))
        }
        continue
      }
      // A real entry.
      matched.add(m.entry.id)
      tally(m.entry.id).let {
        it.matchedKeyframes++
        it.lastSeenMs = kf.tMs
      }
      if (mode == Mode.SCAN) {
        // Applied to the entry as stored now, so a rename made on the UI thread meanwhile stays.
        var better = false
        val e =
          store.update(m.entry.id) { cur ->
            val (next, b) = Merger.addSighting(cur, obs, kf, id, appSessionId)
            better = b
            next
          }
        if (e != null) {
          sighted.add(e.id)
          if (better) crops.add(e.id to obs.crop)
        }
      }
      results.add(ObservationResult(obs, KIND_KNOWN, m.entry.id, m.similarity, m.distanceM))
    }
    return MergeResult(results, matched, added, crops)
  }

  /**
   * What this keyframe says about every eligible entry that no detection matched in it. The cheap
   * depth test runs for all of them; [appearance] (crop the expected window, embed, compare; null
   * when it cannot run) runs for at most [ScanConfig.APPEARANCE_CHECKS_PER_KEYFRAME] observable
   * entries, those checked least recently first, and not for entries already seen in this session.
   * Returns the entries in view, for the log.
   */
  fun evidence(
    kf: KeyframeInfo,
    geom: KeyframeGeometry,
    matchedIds: Set<String>,
    appearance: (InventoryEntry, ViewCheck) -> Float?,
  ): List<EvidenceCheck> {
    val inView = ArrayList<Pair<InventoryEntry, ViewCheck>>()
    val pending = ArrayList<Pair<InventoryEntry, ViewCheck>>()
    for (e in store.snapshot()) {
      if (!eligible(e) || e.id in matchedIds) continue
      val vc = EvidenceRules.classify(e.position, geom)
      if (vc.visibility == Visibility.OUT_OF_VIEW) continue
      val t = tally(e.id)
      t.inViewKeyframes++
      when (vc.visibility) {
        Visibility.HIDDEN -> t.hiddenKeyframes++
        Visibility.NO_DEPTH -> t.noDepthKeyframes++
        else -> {
          t.observableKeyframes++
          if (!t.seen) pending.add(e to vc)
        }
      }
      inView.add(e to vc)
    }
    val sims = HashMap<String, Float>()
    val chosen = EvidenceRules.pickForAppearance(pending, { it.first.id }, { it.second.distanceM }, lastChecked)
    for ((e, vc) in chosen) {
      val sim = appearance(e, vc) ?: continue
      val t = tally(e.id)
      t.checkedKeyframes++
      lastChecked[e.id] = ++checkSeq
      if (sim > (t.bestSimilarity ?: -1f)) t.bestSimilarity = sim
      if (sim >= ScanConfig.SEEN_SIMILARITY) {
        t.appearanceSeen = true
        t.lastSeenMs = kf.tMs
      }
      sims[e.id] = sim
    }
    return inView.map { (e, vc) -> EvidenceCheck(e.id, vc, sims[e.id]) }
  }

  /**
   * End of the scan or window: one outcome per entry, written to the store with its log lines.
   *
   * SCAN: every entry of this map gets an outcome (entries of other targets `not_observed`),
   * tentative entries are dropped, and new entries are compared with entries that were looked for
   * and not seen (moved candidates). PASSIVE: only `seen` and `missed` are recorded; an entry the
   * window learned nothing about is left as it is.
   */
  fun finish(nowMs: Long): SessionSummary {
    val source = if (mode == Mode.SCAN) ScanConfig.SOURCE_SCAN else ScanConfig.SOURCE_PASSIVE
    val results = ArrayList<EvidenceResult>()
    for (e in store.snapshot()) {
      if (e.mapId != mapId) continue
      val elig = eligible(e)
      val t = tallies[e.id]
      val outcome = EvidenceRules.outcome(t, elig)
      if (mode == Mode.PASSIVE && outcome == ScanConfig.OUTCOME_NOT_OBSERVED) continue
      val at = if (outcome == ScanConfig.OUTCOME_SEEN) t?.lastSeenMs ?: nowMs else nowMs
      // Written at once, so the moved-candidate search below sees this scan's `missed` entries.
      val updated = store.update(e.id) { it.withOutcome(outcome, at, source, id) } ?: continue
      results.add(EvidenceResult(updated, outcome, elig, e.id in created, t))
    }

    val moved = ArrayList<MovedCandidate>()
    if (mode == Mode.SCAN) {
      for (createdId in if (movedCandidates) created else emptySet()) {
        val fresh = store.get(createdId) ?: continue
        var best: InventoryEntry? = null
        var bestSim = ScanConfig.MOVED_SIMILARITY
        for (old in store.snapshot()) {
          if (old.id in created || old.mapId != mapId) continue
          if (old.evidence.lastOutcome != ScanConfig.OUTCOME_MISSED || old.label != fresh.label) continue
          val sim = Merger.cosine(old.embedding, fresh.embedding)
          if (sim >= bestSim) {
            best = old
            bestSim = sim
          }
        }
        if (best != null) {
          val from = MovedFrom(best.id, best.name, best.roomId, bestSim, id)
          val marked = store.update(createdId) { it.copy(movedFrom = from) } ?: continue
          moved.add(MovedCandidate(marked, best, bestSim))
        }
      }
      store.setLastScan(id, nowMs)
    }
    // One log line per changed entry: `update` when the scan sighted it, else `evidence`.
    val finished = ArrayList<EvidenceResult>()
    for (r in results) {
      val now = store.get(r.entry.id) ?: continue
      if (r.entry.id in sighted) store.logUpdate(now.id) else store.logEvidence(now.id)
      finished.add(EvidenceResult(now, r.outcome, r.eligible, r.isNew, r.tally))
    }
    store.flush()

    val dropped = tentative.size
    tentative.clear()
    return SessionSummary(
      results = finished,
      moved = moved,
      newCount = finished.count { it.isNew },
      seenAgain = finished.count { !it.isNew && it.outcome == ScanConfig.OUTCOME_SEEN },
      missed = finished.count { it.outcome == ScanConfig.OUTCOME_MISSED },
      notObserved = finished.count { it.outcome == ScanConfig.OUTCOME_NOT_OBSERVED },
      tentativeDropped = dropped,
    )
  }

  /** Distance between two entries' positions (for the moved-candidate log). */
  fun separation(a: InventoryEntry, b: InventoryEntry): Double = distance(a.position, b.position)
}
