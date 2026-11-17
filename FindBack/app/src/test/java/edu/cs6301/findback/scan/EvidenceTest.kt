package edu.cs6301.findback.scan

import edu.cs6301.findback.map.Pose4
import edu.cs6301.findback.map.Vec3
import edu.cs6301.findback.scan.ScanFixtures.embedding
import edu.cs6301.findback.scan.ScanFixtures.keyframe
import edu.cs6301.findback.scan.ScanFixtures.observe
import edu.cs6301.findback.scan.ScanFixtures.poseLookingAt
import java.io.File
import java.nio.file.Files
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

/** Plan section 7: visibility-aware evidence. Absence counts only where the camera really looked. */
class EvidenceTest {
  private val dir: File = Files.createTempDirectory("inventory").toFile()
  private val store = InventoryStore(dir, "sess0001")
  private var nextId = 0

  private fun session(
    mode: ScanSession.Mode = ScanSession.Mode.SCAN,
    container: String? = null,
    id: String = "scan0002",
    mapId: String = "map00001",
  ) = ScanSession(id, mode, mapId, container, "effdet0", "sess0001", "mobilenet_v3_small", 100_000L, store, movedCandidates = true) { "n%07d".format(nextId++) }

  private val shelf = Vec3(0.4, 0.9, -1.1)

  /** An entry already in the inventory, as an earlier scan would have left it. */
  private fun entry(
    id: String,
    p: Vec3,
    label: String = "cup",
    emb: FloatArray = embedding(0),
    container: String? = null,
    mapId: String = "map00001",
  ): InventoryEntry {
    val pose = poseLookingAt(p)
    val e =
      Merger.newEntry(id, observe(p, pose, label, emb)!!, keyframe(0, 1_000L, pose), mapId, container, "scan0001", "effdet0", "sess0000", "mobilenet_v3_small")
        .withOutcome(ScanConfig.OUTCOME_SEEN, 1_000L, ScanConfig.SOURCE_SCAN, "scan0001")
    store.add(e)
    return e
  }

  private fun logLines(): List<JSONObject> =
    File(dir, InventoryStore.LOG_NAME).readLines().filter { it.isNotBlank() }.map { JSONObject(it) }

  /** The camera looks straight at [p] from [back] metres; the depth image shows [depthM] everywhere. */
  private fun geomAt(p: Vec3, depthM: Double, back: Double = 1.0, side: Double = 0.0) =
    ScanFixtures.geometry(poseLookingAt(p, back, side), ScanFixtures.flatDepth(depthM))

  // ------------------------------------------------------------------ one keyframe

  @Test
  fun observableWhenTheDepthReachesThePlace() {
    val vc = EvidenceRules.classify(shelf, geomAt(shelf, 1.0))
    assertEquals(Visibility.OBSERVABLE, vc.visibility)
    assertEquals(960.0, vc.u, 1e-6)
    assertEquals(540.0, vc.v, 1e-6)
    assertEquals(1.0, vc.zM, 1e-9)
    assertEquals(1.0, vc.distanceM, 1e-9)
    assertEquals(1.0, vc.measuredM!!, 1e-9)
    // The surface behind an item that was taken away is farther: the place itself is still seen.
    assertEquals(Visibility.OBSERVABLE, EvidenceRules.classify(shelf, geomAt(shelf, 2.5)).visibility)
  }

  @Test
  fun hiddenWhenSomethingIsClearlyInFront() {
    // At 1 m the margin is 15 cm; at 2 m it is 15% of the distance, 30 cm.
    assertEquals(0.15, EvidenceRules.hiddenMargin(1.0), 1e-12)
    assertEquals(0.15, EvidenceRules.hiddenMargin(0.5), 1e-12)
    assertEquals(0.30, EvidenceRules.hiddenMargin(2.0), 1e-12)
    assertEquals(Visibility.HIDDEN, EvidenceRules.classify(shelf, geomAt(shelf, 0.84)).visibility)
    assertEquals(Visibility.OBSERVABLE, EvidenceRules.classify(shelf, geomAt(shelf, 0.86)).visibility)
    assertEquals(Visibility.HIDDEN, EvidenceRules.classify(shelf, geomAt(shelf, 1.69, back = 2.0)).visibility)
    assertEquals(Visibility.OBSERVABLE, EvidenceRules.classify(shelf, geomAt(shelf, 1.71, back = 2.0)).visibility)
    assertEquals(0.84, EvidenceRules.classify(shelf, geomAt(shelf, 0.84)).measuredM!!, 1e-9)
  }

  @Test
  fun noDepthSaysNothing() {
    assertEquals(Visibility.NO_DEPTH, EvidenceRules.classify(shelf, geomAt(shelf, 0.0)).visibility)
    // A depth image that crops the CPU image has no depth near the cropped edge.
    val cropped = Affine2(160.0 / 1920, 0.0, 0.0, 0.0, 90.0 / 864, -108.0 * 90.0 / 864)
    val pose = poseLookingAt(shelf)
    val up = Vec3(shelf.x, shelf.y + 0.319, shelf.z) // v ≈ 97: in view, but above the depth image (v < 108)
    val geom = ScanFixtures.geometry(pose, ScanFixtures.flatDepth(1.0, cropped))
    assertTrue(geom.projectMap(up)!!.v > ScanConfig.VIEW_MARGIN * ScanFixtures.H)
    assertEquals(Visibility.NO_DEPTH, EvidenceRules.classify(up, geom).visibility)
  }

  @Test
  fun inViewMeansInsideTheMarginAndWithinRange() {
    fun vis(p: Vec3, back: Double = 1.0, side: Double = 0.0) =
      EvidenceRules.classify(p, ScanFixtures.geometry(poseLookingAt(shelf, back, side), ScanFixtures.flatDepth(back))).visibility
    assertEquals(Visibility.OBSERVABLE, vis(shelf))
    // 0.58 m to the side at 1 m is pixel 1765 of 1920, just inside the 8% margin; 0.59 m is outside.
    assertEquals(Visibility.OBSERVABLE, vis(Vec3(shelf.x, shelf.y, shelf.z), side = -0.58))
    assertEquals(Visibility.OUT_OF_VIEW, vis(Vec3(shelf.x, shelf.y, shelf.z), side = -0.59))
    // Nearer than 0.3 m or farther than 3 m is not in view.
    assertEquals(Visibility.OUT_OF_VIEW, vis(shelf, back = 0.29))
    assertEquals(Visibility.OBSERVABLE, vis(shelf, back = 0.31))
    assertEquals(Visibility.OBSERVABLE, vis(shelf, back = 2.99))
    assertEquals(Visibility.OUT_OF_VIEW, vis(shelf, back = 3.01))
    // Behind the camera.
    val behind = ScanFixtures.geometry(ScanFixtures.pose(Vec3.ZERO), ScanFixtures.flatDepth(1.0), Pose4.IDENTITY)
    assertEquals(Visibility.OUT_OF_VIEW, EvidenceRules.classify(Vec3(0.0, 0.0, 1.0), behind).visibility)
  }

  // ------------------------------------------------------------------ outcomes

  @Test
  fun theThreeOutcomes() {
    assertEquals(ScanConfig.OUTCOME_NOT_OBSERVED, EvidenceRules.outcome(null, eligible = true))
    assertEquals(ScanConfig.OUTCOME_SEEN, EvidenceRules.outcome(Tally().apply { matchedKeyframes = 1 }, true))
    assertEquals(ScanConfig.OUTCOME_SEEN, EvidenceRules.outcome(Tally().apply { appearanceSeen = true; checkedKeyframes = 5 }, true))
    assertEquals(ScanConfig.OUTCOME_MISSED, EvidenceRules.outcome(Tally().apply { observableKeyframes = 3; checkedKeyframes = 3 }, true))
    assertEquals(ScanConfig.OUTCOME_NOT_OBSERVED, EvidenceRules.outcome(Tally().apply { observableKeyframes = 2; checkedKeyframes = 2 }, true))
    // Hidden or depth-less keyframes never add up to "missed", however many there are.
    assertEquals(ScanConfig.OUTCOME_NOT_OBSERVED, EvidenceRules.outcome(Tally().apply { inViewKeyframes = 50; hiddenKeyframes = 30; noDepthKeyframes = 20 }, true))
    // Observable keyframes in which the appearance check did not run do not count either.
    assertEquals(ScanConfig.OUTCOME_NOT_OBSERVED, EvidenceRules.outcome(Tally().apply { observableKeyframes = 9; checkedKeyframes = 2 }, true))
    // An entry outside the scanned target is never looked for.
    assertEquals(ScanConfig.OUTCOME_NOT_OBSERVED, EvidenceRules.outcome(Tally().apply { matchedKeyframes = 4 }, eligible = false))
  }

  @Test
  fun missedNeedsThreeObservableKeyframesAndHiddenOnesDoNotCount() {
    val gone = entry("gone0001", shelf)
    val covered = entry("cover001", Vec3(shelf.x + 0.5, shelf.y, shelf.z), label = "book", emb = embedding(1))
    val s = session()
    // The cup's place is in plain view and looks different now; the book is behind something
    // (the depth in its direction is 0.5 m, the book is 1.3 m away).
    val looks = HashMap<String, Int>()
    for (i in 0 until 5) {
      val pose = poseLookingAt(shelf, back = 1.3, side = 0.05 * i)
      val coveredU = ScanFixtures.geometry(pose, ScanFixtures.flatDepth(1.3)).projectMap(covered.position)!!.u
      val depth = ScanFixtures.depthOf { x, _ -> if (kotlin.math.abs(x - coveredU * ScanFixtures.DW / ScanFixtures.W) < 8) 0.5 else 1.3 }
      val geom = ScanFixtures.geometry(pose, depth)
      val checks =
        s.evidence(keyframe(i, 100_000L + 500L * i, pose), geom, emptySet()) { e, _ ->
          looks[e.id] = (looks[e.id] ?: 0) + 1
          0.31f
        }
      assertEquals(setOf("gone0001", "cover001"), checks.map { it.entryId }.toSet())
      assertEquals(Visibility.OBSERVABLE, checks.single { it.entryId == "gone0001" }.view.visibility)
      assertEquals(Visibility.HIDDEN, checks.single { it.entryId == "cover001" }.view.visibility)
      assertEquals(0.31f, checks.single { it.entryId == "gone0001" }.similarity!!, 0f)
      assertNull(checks.single { it.entryId == "cover001" }.similarity)
      if (i == 1) {
        // Two observable keyframes are not yet enough: a scan that ended here would say nothing.
        assertEquals(mapOf("gone0001" to 2), looks)
      }
    }
    assertEquals(mapOf("gone0001" to 5), looks) // the hidden place was never compared
    val sum = s.finish(110_000L)
    val byId = sum.results.associateBy { it.entry.id }
    assertEquals(ScanConfig.OUTCOME_MISSED, byId.getValue("gone0001").outcome)
    assertEquals(5, byId.getValue("gone0001").tally!!.observableKeyframes)
    assertEquals(5, byId.getValue("gone0001").tally!!.checkedKeyframes)
    assertEquals(0.31f, byId.getValue("gone0001").tally!!.bestSimilarity!!, 0f)
    assertEquals(ScanConfig.OUTCOME_NOT_OBSERVED, byId.getValue("cover001").outcome)
    assertEquals(5, byId.getValue("cover001").tally!!.hiddenKeyframes)
    assertEquals(0, byId.getValue("cover001").tally!!.observableKeyframes)
    assertEquals(1, sum.missed)
    assertEquals(1, sum.notObserved)
    assertEquals(0, sum.seenAgain)

    val g = store.get("gone0001")!!
    assertEquals(1, g.evidence.missedCount)
    assertEquals(1, g.evidence.seenCount) // from the scan that created it
    assertEquals(110_000L, g.evidence.lastMissedMs)
    assertEquals(ScanConfig.OUTCOME_MISSED, g.evidence.lastOutcome)
    assertEquals("scan0002", g.evidence.lastSourceId)
    assertEquals(1_000L, g.lastSeenMs) // nothing is moved or deleted, and "last seen" stays
    assertEquals(gone.position, g.position)
    val c = store.get("cover001")!!
    assertEquals(1, c.evidence.notObservedCount)
    assertEquals(110_000L, c.evidence.lastNotObservedMs)
    assertEquals(0, c.evidence.missedCount)
    assertEquals("scan0002", store.lastScanId)
    assertEquals("looked for in the last scan, not seen", InventoryText.evidence(g, 200_000L, store.lastScanId))
  }

  @Test
  fun twoObservableKeyframesAreNotEnough() {
    entry("gone0001", shelf)
    val s = session()
    for (i in 0 until 2) {
      val pose = poseLookingAt(shelf, side = 0.05 * i)
      s.evidence(keyframe(i, 100_000L + 500L * i, pose), ScanFixtures.geometry(pose, ScanFixtures.flatDepth(1.0)), emptySet()) { _, _ -> 0.1f }
    }
    assertEquals(ScanConfig.OUTCOME_NOT_OBSERVED, s.finish(110_000L).results.single().outcome)
    assertEquals(0, store.get("gone0001")!!.evidence.missedCount)
  }

  @Test
  fun seenByDetectionOrByAppearance() {
    entry("det00001", shelf)
    val other = Vec3(shelf.x + 0.5, shelf.y, shelf.z)
    entry("app00001", other, label = "book", emb = embedding(1))
    val s = session()
    var appChecks = 0
    for (i in 0 until 4) {
      val pose = poseLookingAt(shelf, back = 1.3, side = 0.05 * i)
      // The map frame is turned 34° against the world (T_WM), so the book, 0.5 m along map +x,
      // is about 1.6 m from the camera, not 1.3 m. The wall behind both is at 2 m, so both
      // places are observable. (A wall at 1.3 m would stand in front of the book: hidden.)
      val geom = ScanFixtures.geometry(pose, ScanFixtures.flatDepth(2.0))
      val kf = keyframe(i, 100_000L + 500L * i, pose)
      // The detector finds the cup in the second keyframe only, and never the book.
      val obs = if (i == 1) listOf(observe(shelf, pose, "cup", embedding(0, 0.9))!!) else emptyList()
      val m = s.merge(kf, obs)
      if (i == 1) assertEquals(setOf("det00001"), m.matchedIds)
      s.evidence(kf, geom, m.matchedIds) { e, vc ->
        assertEquals(Visibility.OBSERVABLE, vc.visibility)
        if (e.id == "app00001") {
          appChecks++
          if (i == 2) ScanConfig.SEEN_SIMILARITY else 0.40f // matches once, at exactly the threshold
        } else 0.20f
      }
    }
    // Once the book was seen it is not compared again: keyframes 0, 1 and 2 only.
    assertEquals(3, appChecks)
    val byId = s.finish(110_000L).results.associateBy { it.entry.id }
    assertEquals(ScanConfig.OUTCOME_SEEN, byId.getValue("det00001").outcome)
    assertEquals(1, byId.getValue("det00001").tally!!.matchedKeyframes)
    assertEquals(ScanConfig.OUTCOME_SEEN, byId.getValue("app00001").outcome)
    assertEquals(ScanConfig.SEEN_SIMILARITY, byId.getValue("app00001").tally!!.bestSimilarity!!, 0f)
    assertEquals(0, byId.getValue("app00001").tally!!.matchedKeyframes)
    // Last seen is the keyframe that saw it, not the end of the scan.
    assertEquals(100_500L, store.get("det00001")!!.lastSeenMs)
    assertEquals(101_000L, store.get("app00001")!!.lastSeenMs)
    assertEquals(2, store.get("app00001")!!.evidence.seenCount)
    // A detection in a scan is a sighting; an appearance match is not.
    assertEquals(2, store.get("det00001")!!.sightingCount)
    assertEquals(1, store.get("app00001")!!.sightingCount)
  }

  @Test
  fun atMostFourAppearanceChecksPerKeyframeLeastRecentlyCheckedFirst() {
    // Six entries on one shelf, all in view and observable in every keyframe.
    val ids = (0 until 6).map { "shelf00$it" }
    for ((i, id) in ids.withIndex()) entry(id, Vec3(shelf.x - 0.5 + 0.2 * i, shelf.y, shelf.z), label = "item$i", emb = embedding(i % 6))
    val s = session()
    val pose = poseLookingAt(shelf, back = 1.5)
    // The map frame is turned 34° against the world (T_WM), so the row of entries along map +x
    // runs diagonally away from the camera: they are 1.2 to 1.8 m away. The wall behind them is
    // at 2 m, so every place is observable. (With the wall at 1.5 m the farthest entry would be
    // behind it, correctly hidden, and never compared.)
    val geom = ScanFixtures.geometry(pose, ScanFixtures.flatDepth(2.0))
    val perKeyframe = ArrayList<List<String>>()
    for (i in 0 until 5) {
      val checked = ArrayList<String>()
      val checks =
        s.evidence(keyframe(i, 100_000L + 500L * i, pose), geom, emptySet()) { e, _ ->
          checked.add(e.id)
          0.2f
        }
      assertEquals(6, checks.size) // all six are in view and logged...
      assertEquals(ScanConfig.APPEARANCE_CHECKS_PER_KEYFRAME, checked.size) // ...four are compared
      assertEquals(4, checks.count { it.similarity != null })
      perKeyframe.add(checked)
      if (i == 2) {
        // After three keyframes every place was observable three times but compared only twice:
        // that is not yet "missed".
        val counts = perKeyframe.flatten().groupingBy { it }.eachCount()
        assertEquals(ids.associateWith { 2 }, counts)
      }
    }
    // Keyframe 0 takes the four nearest; keyframe 1 the two never checked, then the two oldest.
    assertEquals(setOf("shelf002", "shelf003"), perKeyframe[0].toSet() intersect setOf("shelf002", "shelf003"))
    assertEquals(4, perKeyframe[0].toSet().size)
    val first = perKeyframe[0].toSet()
    val rest = ids.toSet() - first
    assertEquals(rest.toList().sorted(), perKeyframe[1].take(2).sorted())
    assertEquals(perKeyframe[0].take(2), perKeyframe[1].drop(2))
    // Over five keyframes there were 20 checks, three or four per entry.
    val total = perKeyframe.flatten().groupingBy { it }.eachCount()
    assertEquals(20, total.values.sum())
    val results = s.finish(110_000L).results.associateBy { it.entry.id }
    for (id in ids) {
      val t = results.getValue(id).tally!!
      assertEquals(5, t.observableKeyframes)
      assertEquals(total.getValue(id), t.checkedKeyframes)
      val expected = if (total.getValue(id) >= ScanConfig.MISSED_MIN_KEYFRAMES) ScanConfig.OUTCOME_MISSED else ScanConfig.OUTCOME_NOT_OBSERVED
      assertEquals(id, expected, results.getValue(id).outcome)
    }
    assertTrue(total.values.all { it == 3 || it == 4 })

    // The pure rule: never checked first, then the oldest check, nearer first among equals.
    val picked =
      EvidenceRules.pickForAppearance(
        listOf("a" to 2.0, "b" to 1.0, "c" to 3.0, "d" to 0.5, "e" to 0.7, "f" to 9.0),
        { it.first }, { it.second },
        mapOf("a" to 7, "b" to 3, "d" to 9, "e" to 9),
      )
    assertEquals(listOf("c", "f", "b", "a"), picked.map { it.first })
  }

  @Test
  fun entriesInAContainerAreOnlyLookedForInAScanOfThatContainer() {
    entry("open0001", shelf)
    entry("drawer01", Vec3(shelf.x + 0.3, shelf.y, shelf.z), label = "scissors", emb = embedding(1), container = "desk drawer")
    val pose = poseLookingAt(shelf, back = 1.3)
    val geom = ScanFixtures.geometry(pose, ScanFixtures.flatDepth(1.3))

    // A scan of the open area: the drawer's entry is not looked for, although its place is in view.
    val open = session(id = "scan0002")
    for (i in 0 until 4) {
      val checks = open.evidence(keyframe(i, 100_000L + 500L * i, pose), geom, emptySet()) { _, _ -> 0.1f }
      assertEquals(listOf("open0001"), checks.map { it.entryId })
    }
    val r1 = open.finish(110_000L).results.associateBy { it.entry.id }
    assertEquals(ScanConfig.OUTCOME_MISSED, r1.getValue("open0001").outcome)
    assertEquals(ScanConfig.OUTCOME_NOT_OBSERVED, r1.getValue("drawer01").outcome)
    assertFalse(r1.getValue("drawer01").eligible)
    assertNull(r1.getValue("drawer01").tally)
    assertEquals(0, store.get("drawer01")!!.evidence.missedCount)

    // A scan of the drawer (the name in another case): now it is the open-area entry that is left alone.
    val drawer = session(container = "Desk Drawer", id = "scan0003")
    for (i in 0 until 4) {
      val checks = drawer.evidence(keyframe(i, 200_000L + 500L * i, pose), geom, emptySet()) { _, _ -> 0.1f }
      assertEquals(listOf("drawer01"), checks.map { it.entryId })
    }
    val r2 = drawer.finish(210_000L).results.associateBy { it.entry.id }
    assertEquals(ScanConfig.OUTCOME_MISSED, r2.getValue("drawer01").outcome)
    assertEquals(ScanConfig.OUTCOME_NOT_OBSERVED, r2.getValue("open0001").outcome)
    assertEquals(1, store.get("open0001")!!.evidence.missedCount)
    assertEquals(1, store.get("open0001")!!.evidence.notObservedCount)
    assertEquals("not seen on ${InventoryText.day(110_000L)}, not checked since", InventoryText.evidence(store.get("open0001")!!, 300_000L, store.lastScanId))

    // A passive window is of the open area: the drawer is never eligible in it.
    val passive = session(mode = ScanSession.Mode.PASSIVE, id = "win00001")
    assertTrue(passive.eligible(store.get("open0001")!!))
    assertFalse(passive.eligible(store.get("drawer01")!!))
  }

  @Test
  fun aScanGivesEveryEntryOfItsMapOneOutcomeAndLogsIt() {
    entry("seen0001", shelf)
    entry("far00001", Vec3(shelf.x + 6.0, shelf.y, shelf.z + 4.0), label = "vase", emb = embedding(2))
    entry("other001", shelf, mapId = "mapOTHER")
    val before = logLines().size
    val s = session()
    for (i in 0 until 3) {
      val pose = poseLookingAt(shelf, side = 0.05 * i)
      val kf = keyframe(i, 100_000L + 500L * i, pose)
      val m = s.merge(kf, listOf(observe(shelf, pose, "cup", embedding(0))!!))
      s.evidence(kf, ScanFixtures.geometry(pose, ScanFixtures.flatDepth(1.0)), m.matchedIds) { _, _ -> throw AssertionError("nothing to compare") }
    }
    val sum = s.finish(110_000L)
    assertEquals(setOf("seen0001", "far00001"), sum.results.map { it.entry.id }.toSet())
    assertEquals(1, sum.seenAgain)
    assertEquals(1, sum.notObserved)
    assertEquals(0, sum.newCount)
    // The entry of another map is not touched at all.
    assertNull(store.get("other001")!!.evidence.lastNotObservedMs)
    assertEquals(1, store.get("other001")!!.evidence.seenCount)

    // The log: an `update` line with the whole entry for the one the scan sighted, an `evidence`
    // line for the one it only judged.
    val lines = logLines().drop(before)
    assertEquals(listOf("update", "evidence"), lines.map { it.getString("op") })
    val updated = InventoryEntry.fromJson(lines[0].getJSONObject("entry"))
    assertEquals("seen0001", updated.id)
    assertEquals(4, updated.sightingCount)
    assertEquals(2, updated.evidence.seenCount)
    assertEquals("far00001", lines[1].getString("id"))
    assertEquals(setOf("t", "session", "op", "id", "lastSeenMs", "evidence", "source", "sourceId"), lines[1].keys().asSequence().toSet())
    assertEquals(ScanConfig.OUTCOME_NOT_OBSERVED, lines[1].getJSONObject("evidence").getString("lastOutcome"))
    assertEquals(ScanConfig.SOURCE_SCAN, lines[1].getString("source"))
    assertEquals("scan0002", lines[1].getString("sourceId"))
    // And inventory.json on disk agrees with the memory.
    val reloaded = InventoryStore(dir, "sess0002")
    assertEquals(store.snapshot().map { it.toJson().toString() }, reloaded.snapshot().map { it.toJson().toString() })
    assertEquals("scan0002", reloaded.lastScanId)
    assertEquals("not checked since ${InventoryText.day(1_000L)}", InventoryText.evidence(reloaded.get("far00001")!!, 300_000L, reloaded.lastScanId))
  }

  @Test
  fun aPassiveWindowRecordsOnlySeenAndMissed() {
    entry("gone0001", shelf)
    entry("else0001", Vec3(shelf.x + 6.0, shelf.y, shelf.z + 4.0), label = "vase", emb = embedding(2))
    val before = logLines().size
    val p = session(mode = ScanSession.Mode.PASSIVE, id = "win00001")
    assertEquals(0, p.talliedCount)
    for (i in 0 until 3) {
      val pose = poseLookingAt(shelf, side = 0.05 * i)
      val kf = keyframe(i, 100_000L + 2_000L * i, pose)
      val m = p.merge(kf, emptyList())
      p.evidence(kf, ScanFixtures.geometry(pose, ScanFixtures.flatDepth(1.0)), m.matchedIds) { _, _ -> 0.2f }
    }
    assertEquals(1, p.talliedCount)
    val sum = p.finish(160_000L)
    assertEquals(listOf("gone0001"), sum.results.map { it.entry.id })
    assertEquals(ScanConfig.OUTCOME_MISSED, sum.results.single().outcome)
    val g = store.get("gone0001")!!
    assertEquals(ScanConfig.SOURCE_PASSIVE, g.evidence.lastSource)
    assertEquals("win00001", g.evidence.lastSourceId)
    assertEquals("looked for 40s ago, not seen", InventoryText.evidence(g, 200_000L, store.lastScanId))
    // The entry the window never saw keeps its last outcome: a window does not say "not observed".
    val e = store.get("else0001")!!
    assertEquals(ScanConfig.OUTCOME_SEEN, e.evidence.lastOutcome)
    assertEquals(0, e.evidence.notObservedCount)
    assertEquals(listOf("evidence"), logLines().drop(before).map { it.getString("op") })
    assertNull(store.lastScanId) // a window is not a scan
  }

  @Test
  fun aNewEntryThatLooksLikeAMissedOneIsAMovedCandidate() {
    val mugLook = embedding(0)
    entry("mug00001", shelf, label = "cup", emb = mugLook)
    entry("book0001", Vec3(shelf.x, shelf.y - 0.3, shelf.z), label = "book", emb = embedding(1))
    val s = session()
    val table = Vec3(shelf.x + 3.0, shelf.y - 0.2, shelf.z + 2.0)
    val lowTable = Vec3(table.x, table.y - 0.35, table.z)
    for (i in 0 until 4) {
      // Looking at the shelf: the cup and the book are gone (their places look different).
      val pose = poseLookingAt(shelf, back = 1.3, side = 0.05 * i)
      val kf = keyframe(i, 100_000L + 500L * i, pose)
      s.evidence(kf, ScanFixtures.geometry(pose, ScanFixtures.flatDepth(1.3)), s.merge(kf, emptyList()).matchedIds) { _, _ -> 0.25f }
    }
    for (i in 4 until 7) {
      // Looking at the table: a cup that looks like the old one (0.85), and a "cup" that looks
      // like the old book (same look, other label: not a candidate).
      val pose = poseLookingAt(table, back = 1.3, side = 0.05 * i)
      val kf = keyframe(i, 100_000L + 500L * i, pose, room = "room2")
      val obs = listOf(observe(table, pose, "cup", embedding(0, 0.85))!!, observe(lowTable, pose, "cup", embedding(1, 0.95), score = 0.7f)!!)
      val m = s.merge(kf, obs)
      s.evidence(kf, ScanFixtures.geometry(pose, ScanFixtures.flatDepth(1.3)), m.matchedIds) { _, _ -> 0.25f }
    }
    val sum = s.finish(110_000L)
    assertEquals(2, sum.newCount)
    assertEquals(2, sum.missed)
    val moved = sum.moved.single()
    assertEquals("mug00001", moved.oldEntry.id)
    assertEquals(0.85f, moved.similarity, 1e-5f)
    val fresh = store.get(moved.newEntry.id)!!
    assertEquals("mug00001", fresh.movedFrom!!.entryId)
    assertEquals("room1", fresh.movedFrom!!.roomId)
    assertEquals("scan0002", fresh.movedFrom!!.scanId)
    assertEquals("may be cup, last seen in Bedroom", InventoryText.movedHint(fresh) { if (it == "room1") "Bedroom" else null })
    assertEquals("room2", fresh.roomId)
    // The old entry is neither moved nor deleted.
    assertNotNull(store.get("mug00001"))
    assertEquals(shelf.x, store.get("mug00001")!!.position.x, 0.01)
    // The other new entry has no candidate.
    val otherNew = store.snapshot().single { it.createdScan == "scan0002" && it.id != fresh.id }
    assertNull(otherNew.movedFrom)
    // The `update` line of the new entry carries the hint.
    val line = logLines().last { it.getString("op") == "update" && it.getJSONObject("entry").getString("id") == fresh.id }
    assertEquals("mug00001", line.getJSONObject("entry").getJSONObject("movedFrom").getString("entry"))
  }

  @Test
  fun aSimilarityBelowTheMovedThresholdIsNoCandidate() {
    entry("mug00001", shelf, label = "cup", emb = embedding(0))
    val s = session()
    val table = Vec3(shelf.x + 3.0, shelf.y - 0.2, shelf.z + 2.0)
    for (i in 0 until 4) {
      val pose = poseLookingAt(shelf, back = 1.3, side = 0.05 * i)
      val kf = keyframe(i, 100_000L + 500L * i, pose)
      s.evidence(kf, ScanFixtures.geometry(pose, ScanFixtures.flatDepth(1.3)), s.merge(kf, emptyList()).matchedIds) { _, _ -> 0.25f }
    }
    for (i in 4 until 6) {
      val pose = poseLookingAt(table, back = 1.3)
      s.merge(keyframe(i, 100_000L + 500L * i, pose), listOf(observe(table, pose, "cup", embedding(0, 0.65))!!))
    }
    val sum = s.finish(110_000L)
    assertEquals(1, sum.newCount)
    assertEquals(1, sum.missed)
    assertTrue(sum.moved.isEmpty())
  }

  @Test
  fun aConfirmedVerificationCountsAsSeen() {
    val e = entry("mug00001", shelf)
    val v = e.withOutcome(ScanConfig.OUTCOME_SEEN, 500_000L, ScanConfig.SOURCE_VERIFY, "query001")
    assertEquals(2, v.evidence.seenCount)
    assertEquals(500_000L, v.lastSeenMs)
    assertEquals(ScanConfig.OUTCOME_SEEN, v.evidence.lastOutcome)
    assertEquals(500_000L, v.evidence.lastOutcomeMs)
    assertEquals(ScanConfig.SOURCE_VERIFY, v.evidence.lastSource)
    assertEquals("query001", v.evidence.lastSourceId)
    // An outcome never moves "last seen" backwards, and the other outcomes do not touch it.
    assertEquals(500_000L, v.withOutcome(ScanConfig.OUTCOME_SEEN, 400_000L, ScanConfig.SOURCE_PASSIVE, "w").lastSeenMs)
    assertEquals(500_000L, v.withOutcome(ScanConfig.OUTCOME_MISSED, 900_000L, ScanConfig.SOURCE_SCAN, "s").lastSeenMs)
    assertEquals(e.position, v.position)
  }
}
