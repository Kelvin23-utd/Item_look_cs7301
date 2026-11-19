package edu.cs6301.findback.scan

import edu.cs6301.findback.map.Vec3
import edu.cs6301.findback.map.distance
import edu.cs6301.findback.scan.ScanFixtures.embedding
import edu.cs6301.findback.scan.ScanFixtures.keyframe
import edu.cs6301.findback.scan.ScanFixtures.observe
import edu.cs6301.findback.scan.ScanFixtures.poseLookingAt
import java.io.File
import java.nio.file.Files
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * v0.8 study logging in the scan session (docs/PLAN_v0.8_study_logging.md, rules 4 and 6): the
 * baseline matcher, the entries' existing fields and the existing log lines are the same with study
 * logging on or off, and a later scan's merge never changes an entry's `confirmed`.
 */
class StudySessionTest {
  private val centre = Vec3(0.4, 0.9, -1.1)
  private val cupAt = Vec3(centre.x - 0.3, centre.y, centre.z)
  private val bookAt = Vec3(centre.x + 0.35, centre.y, centre.z + 0.05)
  private val bottleAt = Vec3(centre.x + 0.05, centre.y + 0.1, centre.z)

  private fun session(store: InventoryStore, id: String, logging: Boolean): ScanSession {
    var n = 0
    return ScanSession(id, ScanSession.Mode.SCAN, "map00001", null, "effdet0", "sess0001", "mobilenet_v3_small", 1_000L, store, studyLogging = logging) {
      "$id-e%03d".format(n++)
    }
  }

  private fun older(id: String, at: Vec3, label: String, emb: FloatArray): InventoryEntry {
    val pose = poseLookingAt(at)
    return Merger.newEntry(id, observe(at, pose, label, emb)!!, keyframe(0, 500L, pose), "map00001", null, "scanOLD", "effdet0", "sessOLD", "mobilenet_v3_small")
  }

  private fun withId(o: Observation, id: String?) =
    Observation(o.label, o.score, o.box, o.mapPoint, o.worldPoint, o.depthM, o.distanceM, o.sizeM, o.embedding, o.cropWidthPx, o.cropHeightPx, o.crop, id)

  /** One keyframe of the synthetic walk: the detections a perfect detector would give. */
  private fun detections(i: Int, pose: CameraPose): List<Observation> {
    val out = ArrayList<Observation>()
    // The older cup, every keyframe, with a score that varies (the best crop changes now and then).
    out += observe(cupAt, pose, "cup", embedding(0, 0.9 + 0.01 * (i % 5)), sizeM = 0.09, score = 0.5f + 0.04f * (i % 6))!!
    // A new bottle in keyframes 0 to 9; the detector calls it a vase twice, the look stays.
    if (i < 10) out += observe(bottleAt, pose, if (i == 3 || i == 7) "vase" else "bottle", embedding(4, 0.95), sizeM = 0.14, score = 0.7f)!!
    // Something seen once: stays tentative and is dropped.
    if (i == 4) out += observe(Vec3(centre.x + 0.2, centre.y - 0.25, centre.z), pose, "mouse", embedding(5), sizeM = 0.08)!!
    // A remote in keyframes 8, 9 and 10: keyframe 9 comes 200 ms after 8 (still tentative), 10 promotes it.
    if (i in 8..10) out += observe(Vec3(centre.x + 0.3, centre.y - 0.3, centre.z), pose, "remote", embedding(7), sizeM = 0.15)!!
    // A keyboard in keyframes 10 and 11 (450 ms apart: promoted).
    if (i == 10 || i == 11) out += observe(Vec3(centre.x - 0.05, centre.y - 0.3, centre.z + 0.1), pose, "keyboard", embedding(6), sizeM = 0.4)!!
    // Two cups 12 cm apart in keyframe 12: one keyframe never gives one entry two detections.
    if (i == 12) out += observe(Vec3(cupAt.x + 0.12, cupAt.y, cupAt.z), pose, "cup", embedding(0, 0.9), sizeM = 0.09, score = 0.45f)!!
    return out
  }

  private class Run(val store: InventoryStore, val dir: File, val trace: MutableList<String>, val summary: SessionSummary, val results: List<ObservationResult>)

  private fun run(logging: Boolean): Run {
    val dir = Files.createTempDirectory("study").toFile()
    val store = InventoryStore(dir, "sess0001")
    store.add(older("old00001", cupAt, "cup", embedding(0)))
    store.add(older("old00002", bookAt, "book", embedding(1)))
    val s = session(store, "scan0001", logging)
    val trace = ArrayList<String>()
    val all = ArrayList<ObservationResult>()
    for (i in 0 until 14) {
      val pose = poseLookingAt(centre, back = 1.4, side = -0.4 + 0.06 * i)
      val kf = keyframe(i, if (i == 9) 13_800L else 10_000L + 450L * i, pose)
      val obs = detections(i, pose).mapIndexed { j, o -> withId(o, if (logging) StudyConfig.obsId("scan0001", i, j) else null) }
      val m = s.merge(kf, obs)
      all += m.results
      for (r in m.results) trace += "kf$i ${r.observation.label} ${r.kind} ${r.entryId} ${r.similarity} ${r.distanceM} ${r.byLabel}"
      trace += "kf$i matched ${m.matchedIds} added ${m.added.map { canon(strip(it.toJson())) }} crops ${m.cropUpdates.map { it.first }}"
      val checks = s.evidence(kf, ScanFixtures.geometry(pose, ScanFixtures.flatDepth(2.0)), m.matchedIds) { e, _ -> if (e.id == "old00002") 0.31f else 0.62f }
      for (c in checks) trace += "kf$i check ${c.entryId} ${c.view.visibility} ${c.view.u} ${c.view.v} ${c.similarity}"
    }
    val retained = if (logging) setOf(StudyConfig.obsId("scan0001", 1, 1), StudyConfig.obsId("scan0001", 5, 1)) else emptySet()
    val sum = s.finish(20_000L, retained)
    for (r in sum.results) trace += "end ${r.entry.id} ${r.outcome} ${r.eligible} ${r.isNew} ${canon(strip(r.entry.toJson()))}"
    trace += "summary ${sum.newCount} ${sum.seenAgain} ${sum.missed} ${sum.notObserved} ${sum.tentativeDropped} ${sum.droppedTentative} ${sum.createdIds}"
    return Run(store, dir, trace, sum, all)
  }

  /** The entry JSON without the v0.8 fields. */
  private fun strip(o: JSONObject): JSONObject {
    val c = JSONObject(o.toString())
    c.remove("confirmed")
    c.remove("history")
    return c
  }

  /** JSON with sorted keys, so two objects compare by content. */
  private fun canon(v: Any?): String =
    when (v) {
      is JSONObject -> v.keys().asSequence().sorted().joinToString(",", "{", "}") { "\"$it\":" + canon(v.get(it)) }
      is JSONArray -> (0 until v.length()).joinToString(",", "[", "]") { canon(v.get(it)) }
      is String -> JSONObject.quote(v)
      null -> "null"
      else -> v.toString()
    }

  /** An inventory_log.jsonl line without its time and the v0.8 additions. */
  private fun baselineLine(line: String): String {
    val o = JSONObject(line)
    o.remove("t")
    o.remove("historyItem")
    o.optJSONObject("entry")?.let { o.put("entry", strip(it)) }
    return canon(o)
  }

  @Test
  fun studyLoggingChangesNoDecisionNoExistingFieldAndNoExistingLine() {
    val off = run(logging = false)
    val on = run(logging = true)
    assertTrue("the walk exercises every kind", off.trace.size > 60)
    assertEquals(off.trace, on.trace)
    assertEquals(
      setOf(ScanSession.KIND_NEW, ScanSession.KIND_TENTATIVE, ScanSession.KIND_PROMOTED, ScanSession.KIND_KNOWN),
      off.results.map { it.kind }.toSet(),
    )
    assertEquals(2, off.summary.tentativeDropped) // the mouse and the second cup of keyframe 12
    assertEquals(2, off.summary.droppedTentative.size)
    assertEquals(1, off.summary.missed)

    // Entries: everything but the new fields is the same; with logging off they do not exist at all.
    val a = off.store.snapshot().sortedBy { it.id }
    val b = on.store.snapshot().sortedBy { it.id }
    assertEquals(a.map { canon(strip(it.toJson())) }, b.map { canon(strip(it.toJson())) })
    assertEquals(a.map { canon(JSONObject(it.toJson().toString())) }, a.map { canon(strip(it.toJson())) }) // nothing to strip when off
    for (e in a) {
      assertNull(e.confirmed)
      assertTrue(e.history.isEmpty())
      assertFalse(e.toJson().has("confirmed"))
      assertFalse(e.toJson().has("history"))
    }
    assertTrue(b.all { it.history.size == 1 })

    // inventory.json and inventory_log.jsonl: the same, line for line, apart from the additions.
    val logOff = File(off.dir, InventoryStore.LOG_NAME).readLines()
    val logOn = File(on.dir, InventoryStore.LOG_NAME).readLines()
    assertEquals(logOff.size, logOn.size)
    assertEquals(logOff.map { baselineLine(it) }, logOn.map { baselineLine(it) })
    assertFalse(logOff.any { it.contains("historyItem") || it.contains("\"confirmed\"") || it.contains("\"history\"") })
    assertTrue(logOn.any { JSONObject(it).has("historyItem") }) // the missed book has an evidence line
    fun entriesOf(dir: File) =
      JSONObject(File(dir, InventoryStore.FILE_NAME).readText()).getJSONArray("entries").let { arr ->
        (0 until arr.length()).map { canon(strip(arr.getJSONObject(it))) }.sorted()
      }
    assertEquals(entriesOf(off.dir), entriesOf(on.dir))

    // The study log's extras: candidates only with logging on; byLabel in both.
    assertTrue(off.results.all { it.candidates == null })
    assertTrue(on.results.all { it.candidates != null })
    assertTrue(on.results.filter { it.kind == ScanSession.KIND_KNOWN }.all { it.byLabel != null })
  }

  @Test
  fun aLaterScanMergesIntoAnEarlierEntryAndConfirmedStaysAsItWas() {
    val dir = Files.createTempDirectory("study").toFile()
    val store = InventoryStore(dir, "sess0001")
    // Scan 1 creates the cup from three keyframes; a book is created too.
    val s1 = session(store, "scan0001", logging = true)
    for (i in 0 until 3) {
      val pose = poseLookingAt(centre, back = 1.2, side = -0.1 + 0.1 * i)
      val obs =
        listOf(observe(cupAt, pose, "cup", embedding(0), sizeM = 0.09)!!, observe(bookAt, pose, "book", embedding(1), sizeM = 0.15)!!)
          .mapIndexed { j, o -> withId(o, StudyConfig.obsId("scan0001", i, j)) }
      s1.merge(keyframe(i, 1_000L + 500L * i, pose), obs)
    }
    val retained = setOf("scan0001-0-0", "scan0001-2-0", "scan0001-1-1")
    s1.finish(5_000L, retained)
    val cup = store.snapshot().single { it.label == "cup" }
    val book = store.snapshot().single { it.label == "book" }
    val c1 = cup.confirmed!!
    assertEquals(StudyConfig.SOURCE_CREATED, c1.source)
    assertEquals("scan0001", c1.scan)
    assertEquals(cup.position, c1.mapPosition)
    assertEquals(cup.viewPosition, c1.viewPosition)
    assertEquals(cup.roomId, c1.roomId)
    assertEquals(2_000L, c1.tMs) // the last keyframe that saw it
    assertEquals(listOf("scan0001-0-0", "scan0001-2-0"), c1.refs) // its linked observations that kept a view
    assertEquals(listOf("scan0001-1-1"), book.confirmed!!.refs)
    val h1 = cup.history.single()
    assertEquals(StudyConfig.LINK_CREATED, h1.link)
    assertEquals(ScanConfig.OUTCOME_SEEN, h1.outcome)
    assertEquals(listOf("scan0001-0-0", "scan0001-1-0", "scan0001-2-0"), h1.obs)
    assertEquals(3, h1.sightings)
    assertTrue(distance(h1.position!!, cupAt) < 0.01)
    assertEquals(5_000L, h1.tMs)

    // Scan 2, another day: the cup is 6 cm to the side; the matcher merges it into the old entry.
    // The book is not looked at.
    val moved = Vec3(cupAt.x + 0.06, cupAt.y, cupAt.z)
    val s2 = session(store, "scan0002", logging = true)
    val kinds = ArrayList<String>()
    for (i in 0 until 4) {
      val pose = poseLookingAt(moved, back = 0.9, side = 0.05 * i)
      val o = withId(observe(moved, pose, "cup", embedding(0, 0.93), sizeM = 0.09)!!, StudyConfig.obsId("scan0002", i, 0))
      kinds += s2.merge(keyframe(i, 90_000L + 500L * i, pose), listOf(o)).results.single().kind
    }
    assertEquals(List(4) { ScanSession.KIND_KNOWN }, kinds)
    val during = store.get(cup.id)!!
    assertTrue("the baseline moved the entry's position", distance(during.position, cup.position) > 0.02)
    assertEquals(c1, during.confirmed) // a merge is not a confirmation
    s2.finish(100_000L)

    val after = store.get(cup.id)!!
    assertEquals(c1, after.confirmed)
    assertEquals(2, after.history.size)
    assertEquals(h1, after.history[0])
    val h2 = after.history[1]
    assertEquals("scan0002", h2.scan)
    assertEquals(StudyConfig.LINK_BASELINE_UNCONFIRMED, h2.link)
    assertEquals(ScanConfig.OUTCOME_SEEN, h2.outcome)
    assertEquals((0 until 4).map { "scan0002-$it-0" }, h2.obs)
    assertTrue("the scan's own sightings only", distance(h2.position!!, moved) < 0.01)
    assertEquals(91_500L, h2.lastObsMs)
    // The book: an item for the scan, with nothing linked to it, and confirmed untouched.
    val bookAfter = store.get(book.id)!!
    assertEquals(book.confirmed, bookAfter.confirmed)
    val hb = bookAfter.history.last()
    assertEquals(ScanConfig.OUTCOME_NOT_OBSERVED, hb.outcome)
    assertTrue(hb.obs.isEmpty())
    assertNull(hb.position)
    assertEquals(0, hb.sightings)

    // The log says the same: the cup's `update` line carries the unchanged `confirmed`, the book's
    // `evidence` line carries its new history item.
    val lines = File(dir, InventoryStore.LOG_NAME).readLines().map { JSONObject(it) }
    val cupLine = lines.last { it.getString("op") == "update" && it.getJSONObject("entry").getString("id") == cup.id }
    assertEquals(canon(InventoryEntry.confirmedJson(c1)), canon(cupLine.getJSONObject("entry").getJSONObject("confirmed")))
    assertEquals(2, cupLine.getJSONObject("entry").getJSONArray("history").length())
    val bookLine = lines.last { it.getString("op") == "evidence" && it.getString("id") == book.id }
    assertEquals("scan0002", bookLine.getJSONObject("historyItem").getString("scan"))
    // Reloading keeps both fields.
    val again = InventoryStore(dir, "sess0002").get(cup.id)!!
    assertEquals(c1, again.confirmed)
    assertEquals(after.history, again.history)
  }

  @Test
  fun theDecisionListsTheNearestEntriesTheMatcherLookedAt() {
    val dir = Files.createTempDirectory("study").toFile()
    val store = InventoryStore(dir, "sess0001")
    store.add(older("near0001", cupAt, "cup", embedding(0)))
    store.add(older("far00001", Vec3(cupAt.x + 0.45, cupAt.y, cupAt.z), "cup", embedding(0)))
    store.add(older("book0001", Vec3(cupAt.x + 0.15, cupAt.y, cupAt.z), "book", embedding(1)))
    store.add(older("farther1", Vec3(cupAt.x + 0.9, cupAt.y, cupAt.z), "cup", embedding(0)))
    val s = session(store, "scan0001", logging = true)
    val pose = poseLookingAt(cupAt, back = 1.3, side = 0.3)
    val r = s.merge(keyframe(0, 1_000L, pose), listOf(observe(cupAt, pose, "cup", embedding(0, 0.8), sizeM = 0.09)!!)).results.single()
    assertEquals(ScanSession.KIND_KNOWN, r.kind)
    assertEquals("near0001", r.entryId)
    assertEquals(true, r.byLabel)
    val c = r.candidates!!
    assertEquals(listOf("near0001", "book0001", "far00001"), c.map { it.entryId }) // nearest three, nearest first
    assertTrue(c[0].withinRadius && c[0].sameLabel)
    assertTrue(c[1].withinRadius && !c[1].sameLabel)
    assertEquals(0.0f, c[1].similarity, 1e-6f) // a book that looks nothing like it: not a match
    assertFalse(c[2].withinRadius) // 0.45 m: beyond the merge radius
    assertEquals(r.distanceM!!, c[0].distanceM, 1e-12)
    assertTrue(c.none { it.tentative })

    // A second observation in the next keyframe sees the tentative entry the first one could not match.
    val other = Vec3(cupAt.x, cupAt.y - 0.3, cupAt.z)
    s.merge(keyframe(1, 1_500L, pose), listOf(observe(other, pose, "mouse", embedding(5), sizeM = 0.08)!!))
    val r2 = s.merge(keyframe(2, 1_600L, pose), listOf(observe(other, pose, "mouse", embedding(5), sizeM = 0.08)!!)).results.single()
    assertEquals(ScanSession.KIND_TENTATIVE, r2.kind)
    assertTrue(r2.candidates!!.first().tentative)
  }

  @Test
  fun oldFilesLoadAndEntriesWithoutTheNewFieldsAreWrittenAsBefore() {
    val e = older("old00001", cupAt, "cup", embedding(0))
    val v07 = e.toJson() // written by v0.7: no confirmed, no history
    assertFalse(v07.has("confirmed"))
    assertFalse(v07.has("history"))
    val back = InventoryEntry.fromJson(JSONObject(v07.toString()))
    assertNull(back.confirmed)
    assertTrue(back.history.isEmpty())
    assertEquals(v07.toString(), back.toJson().toString())
    // An explicit null loads too.
    assertNull(InventoryEntry.fromJson(JSONObject(v07.toString()).put("confirmed", JSONObject.NULL)).confirmed)

    val full =
      e.copy(
        confirmed = ConfirmedState(Vec3(1.0, 2.0, 3.0), null, "room1", 7_000L, null, StudyConfig.SOURCE_USER_VERIFY, listOf("q0000001-x1"), "q0000001", 0.81f),
        history = listOf(HistoryItem("scan0001", 5_000L, "seen", true, Vec3(0.1, 0.2, 0.3), 3, 4_000L, listOf("scan0001-0-0"), StudyConfig.LINK_CREATED)),
      )
    val round = InventoryEntry.fromJson(JSONObject(full.toJson().toString()))
    assertEquals(full.confirmed, round.confirmed)
    assertEquals(full.history, round.history)
    assertEquals(full.toJson().toString(), round.toJson().toString())
  }

  @Test
  fun aUserConfirmationSetsConfirmedAndIsLogged() {
    val dir = Files.createTempDirectory("study").toFile()
    val store = InventoryStore(dir, "sess0001")
    val e = older("old00001", cupAt, "cup", embedding(0))
      .copy(history = listOf(HistoryItem("scan0001", 5_000L, "seen", true, cupAt, 3, 4_000L, emptyList(), StudyConfig.LINK_CREATED)))
    store.add(e)
    val state = ConfirmedState(cupAt, Vec3(0.0, 1.4, 0.0), "room1", 9_000L, null, StudyConfig.SOURCE_USER_VERIFY, listOf("q0000001-x2"), "q0000001", 0.8f)
    val after = store.confirm(e.id, state)!!
    assertEquals(state, after.confirmed)
    assertEquals(e.history, after.history) // a confirmation is not a scan
    assertEquals(e.copy(confirmed = state), after)
    val line = JSONObject(File(dir, InventoryStore.LOG_NAME).readLines().last())
    assertEquals("confirm", line.getString("op"))
    assertEquals(StudyConfig.SOURCE_USER_VERIFY, line.getString("source"))
    assertEquals("q0000001-x2", line.getJSONObject("entry").getJSONObject("confirmed").getJSONArray("refs").getString(0))
    assertNotNull(InventoryStore(dir, "s").get(e.id)!!.confirmed) // saved at once
    assertNull(store.confirm("missing", state))
  }
}
