package edu.cs6301.findback.scan

import edu.cs6301.findback.map.Vec3
import edu.cs6301.findback.memory.ObjectRecord
import edu.cs6301.findback.ml.VerifyWindow
import edu.cs6301.findback.scan.ScanFixtures.embedding
import java.io.File
import java.nio.file.Files
import org.json.JSONObject
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

/** Plan sections 6 and 10: inventory.json and its log, entry to ObjectRecord, and the Items search. */
class InventoryTest {
  private fun full(id: String = "e0000001"): InventoryEntry =
    InventoryEntry(
      id = id,
      name = "grandma's mug",
      nameEdited = true,
      label = "cup",
      labelVotes = linkedMapOf("cup" to 2.4, "bowl" to 0.5),
      mapId = "map00001",
      position = Vec3(0.4, 0.9, -1.1),
      viewPosition = Vec3(0.3, 1.4, -0.2),
      roomId = "room0001",
      container = "desk drawer",
      sizeM = 0.11,
      cropFile = InventoryEntry.cropPath(id),
      embedding = embedding(3, 0.8),
      embeddingModel = "mobilenet_v3_small",
      captureDistance = 0.92f,
      cropWidthPx = 310.5f,
      cropHeightPx = 287.25f,
      imageWidth = 1920,
      imageHeight = 1080,
      bestQuality = 71234.5f,
      bestScore = 0.81f,
      worldPosition = Vec3(1.5, 0.5, 2.0),
      sessionId = "sess0001",
      sightings = listOf(Sighting(1_000L, Vec3(0.4, 0.9, -1.1), 0.11, "cup", 0.81f, "scan0001", 3)),
      sightingCount = 7,
      firstSeenMs = 1_000L,
      lastSeenMs = 5_000L,
      detector = "effdet0",
      createdScan = "scan0001",
      evidence =
        EvidenceState(
          seenCount = 2, missedCount = 1, notObservedCount = 3, lastMissedMs = 4_000L, lastNotObservedMs = null,
          lastOutcome = "missed", lastOutcomeMs = 4_000L, lastSource = "scan", lastSourceId = "scan0002",
        ),
      movedFrom = MovedFrom("e0000000", "cup", "room0002", 0.77f, "scan0002"),
    )

  @Test
  fun entryJsonRoundTripKeepsEveryField() {
    val e = full()
    val back = InventoryEntry.fromJson(JSONObject(e.toJson().toString()))
    assertEquals(e.toJson().toString(), back.toJson().toString())
    assertArrayEquals(e.embedding, back.embedding, 0f)
    // Data-class equality, apart from the embedding array (compared by reference).
    assertEquals(e, back.copy(embedding = e.embedding))
    // The compact form (events.jsonl) has everything but the embedding itself.
    val compact = e.toJson(withEmbedding = false)
    assertFalse(compact.has("embedding"))
    assertEquals(e.embedding.size, compact.getInt("embeddingDim"))
    // Nulls survive as nulls.
    val bare = e.copy(roomId = null, container = null, movedFrom = null, evidence = EvidenceState())
    val bareBack = InventoryEntry.fromJson(JSONObject(bare.toJson().toString()))
    assertNull(bareBack.roomId)
    assertNull(bareBack.container)
    assertNull(bareBack.movedFrom)
    assertNull(bareBack.evidence.lastOutcome)
    assertEquals(EvidenceState(), bareBack.evidence)
  }

  @Test
  fun storeRoundTripRenameDeleteAndTheLog() {
    val dir = Files.createTempDirectory("inventory").toFile()
    val store = InventoryStore(dir, "sess0001")
    store.add(full("e0000001"))
    store.add(full("e0000002").copy(name = "cup", nameEdited = false))
    store.setLastScan("scan0002", 9_000L)
    assertTrue(store.flush())
    assertFalse(store.flush()) // nothing changed since
    assertEquals("mug on the desk", store.rename("e0000002", "mug on the desk")!!.name)
    assertTrue(store.get("e0000002")!!.nameEdited)
    assertNotNull(store.delete("e0000001"))
    assertNull(store.delete("e0000001"))

    val again = InventoryStore(dir, "sess0002")
    assertEquals(listOf("e0000002"), again.snapshot().map { it.id })
    assertEquals("mug on the desk", again.get("e0000002")!!.name)
    assertEquals("scan0002", again.lastScanId)
    assertEquals(9_000L, again.lastScanEndMs)
    val ops = File(dir, InventoryStore.LOG_NAME).readLines().map { JSONObject(it) }
    assertEquals(listOf("add", "add", "rename", "delete"), ops.map { it.getString("op") })
    assertEquals("e0000001", ops[3].getJSONObject("entry").getString("id"))
    assertTrue(ops[0].getJSONObject("entry").has("embedding")) // log lines carry the full entry
    assertEquals("sess0001", ops[0].getString("session"))
  }

  @Test
  fun updateWorksOnTheEntryAsItIsNow() {
    val store = InventoryStore(Files.createTempDirectory("inventory").toFile(), "s")
    store.add(full().copy(name = "cup", nameEdited = false))
    val stale = store.get("e0000001")!!
    store.rename("e0000001", "my cup") // e.g. on the UI thread while a keyframe is processed
    val after = store.update(stale.id) { it.copy(sightingCount = it.sightingCount + 1) }!!
    assertEquals("my cup", after.name)
    assertEquals(stale.sightingCount + 1, after.sightingCount)
    assertNull(store.update("missing") { it })
  }

  @Test
  fun aConfirmedVerificationIsWrittenAsAnEvidenceLine() {
    // What ScanController.onVerifiedSeen does with the store.
    val dir = Files.createTempDirectory("inventory").toFile()
    val store = InventoryStore(dir, "sess0003")
    store.add(full().copy(evidence = EvidenceState(missedCount = 1, lastOutcome = "missed", lastMissedMs = 4_000L)))
    val e = store.update("e0000001") { it.withOutcome(ScanConfig.OUTCOME_SEEN, 9_000L, ScanConfig.SOURCE_VERIFY, "query001") }!!
    assertTrue(store.logEvidence("e0000001"))
    assertEquals(9_000L, e.lastSeenMs)
    assertEquals(1, e.evidence.seenCount)
    assertEquals("seen", InventoryText.evidence(e, 9_000L + 60_000L, null).substringBefore(' '))
    val line = JSONObject(File(dir, InventoryStore.LOG_NAME).readLines().last())
    assertEquals("evidence", line.getString("op"))
    assertEquals(ScanConfig.SOURCE_VERIFY, line.getString("source"))
    assertEquals("query001", line.getString("sourceId"))
    assertEquals(9_000L, line.getLong("lastSeenMs"))
    assertEquals(1, line.getJSONObject("evidence").getInt("missed")) // the earlier miss stays counted
    assertFalse(store.logEvidence("missing"))
  }

  @Test
  fun aCorruptFileIsCopiedAsideAndArchivingMovesTheFile() {
    val dir = Files.createTempDirectory("inventory").toFile()
    File(dir, InventoryStore.FILE_NAME).writeText("{ not json")
    val broken = InventoryStore(dir, "s")
    assertNotNull(broken.loadError)
    assertTrue(broken.snapshot().isEmpty())
    assertTrue(dir.listFiles()!!.any { it.name.startsWith("inventory.corrupt.") })

    val dir2 = Files.createTempDirectory("inventory").toFile()
    val store = InventoryStore(dir2, "s")
    store.add(full())
    val archive = File(dir2, "archive/20261116-120000")
    assertTrue(store.archiveTo(archive))
    assertFalse(File(dir2, InventoryStore.FILE_NAME).exists())
    assertEquals(1, InventoryStore(archive, "s").snapshot().size)
    val last = JSONObject(File(dir2, InventoryStore.LOG_NAME).readLines().last())
    assertEquals("delete", last.getString("op"))
    assertEquals("map_archived", last.getString("reason"))
    // Nothing is written after archiving.
    store.add(full("e0000009"))
    assertFalse(File(dir2, InventoryStore.FILE_NAME).exists())
    // With no inventory (scan never used) archiving writes nothing at all.
    val empty = Files.createTempDirectory("inventory").toFile()
    assertFalse(InventoryStore(empty, "s").archiveTo(File(empty, "archive/x")))
    assertFalse(File(empty, InventoryStore.LOG_NAME).exists())
  }

  @Test
  fun anEntryBecomesTheRecordTheQueryFlowWorksOn() {
    val e = full()
    val r = e.toObjectRecord()
    assertEquals(e.id, r.id)
    assertEquals(e.name, r.name)
    assertEquals("cup", r.rawLabel)
    assertEquals(0.81f, r.rawConfidence, 0f)
    assertEquals(InventoryEntry.BOX_SOURCE, r.boxSource)
    assertEquals(InventoryEntry.HIT_TYPE, r.hitType)
    assertEquals(e.cropFile, r.cropFile)
    assertTrue(e.embedding === r.embedding)
    assertNull(r.refPosition)
    assertEquals("map00001", r.mapId)
    assertArrayEquals(floatArrayOf(0.4f, 0.9f, -1.1f), r.mapPosition!!, 1e-6f)
    assertArrayEquals(floatArrayOf(0.3f, 1.4f, -0.2f), r.viewPosition!!, 1e-6f)
    assertEquals("room0001", r.roomId)
    assertEquals(5_000L, r.lastSeenMs)
    assertEquals(1_000L, r.lineageStartMs)
    assertEquals(e.id, r.lineageId)
    assertEquals(0, r.moveCount)
    assertNull(r.supersedes)
    assertNull(r.mobility)
    // Verification rescales the stored crop with these four numbers: they must be the scan's.
    val w = VerifyWindow.expected(r.captureDistance, r.cropWidthPx, r.cropHeightPx, r.imageWidth, 0.46f, 1920, 1080, 900f, 500f)
    assertEquals(310.5f * 2f, w.width, 1e-3f) // half the distance, twice the size
    assertEquals(287.25f * 2f, w.height, 1e-3f)
    // The record survives records.json's own format too.
    val back = ObjectRecord.fromJson(JSONObject(r.toJson().toString()))
    assertEquals(r.toJson().toString(), back.toJson().toString())
  }

  @Test
  fun searchAndTheSecondLine() {
    val e = full().copy(name = "Grandma's Mug")
    assertTrue(InventoryText.matches(e, "mug"))
    assertTrue(InventoryText.matches(e, " GRAND "))
    assertTrue(InventoryText.matches(e, "cup")) // the winning label
    assertTrue(InventoryText.matches(e, "Bowl")) // any label it was detected as
    assertTrue(InventoryText.matches(e, "drawer")) // its container
    assertTrue(InventoryText.matches(e, ""))
    assertFalse(InventoryText.matches(e, "remote"))
    val rec = e.toObjectRecord().copy(name = "keys", rawLabel = "Home good")
    assertTrue(InventoryText.matches(rec, "KEY"))
    assertTrue(InventoryText.matches(rec, "home"))
    assertFalse(InventoryText.matches(rec, "mug"))

    assertEquals("Bedroom", InventoryText.location(e.copy(container = null), "Bedroom", true))
    assertEquals("in desk drawer, Office", InventoryText.location(e, "Office", true))
    assertEquals("room not named", InventoryText.location(e.copy(container = null), null, true))
    val seen = e.copy(evidence = EvidenceState(seenCount = 1, lastOutcome = "seen"), lastSeenMs = 0L)
    assertEquals("seen 2 d ago", InventoryText.evidence(seen, 2 * 86_400_000L + 5_000L, null))
  }

  @Test
  fun lastSeenLineAndMissNote() {
    val now = 2 * 86_400_000L + 5_000L
    assertEquals("Last seen: Kitchen · 2 d ago", InventoryText.lastSeenLine("Kitchen", 0L, now))
    assertEquals("Last seen: in desk drawer, Office · 2 d ago", InventoryText.lastSeenLine("in desk drawer, Office", 0L, now))
    // A tapped record off the map: the time first, then what its place is.
    assertEquals("Last seen: 5 min ago · photo only", InventoryText.lastSeenLine(null, 0L, 300_000L, "photo only"))
    assertEquals("Last seen: 5 min ago · this session only", InventoryText.lastSeenLine(null, 0L, 300_000L, "this session only"))

    val e = full().copy(lastSeenMs = 0L)
    // `missed` in the last scan: the existing note follows the time.
    assertEquals("looked for in the last scan, not seen", InventoryText.missNote(e, now, "scan0002"))
    assertEquals(
      "Last seen: Bedroom · 2 d ago · looked for in the last scan, not seen",
      InventoryText.lastSeenLine("Bedroom", e.lastSeenMs, now, InventoryText.missNote(e, now, "scan0002")),
    )
    val notObserved = e.copy(evidence = EvidenceState(lastOutcome = "not_observed"))
    assertEquals("not checked since " + InventoryText.day(0L), InventoryText.missNote(notObserved, now, null))
    // `seen` or no outcome yet: nothing to add to the time.
    assertNull(InventoryText.missNote(e.copy(evidence = EvidenceState(seenCount = 1, lastOutcome = "seen")), now, null))
    assertNull(InventoryText.missNote(e.copy(evidence = EvidenceState()), now, null))
  }

  @Test
  fun seenInScanForTheLastScanChip() {
    // full(): one sighting from scan0001, last outcome `missed` by scan0002.
    val e = full()
    assertTrue(InventoryText.seenInScan(e, "scan0001")) // a detection in that scan
    assertFalse(InventoryText.seenInScan(e, "scan0002")) // looked for, not seen
    assertFalse(InventoryText.seenInScan(e, null)) // no finished scan yet
    // `seen` by appearance only (no sighting from that scan), as the last outcome.
    val byLook =
      e.copy(evidence = EvidenceState(seenCount = 1, lastOutcome = "seen", lastOutcomeMs = 9_000L, lastSource = "scan", lastSourceId = "scan0003"))
    assertTrue(InventoryText.seenInScan(byLook, "scan0003"))
    // A later passive window replaced the last outcome: only the history (study logging) still says it.
    val later = byLook.copy(evidence = byLook.evidence.copy(lastSource = "passive", lastSourceId = "pw0001"))
    assertFalse(InventoryText.seenInScan(later, "scan0003"))
    val withHistory = later.copy(history = listOf(HistoryItem("scan0003", 9_000L, "seen", true, null, 0, null, emptyList(), "baseline_unconfirmed")))
    assertTrue(InventoryText.seenInScan(withHistory, "scan0003"))
    // A `seen` outcome from verification is not a scan.
    val verified = e.copy(evidence = EvidenceState(lastOutcome = "seen", lastSource = "verify", lastSourceId = "scan0003"))
    assertFalse(InventoryText.seenInScan(verified, "scan0003"))
  }
}
