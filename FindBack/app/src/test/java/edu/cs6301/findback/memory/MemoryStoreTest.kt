package edu.cs6301.findback.memory

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

/** records.json keeps the active records; records_log.jsonl keeps every change (v0.4, S5 and S7). */
class MemoryStoreTest {
  private fun legacyText() = javaClass.classLoader!!.getResource("legacy_records_v02.json")!!.readText()

  private fun base(): ObjectRecord = ObjectRecord.fromJson(JSONArray(legacyText()).getJSONObject(0))

  private fun fresh(id: String, name: String, t: Long): ObjectRecord {
    val l = Lineage.first(id, t)
    return base().copy(
      id = id, name = name, lastSeenMs = t, mobility = "some",
      lineageId = l.lineageId, supersedes = l.supersedes, moveCount = l.moveCount, lineageStartMs = l.lineageStartMs,
    )
  }

  private fun successor(old: ObjectRecord, id: String, t: Long): ObjectRecord {
    val l = Lineage.successor(old)
    return old.copy(
      id = id, lastSeenMs = t,
      lineageId = l.lineageId, supersedes = l.supersedes, moveCount = l.moveCount, lineageStartMs = l.lineageStartMs,
    )
  }

  private fun logLines(dir: File): List<JSONObject> =
    File(dir, MemoryStore.RECORDS_LOG).let { f ->
      if (!f.exists()) emptyList() else f.readLines().filter { it.isNotBlank() }.map { JSONObject(it) }
    }

  /** Replays the log the way the analysis does: the active ids after every line. */
  private fun replay(lines: List<JSONObject>): Set<String> {
    val active = LinkedHashSet<String>()
    for (l in lines) {
      val rec = l.getJSONObject("record")
      when (l.getString("op")) {
        "add" -> active.add(rec.getString("id"))
        "rename" -> assertTrue(rec.getString("id") in active)
        "delete" -> active.remove(rec.getString("id"))
        "supersede" -> {
          active.remove(rec.getString("supersedes"))
          active.add(rec.getString("id"))
        }
        else -> throw AssertionError("unknown op " + l.getString("op"))
      }
    }
    return active
  }

  @Test
  fun everyChangeIsLoggedWithTheFullRecord() {
    val dir = Files.createTempDirectory("memstore").toFile()
    val s = MemoryStore(dir, "sess0001")
    assertFalse(File(dir, MemoryStore.RECORDS_LOG).exists()) // nothing to log yet
    val keys = fresh("keys0001", "keys", 1_000L)
    val mug = fresh("mug00001", "mug", 2_000L)
    s.add(keys)
    s.add(mug)
    s.rename("mug00001", "blue mug")
    s.delete("keys0001")

    val lines = logLines(dir)
    assertEquals(listOf("add", "add", "rename", "delete"), lines.map { it.getString("op") })
    for (l in lines) {
      assertEquals("sess0001", l.getString("session"))
      assertTrue(l.getLong("t") > 0)
      val rec = ObjectRecord.fromJson(l.getJSONObject("record"))
      assertEquals(1024, rec.embedding.size) // the embedding is in the log, not only in records.json
      assertEquals(setOf("t", "session", "op", "record"), l.keys().asSequence().toSet())
    }
    assertEquals("blue mug", lines[2].getJSONObject("record").getString("name"))
    assertEquals("keys0001", lines[3].getJSONObject("record").getString("id"))
    assertEquals(setOf("mug00001"), replay(lines))
    assertEquals(listOf("mug00001"), s.snapshot().map { it.id })
    // A rename or delete of an unknown id changes nothing and logs nothing.
    s.rename("nope", "x")
    s.delete("nope")
    assertEquals(4, logLines(dir).size)
  }

  @Test
  fun supersedeReplacesTheOldRecordInOneStep() {
    val dir = Files.createTempDirectory("memstore").toFile()
    val s = MemoryStore(dir, "sess0001")
    val keys1 = fresh("keys0001", "keys", 1_000L)
    val mug = fresh("mug00001", "mug", 2_000L)
    s.add(keys1)
    s.add(mug)
    val keys2 = successor(keys1, "keys0002", 60_000L)
    assertEquals("keys0001", s.supersede("keys0001", keys2)?.id)
    val keys3 = successor(keys2, "keys0003", 120_000L)
    assertEquals("keys0002", s.supersede("keys0002", keys3)?.id)

    // Active list: the old records are gone, the other item is untouched.
    assertEquals(listOf("mug00001", "keys0003"), s.snapshot().map { it.id })
    assertNull(s.get("keys0001"))
    val active = s.get("keys0003")!!
    assertEquals("keys0001", active.lineageId)
    assertEquals("keys0002", active.supersedes)
    assertEquals(2, active.moveCount)
    assertEquals(1_000L, active.lineageStartMs)

    // The same after a restart, from records.json alone.
    val again = MemoryStore(dir, "sess0002")
    assertEquals(listOf("mug00001", "keys0003"), again.snapshot().map { it.id })
    assertEquals(2, again.get("keys0003")!!.moveCount)

    // The log still has every record of the lineage, with embeddings.
    val lines = logLines(dir)
    assertEquals(listOf("add", "add", "supersede", "supersede"), lines.map { it.getString("op") })
    assertEquals("keys0002", lines[2].getJSONObject("record").getString("id"))
    assertEquals("keys0001", lines[2].getJSONObject("record").getString("supersedes"))
    assertEquals(1, lines[2].getJSONObject("record").getInt("moveCount"))
    assertEquals(setOf("mug00001", "keys0003"), replay(lines))
    val lineage = lines.map { ObjectRecord.fromJson(it.getJSONObject("record")) }.filter { it.lineageId == "keys0001" }
    assertEquals(listOf("keys0001", "keys0002", "keys0003"), lineage.map { it.id })
    assertTrue(lineage.all { it.embedding.size == 1024 })
    // Restarting did not add lines (the log existed).
    assertEquals(4, logLines(dir).size)
  }

  @Test
  fun supersedingAMissingRecordStillAddsTheNewOne() {
    val dir = Files.createTempDirectory("memstore").toFile()
    val s = MemoryStore(dir, "sess0001")
    val old = fresh("gone0001", "gone", 1_000L)
    assertNull(s.supersede("gone0001", successor(old, "new00001", 2_000L)))
    assertEquals(listOf("new00001"), s.snapshot().map { it.id })
  }

  @Test
  fun recordsFromBeforeTheLogAreBackfilledOnce() {
    val dir = Files.createTempDirectory("memstore").toFile()
    File(dir, "records.json").writeText(legacyText()) // a v0.2 file, as on the phone
    val s = MemoryStore(dir, "sess0001")
    assertEquals(3, s.snapshot().size)
    var lines = logLines(dir)
    assertEquals(listOf("add", "add", "add"), lines.map { it.getString("op") })
    assertTrue(lines.all { it.getBoolean("backfill") })
    // Reconstructed as of the registration: the record's own time and session, not this start's.
    for (l in lines) {
      assertEquals(l.getJSONObject("record").getLong("lastSeenMs"), l.getLong("t"))
      assertEquals("ff5fbb39", l.getString("session"))
    }
    assertEquals(listOf("62907d85", "e7be2592", "1b328d30"), lines.map { it.getJSONObject("record").getString("id") })
    // The old file itself is not rewritten by loading.
    assertEquals(legacyText(), File(dir, "records.json").readText())

    // An old record can be superseded; a second start does not backfill again.
    val old = s.get("1b328d30")!!
    val moved = successor(old, "moved001", old.lastSeenMs + 600_000L)
    assertNotNull(s.supersede(old.id, moved))
    val s2 = MemoryStore(dir, "sess0002")
    assertEquals(listOf("62907d85", "e7be2592", "moved001"), s2.snapshot().map { it.id })
    assertEquals("1b328d30", s2.get("moved001")!!.lineageId)
    lines = logLines(dir)
    assertEquals(4, lines.size)
    assertFalse(lines[3].has("backfill"))
    assertEquals("sess0001", lines[3].getString("session"))
    assertEquals(setOf("62907d85", "e7be2592", "moved001"), replay(lines))
  }

  @Test
  fun newMapEndsEveryActiveRecordInTheLog() {
    val dir = Files.createTempDirectory("memstore").toFile()
    val s = MemoryStore(dir, "sess0001")
    s.add(fresh("keys0001", "keys", 1_000L))
    s.add(fresh("mug00001", "mug", 2_000L))
    val archive = File(dir, "archive/20261116-120000")
    assertTrue(s.archiveTo(archive))
    assertFalse(File(dir, "records.json").exists())
    assertEquals(2, JSONArray(File(archive, "records.json").readText()).length())
    val lines = logLines(dir)
    assertEquals(listOf("add", "add", "delete", "delete"), lines.map { it.getString("op") })
    assertEquals("map_archived", lines[2].getString("reason"))
    assertEquals("map_archived", lines[3].getString("reason"))
    assertTrue(replay(lines).isEmpty())
    // Nothing is written after the archive (the activity restarts with a new store).
    s.add(fresh("late0001", "late", 3_000L))
    assertEquals(4, logLines(dir).size)
    assertFalse(File(dir, "records.json").exists())
    // The fresh store starts empty and keeps appending to the same log.
    val next = MemoryStore(dir, "sess0002")
    assertTrue(next.snapshot().isEmpty())
    next.add(fresh("new00001", "new", 4_000L))
    assertEquals(setOf("new00001"), replay(logLines(dir)))
  }

  @Test
  fun saveTextWritesUnderTheRoot() {
    val dir = Files.createTempDirectory("memstore").toFile()
    val s = MemoryStore(dir, "sess0001")
    val n = s.saveText("verify/q1_1_emb.json", "{\"a\":1}")
    assertEquals(7L, n)
    assertEquals("{\"a\":1}", File(dir, "verify/q1_1_emb.json").readText())
  }
}
