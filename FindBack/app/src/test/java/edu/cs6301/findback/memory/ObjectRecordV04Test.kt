package edu.cs6301.findback.memory

import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

/** v0.4 record fields (mobility, lineage) and the lineage rules of "Update location". */
class ObjectRecordV04Test {
  private fun legacy(): JSONArray =
    JSONArray(javaClass.classLoader!!.getResource("legacy_records_v02.json")!!.readText())

  private fun record(id: String, lastSeenMs: Long = 1_000_000L) =
    ObjectRecord.fromJson(legacy().getJSONObject(0)).copy(id = id, lastSeenMs = lastSeenMs).let {
      // copy() keeps the loaded lineage fields; rebuild them as a fresh registration would.
      val l = Lineage.first(id, lastSeenMs)
      it.copy(lineageId = l.lineageId, supersedes = l.supersedes, moveCount = l.moveCount, lineageStartMs = l.lineageStartMs)
    }

  @Test
  fun oldFilesReadAsAFirstRecordOfTheirOwnLineage() {
    val arr = legacy()
    for (i in 0 until arr.length()) {
      val o = arr.getJSONObject(i)
      for (k in listOf("mobility", "lineageId", "supersedes", "moveCount", "lineageStartMs")) assertFalse(k, o.has(k))
      val r = ObjectRecord.fromJson(o)
      assertNull(r.mobility)
      assertEquals(r.id, r.lineageId)
      assertNull(r.supersedes)
      assertEquals(0, r.moveCount)
      assertEquals(r.lastSeenMs, r.lineageStartMs)
    }
  }

  @Test
  fun explicitNullsReadLikeMissingKeys() {
    val o = legacy().getJSONObject(1)
    for (k in listOf("mobility", "lineageId", "supersedes", "moveCount", "lineageStartMs")) o.put(k, JSONObject.NULL)
    val r = ObjectRecord.fromJson(o)
    assertNull(r.mobility)
    assertEquals("e7be2592", r.lineageId)
    assertNull(r.supersedes)
    assertEquals(0, r.moveCount)
    assertEquals(r.lastSeenMs, r.lineageStartMs)
  }

  @Test
  fun legacyRecordIsRewrittenWithTheNewKeys() {
    val r = ObjectRecord.fromJson(legacy().getJSONObject(0))
    val j = JSONObject(r.toJson().toString())
    assertTrue(j.isNull("mobility"))
    assertTrue(j.isNull("supersedes"))
    assertEquals("62907d85", j.getString("lineageId"))
    assertEquals(0, j.getInt("moveCount"))
    assertEquals(r.lastSeenMs, j.getLong("lineageStartMs"))
    // Every older key is still written.
    for (k in legacy().getJSONObject(0).keys()) assertTrue(k, j.has(k))
    for (k in listOf("mapId", "mapPosition", "viewPosition", "roomId")) assertTrue(k, j.has(k))
    val back = ObjectRecord.fromJson(j)
    assertEquals(r.lineageId, back.lineageId)
    assertArrayEquals(r.embedding, back.embedding, 0f)
  }

  @Test
  fun newFieldsRoundTrip() {
    val r =
      ObjectRecord.fromJson(legacy().getJSONObject(2))
        .copy(mobility = "often", lineageId = "aaaa1111", supersedes = "bbbb2222", moveCount = 3, lineageStartMs = 1_794_000_000_123)
    val back = ObjectRecord.fromJson(JSONObject(r.toJson().toString()))
    assertEquals("often", back.mobility)
    assertEquals("aaaa1111", back.lineageId)
    assertEquals("bbbb2222", back.supersedes)
    assertEquals(3, back.moveCount)
    assertEquals(1_794_000_000_123, back.lineageStartMs)
    assertEquals(r.lastSeenMs, back.lastSeenMs)
  }

  @Test
  fun ordinaryRegistrationStartsItsOwnLineage() {
    val l = Lineage.first("abcd1234", 5_000L)
    assertEquals(LineageFields("abcd1234", null, 0, 5_000L), l)
    val r = record("abcd1234", 5_000L)
    assertEquals("abcd1234", r.lineageId)
    assertEquals(0, r.moveCount)
    assertEquals(0.0, Lineage.ageMinutes(r, 5_000L), 1e-9)
  }

  @Test
  fun successorKeepsTheLineageAndCountsMoves() {
    val first = record("first001", 1_000_000L)
    val l1 = Lineage.successor(first)
    assertEquals(LineageFields("first001", "first001", 1, 1_000_000L), l1)
    val second =
      record("second02", 4_000_000L)
        .copy(lineageId = l1.lineageId, supersedes = l1.supersedes, moveCount = l1.moveCount, lineageStartMs = l1.lineageStartMs)
    val l2 = Lineage.successor(second)
    assertEquals(LineageFields("first001", "second02", 2, 1_000_000L), l2)
    // Lineage age counts from the first registration, not from the latest move.
    assertEquals(100.0, Lineage.ageMinutes(second, 7_000_000L), 1e-9)
    assertEquals(50.0, (7_000_000L - second.lastSeenMs) / 60000.0, 1e-9)
  }

  @Test
  fun successorOfAPreV04RecordUsesItsIdAsTheLineage() {
    val old = ObjectRecord.fromJson(legacy().getJSONObject(2)) // no lineage keys in the file
    val l = Lineage.successor(old)
    assertEquals("1b328d30", l.lineageId)
    assertEquals("1b328d30", l.supersedes)
    assertEquals(1, l.moveCount)
    assertEquals(old.lastSeenMs, l.lineageStartMs)
  }
}
