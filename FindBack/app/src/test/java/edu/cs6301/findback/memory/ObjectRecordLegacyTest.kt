package edu.cs6301.findback.memory

import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

/** v0.2 records.json files (the pilot items on the phone) must load with null map fields. */
class ObjectRecordLegacyTest {
  /** First three records of findback_data/records.json, copied verbatim (Nov 13 pilot). */
  private fun legacy(): JSONArray {
    val text = javaClass.classLoader!!.getResource("legacy_records_v02.json")!!.readText()
    return JSONArray(text)
  }

  @Test
  fun pilotRecordsLoadWithNullMapFields() {
    val arr = legacy()
    assertEquals(3, arr.length())
    val recs = (0 until arr.length()).map { ObjectRecord.fromJson(arr.getJSONObject(it)) }
    assertEquals(listOf("62907d85", "e7be2592", "1b328d30"), recs.map { it.id })
    for (r in recs) {
      assertNull(r.mapId)
      assertNull(r.mapPosition)
      assertNull(r.viewPosition)
      assertNull(r.roomId)
      assertNull(r.refPosition)
      assertEquals(1024, r.embedding.size)
      assertEquals("ff5fbb39", r.sessionId)
      assertEquals(3, r.worldPosition.size)
    }
    assertEquals("Home good", recs[1].name)
    assertEquals(0.7323592305183411f, recs[2].captureDistance, 1e-6f)
    // The old file has none of the new keys at all.
    assertFalse(arr.getJSONObject(0).has("mapId"))
  }

  @Test
  fun legacyRecordSurvivesARewrite() {
    // A rename re-saves old records; they must stay "photo only" and keep their data.
    val o = legacy().getJSONObject(0)
    val r = ObjectRecord.fromJson(o)
    val back = ObjectRecord.fromJson(JSONObject(r.toJson().toString()))
    assertNull(back.mapId)
    assertNull(back.mapPosition)
    assertArrayEquals(r.embedding, back.embedding, 0f)
    assertArrayEquals(r.worldPosition, back.worldPosition, 0f)
    assertEquals(r.lastSeenMs, back.lastSeenMs)
  }

  @Test
  fun v02ReferencePositionIsStillReadable() {
    val o = legacy().getJSONObject(1)
    o.put("refPosition", JSONArray(listOf(0.1, -0.2, 0.3)))
    val r = ObjectRecord.fromJson(o)
    assertArrayEquals(floatArrayOf(0.1f, -0.2f, 0.3f), r.refPosition, 1e-6f)
    assertNull(r.mapPosition)
  }

  @Test
  fun v03FieldsRoundTrip() {
    val r =
      ObjectRecord.fromJson(legacy().getJSONObject(2))
        .copy(
          refPosition = null,
          mapId = "ab12cd34",
          mapPosition = floatArrayOf(1f, 0.5f, -2f),
          viewPosition = floatArrayOf(1.3f, 1.4f, -1.2f),
          roomId = "room0001",
        )
    val j = r.toJson()
    assertTrue(j.has("mapPosition"))
    val back = ObjectRecord.fromJson(JSONObject(j.toString()))
    assertEquals("ab12cd34", back.mapId)
    assertArrayEquals(r.mapPosition, back.mapPosition, 1e-6f)
    assertArrayEquals(r.viewPosition, back.viewPosition, 1e-6f)
    assertEquals("room0001", back.roomId)
    assertNull(back.refPosition)
  }
}
