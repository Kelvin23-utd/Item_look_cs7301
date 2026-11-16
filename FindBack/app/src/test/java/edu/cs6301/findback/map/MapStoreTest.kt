package edu.cs6301.findback.map

import java.io.File
import java.nio.file.Files
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class MapStoreTest {
  private fun sample(): HomeMap {
    val g = WalkGraph()
    g.pickRoom("r1", null)
    for (i in 0..8) g.visit(Vec3(0.5 * i, 1.35, -0.1 * i), "sess0001", 1000L + i)
    return HomeMap(
      mapId = "ab12cd34",
      createdMs = 1_794_000_000_000,
      fixPoints =
        listOf(
          FixPoint("fp000001", "Home picture", FixType.IMAGE, 0.0, 0.0, 0.0, 0.0, "X", 0.116, null, 1L, "sess0001"),
          FixPoint("fp000002", "Door spot", FixType.SPOT, 2.5, 1.3, -0.7, -1.25, null, null, "fix/fp000002.jpg", 2L, "sess0001"),
        ),
      rooms = listOf(Room("r1", "Kitchen", 3L), Room("r2", "Bed room \"B\"", 4L)),
      nodes = g.nodes(),
      edges = g.edges(),
    )
  }

  @Test
  fun jsonRoundTrip() {
    val m = sample()
    val text = HomeMapJson.toJson(m).toString(1)
    val back = HomeMapJson.fromJson(JSONObject(text))
    assertEquals(m, back)
    // Keys from the plan's schema.
    val o = JSONObject(text)
    for (k in listOf("version", "mapId", "createdMs", "fixPoints", "rooms", "nodes", "edges")) assertTrue(k, o.has(k))
    val fp = o.getJSONArray("fixPoints").getJSONObject(0)
    for (k in listOf("id", "name", "type", "x", "y", "z", "yaw", "yawAxis", "widthM", "ghostFile", "createdMs", "sessionId")) {
      assertTrue(k, fp.has(k))
    }
    assertTrue(fp.isNull("ghostFile"))
    assertEquals(2, o.getJSONArray("edges").getJSONArray(0).length())
  }

  @Test
  fun saveLoadAndCorruptFile() {
    val dir = Files.createTempDirectory("mapstore").toFile()
    val saved = ArrayList<HomeMap>()
    val s = MapStore(dir, onSaved = { saved.add(it) })
    assertNull(s.load())
    assertFalse(s.flush()) // nothing published yet
    s.publish(sample())
    assertTrue(s.isDirty())
    assertTrue(s.flush())
    assertFalse(s.isDirty())
    assertFalse(s.flush()) // not dirty
    assertEquals(1, saved.size)
    assertEquals(sample(), MapStore(dir).load())
    assertFalse(File(dir, "map.json.tmp").exists())

    File(dir, "map.json").writeText("{ not json")
    val errors = ArrayList<String>()
    assertNull(MapStore(dir, onError = { m, _ -> errors.add(m) }).load())
    assertTrue(File(dir, "map.json").exists()) // never deleted
    assertEquals(1, dir.listFiles()!!.count { it.name.startsWith("map.corrupt.") })
    assertEquals(1, errors.size)
  }

  @Test
  fun archiveMovesTheFileAndStopsWriting() {
    val dir = Files.createTempDirectory("mapstore").toFile()
    val s = MapStore(dir)
    s.publish(sample())
    val archive = File(dir, "archive/20261116-120000")
    assertTrue(s.archiveTo(archive))
    assertFalse(File(dir, "map.json").exists())
    assertEquals(sample(), MapStore(archive).load())
    // A late publish from the GL thread must not recreate map.json.
    s.publish(sample())
    assertFalse(s.flush())
    assertFalse(File(dir, "map.json").exists())
  }
}
