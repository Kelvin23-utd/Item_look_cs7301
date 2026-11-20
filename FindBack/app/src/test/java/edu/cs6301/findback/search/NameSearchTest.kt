package edu.cs6301.findback.search

import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

/** v0.9 matching rule (docs/PLAN_v0.9_find_by_name.md), checked against the cases shared with the offline harness. */
class NameSearchTest {
  private val cases: JSONObject by lazy { JSONObject(javaClass.classLoader!!.getResource("name_search_cases.json")!!.readText()) }

  private fun strings(a: JSONArray?): List<String> = if (a == null) emptyList() else (0 until a.length()).map { a.getString(it) }

  private fun ints(a: JSONArray): List<Int> = (0 until a.length()).map { a.getInt(it) }

  /** The file's items; their extra names and descriptors count as bound. */
  private fun items(): List<SearchItem> {
    val arr = cases.getJSONArray("items")
    return (0 until arr.length()).map { i ->
      val o = arr.getJSONObject(i)
      SearchItem(
        id = o.getString("id"),
        name = o.getString("name"),
        labels = strings(o.optJSONArray("labels")),
        container = if (o.isNull("container")) null else o.getString("container"),
        room = if (o.isNull("room")) null else o.getString("room"),
        lastSeenMs = o.getLong("lastSeenMs"),
        extraNames = strings(o.optJSONArray("extraNames")),
        descriptors = strings(o.optJSONArray("descriptors")),
      )
    }
  }

  private fun search(query: String, extraOn: Boolean, items: List<SearchItem> = items()) = NameSearch.search(items, query, extraOn) { it }

  @Test
  fun everySharedCase() {
    val items = items()
    val arr = cases.getJSONArray("cases")
    assertTrue(arr.length() >= 20)
    for (i in 0 until arr.length()) {
      val c = arr.getJSONObject(i)
      val q = c.getString("query")
      val off = search(q, false, items)
      val on = search(q, true, items)
      assertEquals("off order for \"$q\"", strings(c.getJSONArray("off")), off.map { it.item.id })
      assertEquals("on order for \"$q\"", strings(c.getJSONArray("on")), on.map { it.item.id })
      assertEquals("viaExtra for \"$q\"", strings(c.getJSONArray("viaExtra")).toSet(), on.filter { it.viaExtra }.map { it.item.id }.toSet())
      // With extra names off nothing is via extra.
      assertTrue("off viaExtra for \"$q\"", off.none { it.viaExtra })
      if (c.has("offClasses")) assertEquals("off classes for \"$q\"", ints(c.getJSONArray("offClasses")), off.map { it.matchClass })
      if (c.has("onClasses")) assertEquals("on classes for \"$q\"", ints(c.getJSONArray("onClasses")), on.map { it.matchClass })
      // matchClass agrees with the class of each result.
      for (h in on) assertEquals(h.matchClass, NameSearch.matchClass(h.item, q, true))
    }
  }

  @Test
  fun normalise() {
    assertEquals("cup", NameSearch.normalise("  CUP!! "))
    assertEquals("coffee cup", NameSearch.normalise("coffee-cup"))
    assertEquals("grandma s mug 2", NameSearch.normalise("Grandma's   Mug #2"))
    assertEquals("", NameSearch.normalise(" -!- "))
    assertEquals(listOf("tv", "remote"), NameSearch.words("TV/remote"))
    assertEquals(emptyList<String>(), NameSearch.words("  "))
    // Letters outside ASCII are letters.
    assertEquals("café crème", NameSearch.normalise("Café, Crème!"))
  }

  @Test
  fun queryWithoutWordsListsEverythingNewestFirst() {
    val all = search("", true).map { it.item.id }
    assertEquals(all, search("?!", true).map { it.item.id })
    assertTrue(search("  ", false).all { it.matchClass == null && !it.viaExtra && it.extraMatches.isEmpty() })
    assertNull(NameSearch.matchClass(items()[0], "--", true))
  }

  @Test
  fun tiesGoByIdAscending() {
    val a = SearchItem("b2", "mug", listOf("cup"), null, null, 50)
    val b = SearchItem("a1", "mug", listOf("cup"), null, null, 50)
    val c = SearchItem("c3", "mug", listOf("cup"), null, null, 60)
    assertEquals(listOf("c3", "a1", "b2"), NameSearch.search(listOf(a, b, c), "mug", true) { it }.map { it.item.id })
    assertEquals(listOf("c3", "a1", "b2"), NameSearch.search(listOf(a, b, c), "", true) { it }.map { it.item.id })
  }

  @Test
  fun extraMatchesNameWhatMatched() {
    val on = search("cup", true).associateBy { it.item.id }
    assertEquals(listOf("coffee cup"), on.getValue("c").extraMatches)
    assertEquals(listOf("teacup"), on.getValue("f").extraMatches) // inside a word
    assertEquals(listOf("starbucks cup"), on.getValue("a").extraMatches) // not via extra, still listed
    assertFalse(on.getValue("a").viaExtra)
    // Extra names first, then descriptors.
    assertEquals(listOf("coffee cup", "red"), search("red cup", true).single().extraMatches)
    assertEquals(listOf("red"), search("red", true).single().extraMatches)
    // Off: no extra text at all.
    assertTrue(search("cup", false).all { it.extraMatches.isEmpty() })
  }

  @Test
  fun unboundItemsUseOnlyTheirOwnFields() {
    val bare = items().map { it.copy(extraNames = emptyList(), descriptors = emptyList()) }
    for (q in listOf("cup", "starbucks", "red cup", "remote", "kitchen")) {
      assertEquals(search(q, false, bare).map { it.item.id }, search(q, true, bare).map { it.item.id })
      assertTrue(search(q, true, bare).none { it.viaExtra })
    }
  }

  @Test
  fun containmentNeedsThreeCharacters() {
    val item = SearchItem("x", "teacup", emptyList(), null, null, 1)
    assertEquals(3, NameSearch.matchClass(item, "cup", false))
    assertNull(NameSearch.matchClass(item, "up", false))
    assertEquals(1, NameSearch.matchClass(item, "tea", false))
    assertEquals(0, NameSearch.matchClass(item, "TEACUP", false))
  }
}
