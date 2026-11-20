package edu.cs6301.findback.search

import java.io.File
import java.nio.file.Files
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
import org.junit.Test

/** v0.9 names.json: the parser, the binding rule over photo bytes, and the store's hash cache. */
class ExtraNamesTest {
  private val photoA = "photo A bytes".toByteArray()
  private val photoB = "photo B bytes".toByteArray()

  private fun item(kind: String, id: String, sha: String, names: List<String> = listOf("drink can"), descriptors: List<String> = listOf("red")) =
    JSONObject()
      .put("kind", kind)
      .put("id", id)
      .put("cropSha256", sha)
      .put("names", JSONArray(names))
      .put("descriptors", JSONArray(descriptors))

  private fun file(vararg items: JSONObject, version: Any = 1): String =
    JSONObject()
      .put("version", version)
      .put("source", "model_c:2026-11-18")
      .put("createdMs", 1795000000000L)
      .put("items", JSONArray(items.toList()))
      .toString()

  private fun reason(text: String): String =
    try {
      ExtraNames.parse(text)
      fail("expected Unusable")
      ""
    } catch (e: ExtraNames.Unusable) {
      e.reason
    }

  @Test
  fun sha256IsTheStandardDigest() {
    assertEquals("ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad", ExtraNames.sha256Hex("abc".toByteArray()))
    assertEquals("e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855", ExtraNames.sha256Hex(ByteArray(0)))
  }

  @Test
  fun parsesThePlanSchema() {
    val sha = ExtraNames.sha256Hex(photoA)
    val f = ExtraNames.parse(file(item("entry", "e98ae991", sha), item("record", "r1", sha.uppercase(), listOf(" mug ", "", "mug"), emptyList())))
    assertEquals("model_c:2026-11-18", f.source)
    assertEquals(1795000000000L, f.createdMs)
    assertEquals(2, f.items.size)
    assertEquals(0, f.skipped)
    val e = f.item("entry", "e98ae991")!!
    assertEquals(listOf("drink can"), e.names)
    assertEquals(listOf("red"), e.descriptors)
    // Hex is kept in lower case; names are trimmed, blank and repeated ones dropped.
    val r = f.item("record", "r1")!!
    assertEquals(sha, r.cropSha256)
    assertEquals(listOf("mug"), r.names)
    // Kinds are separate: an entry and a record may share an id.
    assertNull(f.item("record", "e98ae991"))
  }

  @Test
  fun malformedItemsAreSkipped() {
    val sha = ExtraNames.sha256Hex(photoA)
    val f =
      ExtraNames.parse(
        file(
          item("entry", "ok", sha),
          item("thing", "x1", sha), // unknown kind
          item("entry", "", sha), // no id
          item("entry", "x2", "abc"), // not 64 hex
          item("entry", "ok", ExtraNames.sha256Hex(photoB), listOf("second")), // repeated: the first wins
        )
      )
    assertEquals(listOf("ok"), f.items.map { it.id })
    assertEquals(4, f.skipped)
    assertEquals(listOf("drink can"), f.item("entry", "ok")!!.names)
    // No source: null, not the text "null".
    assertNull(ExtraNames.parse("""{"version": 1, "source": null, "items": []}""").source)
  }

  @Test
  fun unusableFilesSayWhy() {
    assertEquals("not JSON", reason("not json"))
    assertEquals("not JSON", reason("[1, 2]"))
    assertEquals("no version", reason("""{"items": []}"""))
    assertEquals("version 2", reason(file(version = 2)))
    assertEquals("version \"1\"", reason(file(version = "1")))
    assertEquals("no items", reason("""{"version": 1}"""))
    assertEquals("no items", reason("""{"version": 1, "items": {}}"""))
  }

  @Test
  fun bindingNeedsTheSamePhotoBytes() {
    val sha = ExtraNames.sha256Hex(photoA)
    val it = ExtraNames.parse(file(item("entry", "e1", sha))).item("entry", "e1")
    assertEquals(NameSet(listOf("drink can"), listOf("red")), ExtraNames.bindBytes(it, photoA))
    assertNull(ExtraNames.bindBytes(it, photoB)) // a scan replaced the photo
    assertNull(ExtraNames.bindBytes(it, null)) // no photo
    assertNull(ExtraNames.bindBytes(null, photoA)) // not in the file
    assertNotNull(ExtraNames.bind(it, sha.uppercase()))
  }

  private fun tempRoot(): File = Files.createTempDirectory("names").toFile()

  @Test
  fun storeWithoutAFileHasNoNames() {
    val root = tempRoot()
    val s = ExtraNamesStore(root)
    assertNull(s.file)
    assertEquals("missing", s.loadError)
    assertEquals(emptyMap<String, NameSet>(), s.bind(listOf(PhotoRef("entry", "e1", "inventory/e1.jpg"))))
    File(root, ExtraNames.FILE_NAME).writeText(file(version = 3))
    assertEquals("version 3", ExtraNamesStore(root).loadError)
  }

  @Test
  fun storeBindsMatchingPhotosAndCachesTheirHashes() {
    val root = tempRoot()
    File(root, "inventory").mkdirs()
    File(root, "crops").mkdirs()
    File(root, "inventory/e1.jpg").writeBytes(photoA)
    File(root, "inventory/e2.jpg").writeBytes(photoB)
    File(root, "crops/r1.jpg").writeBytes(photoA)
    val shaA = ExtraNames.sha256Hex(photoA)
    File(root, ExtraNames.FILE_NAME)
      .writeText(file(item("entry", "e1", shaA), item("entry", "e2", shaA), item("record", "r1", shaA, listOf("keys"), emptyList())))
    val s = ExtraNamesStore(root)
    assertNull(s.loadError)
    val refs =
      listOf(
        PhotoRef("entry", "e1", "inventory/e1.jpg"),
        PhotoRef("entry", "e2", "inventory/e2.jpg"), // its photo is not the one named
        PhotoRef("record", "r1", "crops/r1.jpg"),
        PhotoRef("entry", "e3", "inventory/e3.jpg"), // not in the file: never read
      )
    val bound = s.bind(refs)
    assertEquals(setOf("entry:e1", "record:r1"), bound.keys)
    assertEquals(listOf("keys"), bound.getValue("record:r1").names)
    assertEquals(3, s.photosHashed)
    // Unchanged photos are not read again.
    assertEquals(bound, s.bind(refs))
    assertEquals(3, s.photosHashed)
    // A replaced photo (new length) is read again and unbinds the names.
    File(root, "inventory/e1.jpg").writeBytes(photoA + photoB)
    assertEquals(setOf("record:r1"), s.bind(refs).keys)
    assertEquals(4, s.photosHashed)
    // A missing photo binds nothing.
    File(root, "crops/r1.jpg").delete()
    assertTrue(s.bind(refs).isEmpty())
  }
}
