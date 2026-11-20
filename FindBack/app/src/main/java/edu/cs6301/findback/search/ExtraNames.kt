package edu.cs6301.findback.search

import java.security.MessageDigest
import java.util.Locale
import org.json.JSONArray
import org.json.JSONObject

/** v0.9: the extra names and descriptors of one item, bound to its current photo. */
data class NameSet(val names: List<String>, val descriptors: List<String>)

/** One item of names.json. [cropSha256] is lower-case hex. */
data class NamesItem(
  val kind: String,
  val id: String,
  val cropSha256: String,
  val names: List<String>,
  val descriptors: List<String>,
)

/**
 * names.json as read. [source] is the alias of what made the names ("model_c:2026-11-18"), null
 * when absent. [skipped]: items left out because a field was missing or malformed, or because an
 * earlier item had the same kind and id.
 */
data class NamesFile(val source: String?, val createdMs: Long?, val items: List<NamesItem>, val skipped: Int = 0) {
  private val byKey: Map<String, NamesItem> = items.associateBy { ExtraNames.key(it.kind, it.id) }

  fun item(kind: String, id: String): NamesItem? = byKey[ExtraNames.key(kind, id)]
}

/**
 * v0.9 (docs/PLAN_v0.9_find_by_name.md, "Extra names"): names.json in the files root.
 *
 * ```
 * { "version": 1, "source": "<alias>", "createdMs": 0,
 *   "items": [ {"kind": "entry"|"record", "id": "...", "cropSha256": "<64 hex>",
 *               "names": ["..."], "descriptors": ["..."]} ] }
 * ```
 * The phone's own names and labels are never changed. An item's extra names are used only while
 * the SHA-256 of its current photo file's bytes equals `cropSha256` ([bind]). Pure Kotlin, no
 * Android imports; parsing and binding are unit-tested.
 */
object ExtraNames {
  const val FILE_NAME = "names.json"
  const val VERSION = 1
  const val KIND_ENTRY = "entry"
  const val KIND_RECORD = "record"

  private val HEX64 = Regex("[0-9a-f]{64}")

  /** The key of an item in [NamesFile] and in bound-name maps: "entry:<id>" or "record:<id>". */
  fun key(kind: String, id: String) = "$kind:$id"

  /** Thrown by [parse]: why the file cannot be used ("not JSON", "version 2", "no items"). */
  class Unusable(val reason: String) : Exception(reason)

  /**
   * The file's items. Throws [Unusable] when the text is not a JSON object, its `version` is not
   * [VERSION] or it has no `items` array. A malformed item is skipped (counted in
   * [NamesFile.skipped]); the first item of a kind and id wins. Names and descriptors are trimmed
   * and blank ones dropped.
   */
  fun parse(text: String): NamesFile {
    val o =
      try {
        JSONObject(text)
      } catch (e: Exception) {
        throw Unusable("not JSON")
      }
    when (val version = o.opt("version")) {
      null -> throw Unusable("no version")
      is Number -> if (version.toDouble() != VERSION.toDouble()) throw Unusable("version $version")
      else -> throw Unusable("version " + if (version is String) JSONObject.quote(version) else version.toString())
    }
    val arr = o.opt("items") as? JSONArray ?: throw Unusable("no items")
    val items = ArrayList<NamesItem>()
    val seen = HashSet<String>()
    var skipped = 0
    for (i in 0 until arr.length()) {
      val item = (arr.opt(i) as? JSONObject)?.let(::item)
      if (item == null || !seen.add(key(item.kind, item.id))) {
        skipped++
        continue
      }
      items.add(item)
    }
    return NamesFile(
      source = o.opt("source") as? String,
      createdMs = (o.opt("createdMs") as? Number)?.toLong(),
      items = items,
      skipped = skipped,
    )
  }

  private fun item(o: JSONObject): NamesItem? {
    val kind = o.opt("kind") as? String
    if (kind != KIND_ENTRY && kind != KIND_RECORD) return null
    val id = (o.opt("id") as? String)?.takeIf { it.isNotBlank() } ?: return null
    val sha = (o.opt("cropSha256") as? String)?.lowercase(Locale.ROOT)?.takeIf { HEX64.matches(it) } ?: return null
    return NamesItem(kind, id, sha, strings(o.opt("names")), strings(o.opt("descriptors")))
  }

  private fun strings(v: Any?): List<String> {
    val a = v as? JSONArray ?: return emptyList()
    return (0 until a.length()).mapNotNull { (a.opt(it) as? String)?.trim()?.takeIf { s -> s.isNotEmpty() } }.distinct()
  }

  /** Lower-case hex SHA-256 of [bytes]. */
  fun sha256Hex(bytes: ByteArray): String = hex(MessageDigest.getInstance("SHA-256").digest(bytes))

  fun hex(digest: ByteArray): String {
    val sb = StringBuilder(digest.size * 2)
    for (b in digest) {
      val v = b.toInt() and 0xFF
      sb.append(HEX_DIGITS[v ushr 4]).append(HEX_DIGITS[v and 0x0F])
    }
    return sb.toString()
  }

  private const val HEX_DIGITS = "0123456789abcdef"

  /** The binding rule: [item]'s names when [photoSha256] (hex of the current photo's bytes) equals its `cropSha256`, else null. */
  fun bind(item: NamesItem?, photoSha256: String?): NameSet? {
    if (item == null || photoSha256 == null) return null
    if (item.cropSha256 != photoSha256.lowercase(Locale.ROOT)) return null
    return NameSet(item.names, item.descriptors)
  }

  /** [bind] over the photo's bytes; null bytes (no photo) bind nothing. */
  fun bindBytes(item: NamesItem?, photo: ByteArray?): NameSet? = bind(item, photo?.let(::sha256Hex))
}
