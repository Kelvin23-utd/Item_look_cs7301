package edu.cs6301.findback.search

import java.util.Locale

/**
 * v0.9 (docs/PLAN_v0.9_find_by_name.md): what the matching rule sees of one item. [extraNames] and
 * [descriptors] are the item's bound extra names (empty when none are bound); the search uses them
 * only when extra names are on.
 */
data class SearchItem(
  val id: String,
  val name: String,
  /** Every detector label: the label and all label-vote keys, or a tapped record's raw label. */
  val labels: List<String>,
  val container: String?,
  /** Room name, or null when the item has no room in the current map. */
  val room: String?,
  val lastSeenMs: Long,
  val extraNames: List<String> = emptyList(),
  val descriptors: List<String> = emptyList(),
)

/**
 * One result. [matchClass]: 0 to 3, lower is better; null for an empty query. [viaExtra]: a result
 * with extra names on and not with them off. [extraMatches]: the extra names, then the descriptors,
 * in which a query word matched a word (empty with extra names off or an empty query).
 */
data class SearchHit<T>(val item: T, val matchClass: Int?, val viaExtra: Boolean, val extraMatches: List<String>)

/**
 * The plan's matching rule. Pure Kotlin, no Android imports; the offline harness implements the
 * same rule and both are tested against `name_search_cases.json`.
 *
 * | Class | Every query word ... |
 * | --- | --- |
 * | 0 | (the whole normalised query equals the normalised name or one extra name) |
 * | 1 | is a prefix of a word of the naming pool (name, extra names) |
 * | 2 | is a prefix of a word of the naming or other pool (labels, descriptors, container, room) |
 * | 3 | is a prefix of a word of either pool, or has 3+ characters and is inside one |
 */
object NameSearch {
  /** A query word this long or longer also matches inside a word (class 3). */
  const val CONTAINS_MIN_CHARS = 3

  /** Lower case (root locale); every character that is not a letter or digit is a space; runs of spaces collapse; trimmed. */
  fun normalise(s: String): String {
    val lower = s.lowercase(Locale.ROOT)
    val sb = StringBuilder(lower.length)
    var gap = false
    var i = 0
    while (i < lower.length) {
      val cp = lower.codePointAt(i)
      if (Character.isLetterOrDigit(cp)) {
        if (gap && sb.isNotEmpty()) sb.append(' ')
        gap = false
        sb.appendCodePoint(cp)
      } else {
        gap = true
      }
      i += Character.charCount(cp)
    }
    return sb.toString()
  }

  /** The words of [s] after [normalise]. */
  fun words(s: String): List<String> = normalise(s).let { if (it.isEmpty()) emptyList() else it.split(' ') }

  /** Match class of [item] for [query], or null when it is not a result (also for a query without words). */
  fun matchClass(item: SearchItem, query: String, extraOn: Boolean): Int? {
    val q = normalise(query)
    return if (q.isEmpty()) null else classOf(item, q, q.split(' '), extraOn)
  }

  /**
   * The results for [query], ordered by class ascending, then last seen descending, then id
   * ascending. A query without words lists every item by last seen descending (class null).
   * [fields] gives what the rule sees of each item.
   */
  fun <T> search(items: List<T>, query: String, extraOn: Boolean, fields: (T) -> SearchItem): List<SearchHit<T>> {
    val q = normalise(query)
    val qWords = if (q.isEmpty()) emptyList() else q.split(' ')
    val found = ArrayList<Pair<SearchHit<T>, SearchItem>>()
    for (t in items) {
      val si = fields(t)
      if (qWords.isEmpty()) {
        found.add(SearchHit(t, null, false, emptyList()) to si)
        continue
      }
      val cls = classOf(si, q, qWords, extraOn) ?: continue
      val via = extraOn && classOf(si, q, qWords, false) == null
      found.add(SearchHit(t, cls, via, if (extraOn) extraMatches(si, qWords) else emptyList()) to si)
    }
    return found
      .sortedWith(
        compareBy<Pair<SearchHit<T>, SearchItem>> { it.first.matchClass ?: 0 }
          .thenByDescending { it.second.lastSeenMs }
          .thenBy { it.second.id }
      )
      .map { it.first }
  }

  /** [q] is the normalised query, [qWords] its words (at least one). */
  private fun classOf(item: SearchItem, q: String, qWords: List<String>, extraOn: Boolean): Int? {
    val names = if (extraOn) listOf(item.name) + item.extraNames else listOf(item.name)
    if (names.any { normalise(it) == q }) return 0
    val naming = names.flatMap(::words)
    val otherFields = item.labels + listOfNotNull(item.container, item.room) + if (extraOn) item.descriptors else emptyList()
    val all = naming + otherFields.flatMap(::words)
    return when {
      qWords.all { w -> naming.any { it.startsWith(w) } } -> 1
      qWords.all { w -> all.any { it.startsWith(w) } } -> 2
      qWords.all { w -> all.any { wordMatches(it, w) } } -> 3
      else -> null
    }
  }

  /** Query word [w] is a prefix of [word], or has at least [CONTAINS_MIN_CHARS] characters and is inside it. */
  private fun wordMatches(word: String, w: String): Boolean =
    word.startsWith(w) || (w.codePointCount(0, w.length) >= CONTAINS_MIN_CHARS && word.contains(w))

  private fun extraMatches(item: SearchItem, qWords: List<String>): List<String> =
    (item.extraNames + item.descriptors)
      .filter { text -> words(text).let { ws -> qWords.any { w -> ws.any { wordMatches(it, w) } } } }
      .map { it.trim() }
      .distinct()
}
