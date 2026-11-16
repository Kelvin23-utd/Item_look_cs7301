package edu.cs6301.findback.memory

import java.nio.ByteBuffer
import java.nio.ByteOrder
import org.json.JSONArray
import org.json.JSONObject

/**
 * `verify/<vid>_emb.json` (spec v0.4, S5): the embedding of every candidate crop of one
 * verification, so thresholds, margins and other decision rules can be re-run offline.
 *
 * The embeddings are the ones the verification pipeline computed for its own scores; nothing is
 * embedded again for this file. Pure Kotlin (no Android imports) so the encoding is unit-tested.
 *
 * ```
 * { "model": "mobilenet_v3_small", "dim": 1024, "encoding": "base64-f32le",
 *   "record": "<target record id>", "query": "<query id>", "attempt": 1,
 *   "candidates": [ { "kind": "projected", "rect": [l, t, r, b], "sim": 0.81, "embedding": "<base64>" } ] }
 * ```
 *
 * `candidates` has the order of the `verify` event's `scores`; `rect` is in sensor image pixels,
 * as there. `embedding` is the float32 vector as little-endian bytes in standard base64 (RFC 4648,
 * with padding): numpy reads it with `np.frombuffer(base64.b64decode(s), dtype="<f4")`.
 */
object EmbeddingFile {
  const val ENCODING = "base64-f32le"

  class Candidate(
    /** "projected" or "detector", as in the `verify` event. */
    val kind: String,
    /** Left, top, right, bottom in sensor image pixels. */
    val rect: FloatArray,
    /** Cosine similarity to the target record's stored embedding. */
    val similarity: Float,
    val embedding: FloatArray,
  )

  fun toJson(model: String, record: String, query: String, attempt: Int, candidates: List<Candidate>): JSONObject =
    JSONObject().apply {
      put("model", model)
      put("dim", candidates.firstOrNull()?.embedding?.size ?: 0)
      put("encoding", ENCODING)
      put("record", record)
      put("query", query)
      put("attempt", attempt)
      put(
        "candidates",
        JSONArray().apply {
          for (c in candidates) {
            put(
              JSONObject().apply {
                put("kind", c.kind)
                put("rect", JSONArray().apply { c.rect.forEach { put(it.toDouble()) } })
                put("sim", c.similarity.toDouble())
                put("embedding", encode(c.embedding))
              }
            )
          }
        },
      )
    }

  /** Float32 values as little-endian bytes, in standard base64 with padding. */
  fun encode(values: FloatArray): String {
    val buf = ByteBuffer.allocate(values.size * 4).order(ByteOrder.LITTLE_ENDIAN)
    for (v in values) buf.putFloat(v)
    return base64(buf.array())
  }

  /** Inverse of [encode]. Throws IllegalArgumentException on text that [encode] cannot produce. */
  fun decode(text: String): FloatArray {
    val bytes = unbase64(text)
    require(bytes.size % 4 == 0) { "Embedding byte length ${bytes.size} is not a multiple of 4" }
    val buf = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
    return FloatArray(bytes.size / 4) { buf.getFloat() }
  }

  // Base64 is written out here because java.util.Base64 needs API 26 (minSdk is 24) and
  // android.util.Base64 is a stub in JVM unit tests.
  private const val ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"

  fun base64(bytes: ByteArray): String {
    val out = StringBuilder((bytes.size + 2) / 3 * 4)
    var i = 0
    while (i + 2 < bytes.size) {
      val n = ((bytes[i].toInt() and 0xff) shl 16) or ((bytes[i + 1].toInt() and 0xff) shl 8) or (bytes[i + 2].toInt() and 0xff)
      out.append(ALPHABET[(n shr 18) and 63]).append(ALPHABET[(n shr 12) and 63])
      out.append(ALPHABET[(n shr 6) and 63]).append(ALPHABET[n and 63])
      i += 3
    }
    when (bytes.size - i) {
      1 -> {
        val n = (bytes[i].toInt() and 0xff) shl 16
        out.append(ALPHABET[(n shr 18) and 63]).append(ALPHABET[(n shr 12) and 63]).append("==")
      }
      2 -> {
        val n = ((bytes[i].toInt() and 0xff) shl 16) or ((bytes[i + 1].toInt() and 0xff) shl 8)
        out.append(ALPHABET[(n shr 18) and 63]).append(ALPHABET[(n shr 12) and 63])
        out.append(ALPHABET[(n shr 6) and 63]).append('=')
      }
    }
    return out.toString()
  }

  fun unbase64(text: String): ByteArray {
    require(text.length % 4 == 0) { "Base64 length ${text.length} is not a multiple of 4" }
    if (text.isEmpty()) return ByteArray(0)
    val pad = if (text.endsWith("==")) 2 else if (text.endsWith("=")) 1 else 0
    val out = ByteArray(text.length / 4 * 3 - pad)
    var o = 0
    var i = 0
    while (i < text.length) {
      var n = 0
      for (k in 0 until 4) {
        val c = text[i + k]
        val last = i + 4 == text.length
        val v =
          if (c == '=' && last && k >= 4 - pad) 0
          else ALPHABET.indexOf(c).also { require(it >= 0) { "Not a base64 character at ${i + k}" } }
        n = (n shl 6) or v
      }
      if (o < out.size) out[o++] = (n shr 16).toByte()
      if (o < out.size) out[o++] = (n shr 8).toByte()
      if (o < out.size) out[o++] = n.toByte()
      i += 4
    }
    return out
  }
}
