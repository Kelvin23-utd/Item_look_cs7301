package edu.cs6301.findback.memory

import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.Base64
import java.util.Random
import org.json.JSONObject
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
import org.junit.Test

/** verify/<vid>_emb.json: base64 of little-endian float32, readable by numpy's "<f4". */
class EmbeddingFileTest {
  /** Reference encoding with the JDK's own Base64 and ByteBuffer. */
  private fun reference(values: FloatArray): String {
    val buf = ByteBuffer.allocate(values.size * 4).order(ByteOrder.LITTLE_ENDIAN)
    values.forEach { buf.putFloat(it) }
    return Base64.getEncoder().encodeToString(buf.array())
  }

  @Test
  fun knownVectors() {
    // 1.0f is 0x3F800000: little-endian bytes 00 00 80 3F.
    assertEquals("AACAPw==", EmbeddingFile.encode(floatArrayOf(1f)))
    // -2.0f is 0xC0000000: 00 00 00 C0. 0.5f is 0x3F000000: 00 00 00 3F.
    assertEquals("AAAAwAAAAD8=", EmbeddingFile.encode(floatArrayOf(-2f, 0.5f)))
    assertEquals("", EmbeddingFile.encode(FloatArray(0)))
    assertArrayEquals(floatArrayOf(1f), EmbeddingFile.decode("AACAPw=="), 0f)
  }

  @Test
  fun matchesTheJdkEncoderForEveryLength() {
    val rnd = Random(7)
    for (n in 0..40) {
      val v = FloatArray(n) { rnd.nextFloat() * 2f - 1f }
      assertEquals("length $n", reference(v), EmbeddingFile.encode(v))
    }
    // Base64 padding cases on raw bytes (1, 2 and 3 bytes over a multiple of 3).
    for (n in 0..10) {
      val b = ByteArray(n).also { rnd.nextBytes(it) }
      val text = EmbeddingFile.base64(b)
      assertEquals(Base64.getEncoder().encodeToString(b), text)
      assertArrayEquals(b, EmbeddingFile.unbase64(text))
    }
  }

  @Test
  fun embeddingRoundTripIsBitExact() {
    val rnd = Random(11)
    val v = FloatArray(1024) { rnd.nextGaussian().toFloat() * 0.03f }
    v[0] = 0f
    v[1] = -0f
    v[2] = Float.MIN_VALUE
    v[3] = Float.MAX_VALUE
    v[4] = -1.17549435E-38f
    val text = EmbeddingFile.encode(v)
    assertEquals(5464, text.length) // 4096 bytes -> 1366 groups of 4 characters
    val back = EmbeddingFile.decode(text)
    assertEquals(v.size, back.size)
    for (i in v.indices) assertEquals("index $i", v[i].toRawBits(), back[i].toRawBits())
    // And the JDK decoder reads the same floats, as numpy's "<f4" would.
    val jdk = ByteBuffer.wrap(Base64.getDecoder().decode(text)).order(ByteOrder.LITTLE_ENDIAN)
    for (i in v.indices) assertEquals(v[i].toRawBits(), jdk.getFloat().toRawBits())
  }

  @Test
  fun rejectsTextThatIsNotAnEmbedding() {
    for (bad in listOf("AAA", "AAAA", "AA=A", "A*AA", "AAAAAAA=")) {
      try {
        EmbeddingFile.decode(bad)
        fail("accepted \"$bad\"")
      } catch (e: IllegalArgumentException) {
        // expected: bad length, bad character, misplaced padding, or not a multiple of 4 bytes
      }
    }
  }

  @Test
  fun fileShape() {
    val a = FloatArray(1024) { it / 1024f }
    val b = FloatArray(1024) { -it / 2048f }
    val j =
      EmbeddingFile.toJson(
        model = "mobilenet_v3_small",
        record = "1b328d30",
        query = "9f3c2a71",
        attempt = 2,
        candidates =
          listOf(
            EmbeddingFile.Candidate("projected", floatArrayOf(812f, 402.5f, 1100f, 690.5f), 0.8125f, a),
            EmbeddingFile.Candidate("detector", floatArrayOf(790f, 380f, 1130f, 720f), 0.5f, b),
          ),
      )
    val o = JSONObject(j.toString()) // as read back from the file
    assertEquals(setOf("model", "dim", "encoding", "record", "query", "attempt", "candidates"), o.keys().asSequence().toSet())
    assertEquals("mobilenet_v3_small", o.getString("model"))
    assertEquals(1024, o.getInt("dim"))
    assertEquals("base64-f32le", o.getString("encoding"))
    assertEquals("1b328d30", o.getString("record"))
    assertEquals("9f3c2a71", o.getString("query"))
    assertEquals(2, o.getInt("attempt"))
    val cs = o.getJSONArray("candidates")
    assertEquals(2, cs.length())
    val c0 = cs.getJSONObject(0)
    assertEquals(setOf("kind", "rect", "sim", "embedding"), c0.keys().asSequence().toSet())
    assertEquals("projected", c0.getString("kind"))
    assertEquals(4, c0.getJSONArray("rect").length())
    assertEquals(402.5, c0.getJSONArray("rect").getDouble(1), 0.0)
    assertEquals(0.8125, c0.getDouble("sim"), 0.0)
    assertArrayEquals(a, EmbeddingFile.decode(c0.getString("embedding")), 0f)
    assertEquals("detector", cs.getJSONObject(1).getString("kind"))
    assertArrayEquals(b, EmbeddingFile.decode(cs.getJSONObject(1).getString("embedding")), 0f)
    assertTrue(o.toString().length < 12_000) // two candidates: about 11 kB
  }

  /** What the analysis does: the stored embedding from the records log, the fresh one from the file. */
  @Test
  fun similarityCanBeRecomputedOfflineFromTheTwoFiles() {
    fun cosine(a: FloatArray, b: FloatArray): Float {
      var dot = 0f
      var na = 0f
      var nb = 0f
      for (i in a.indices) {
        dot += a[i] * b[i]
        na += a[i] * a[i]
        nb += b[i] * b[i]
      }
      return dot / (kotlin.math.sqrt(na) * kotlin.math.sqrt(nb))
    }
    val legacy = org.json.JSONArray(javaClass.classLoader!!.getResource("legacy_records_v02.json")!!.readText())
    val rec = ObjectRecord.fromJson(legacy.getJSONObject(0))
    val rnd = Random(3)
    val fresh = FloatArray(rec.embedding.size) { rec.embedding[it] + rnd.nextGaussian().toFloat() * 0.01f }
    val sim = cosine(fresh, rec.embedding)
    assertTrue(sim > 0.5f && sim < 0.9999f)

    // As written on the phone: a records_log.jsonl line and a verify/<vid>_emb.json file.
    val logLine = JSONObject().put("t", 1L).put("session", "s").put("op", "add").put("record", rec.toJson()).toString()
    val embText =
      EmbeddingFile.toJson(
          "mobilenet_v3_small", rec.id, "q0000001", 1,
          listOf(EmbeddingFile.Candidate("projected", floatArrayOf(0f, 0f, 10f, 10f), sim, fresh)),
        )
        .toString()

    // As read offline.
    val stored = ObjectRecord.fromJson(JSONObject(logLine).getJSONObject("record")).embedding
    val file = JSONObject(embText)
    assertEquals(rec.id, file.getString("record"))
    assertEquals(stored.size, file.getInt("dim"))
    val c = file.getJSONArray("candidates").getJSONObject(0)
    val decoded = EmbeddingFile.decode(c.getString("embedding"))
    assertArrayEquals(rec.embedding, stored, 0f) // the JSON number array is exact for float32
    assertArrayEquals(fresh, decoded, 0f)
    assertEquals(sim, cosine(decoded, stored), 0f)
    assertEquals(sim.toDouble(), c.getDouble("sim"), 0.0)
  }

  @Test
  fun noCandidatesGivesDimensionZero() {
    val o = EmbeddingFile.toJson("m", "r", "q", 1, emptyList())
    assertEquals(0, o.getInt("dim"))
    assertEquals(0, o.getJSONArray("candidates").length())
  }
}
