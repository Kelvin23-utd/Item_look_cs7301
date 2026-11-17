package edu.cs6301.findback.trace

import java.util.Locale
import org.junit.Assert.assertEquals
import org.junit.Test

/** v0.7 plan section 6, item 1: CSV line formatting (locale, empty fields). */
class CsvLineTest {
  private fun fixed(v: Double, d: Int): String = StringBuilder().also { CsvLine.appendFixed(it, v, d) }.toString()

  /** The line without its leading file tag. */
  private fun body(s: String) = s.substring(1)

  @Test
  fun numbersUseAPointWhateverTheLocale() {
    val saved = Locale.getDefault()
    try {
      for (loc in listOf(Locale.GERMANY, Locale.FRANCE, Locale("ar", "EG"), Locale.US)) {
        Locale.setDefault(loc)
        val s = CsvLine().begin(TraceFile.SENSORS).long(1_700_000_000_123L).long(42L).str("acc").num(9.80665, 4).num(-0.5, 4).num(1234567.25, 2).int(3).end()
        assertEquals("1700000000123,42,acc,9.8067,-0.5,1234567.25,3\n", body(s))
      }
    } finally {
      Locale.setDefault(saved)
    }
  }

  @Test
  fun fixedPointRoundsAndDropsTrailingZeros() {
    assertEquals("1.05", fixed(1.05, 4))
    assertEquals("1.5", fixed(1.5, 4))
    assertEquals("12.3456", fixed(12.34561, 4))
    assertEquals("0.0001", fixed(0.00005, 4)) // half up
    assertEquals("0.001", fixed(0.000999, 4))
    assertEquals("2", fixed(1.99999, 4))
    assertEquals("-3.25", fixed(-3.25, 2))
    assertEquals("7", fixed(7.0, 3))
    assertEquals("123", fixed(123.4, 0))
    assertEquals("0.5", fixed(0.5, 1))
    // No "-0" for a small negative value.
    assertEquals("0", fixed(-0.00001, 3))
    assertEquals("0", fixed(-0.0, 3))
    // Very large values fall back to Double.toString, which never uses a locale either.
    assertEquals("1.0E12", fixed(1e12, 2))
  }

  @Test
  fun missingValuesAreEmptyFields() {
    val s =
      CsvLine().begin(TraceFile.DEPTH).long(1L).num(Double.NaN, 3).num(Double.POSITIVE_INFINITY, 2).str(null).empty().num(Float.NaN, 1).bool(true).bool(false).end()
    assertEquals("1,,,,,,1,0\n", body(s))
  }

  @Test
  fun lineStartsWithTheFileTagAndIsReusable() {
    val line = CsvLine(8) // grows as needed
    val a = line.begin(TraceFile.POSE).str("TRACKING").num(0.123456789, 5).end()
    val b = line.begin(TraceFile.GUIDANCE).str("q1").end()
    assertEquals('0', a[0])
    assertEquals("TRACKING,0.12346\n", body(a))
    assertEquals('3', b[0])
    assertEquals("q1\n", body(b))
  }

  @Test
  fun textIsQuotedOnlyWhenNeeded() {
    val s = CsvLine().begin(TraceFile.POSE).str("FINDING").str("a,b").str("say \"hi\"").str("two\nlines").end()
    assertEquals("FINDING,\"a,b\",\"say \"\"hi\"\"\",\"two\nlines\"\n", body(s))
  }

  @Test
  fun headersHaveTheColumnsOfThePlan() {
    assertEquals(
      listOf("t", "ns", "tracking", "failure", "wx", "wy", "wz", "qx", "qy", "qz", "qw", "located", "mx", "my", "mz", "myaw", "metersSinceFix", "speed", "turnRate", "screen", "room", "query"),
      TraceFile.POSE.header.split(","),
    )
    assertEquals(listOf("t", "ns", "type", "x", "y", "z", "accuracy"), TraceFile.SENSORS.header.split(","))
    assertEquals(
      listOf("t", "ns", "ok", "w", "h", "validShare", "usableShare", "median", "p10", "p90", "centre", "points", "planes", "planeArea", "lightIntensity", "speed", "turnRate"),
      TraceFile.DEPTH.header.split(","),
    )
    assertEquals(
      listOf("t", "ns", "query", "mode", "resolvable", "distance", "bearingDeg", "elevationDeg", "routeM", "straightM", "inView", "ringX", "ringY"),
      TraceFile.GUIDANCE.header.split(","),
    )
    assertEquals(listOf("pose.csv", "sensors.csv", "depth.csv", "guidance.csv"), TraceFile.values().map { it.fileName })
    for (f in TraceFile.values()) assertEquals(f, TraceFile.ofTag(f.tag))
    assertEquals(null, TraceFile.ofTag('x'))
  }

  @Test
  fun aSensorRowHasAsManyFieldsAsTheHeader() {
    // As SensorRecorder writes a barometer row: x only, y and z empty.
    val s = CsvLine().begin(TraceFile.SENSORS).long(1L).long(2L).str("baro").num(1013.25, 3).empty().empty().int(3).end()
    assertEquals(TraceFile.SENSORS.columns, body(s).trimEnd('\n').split(",").size)
  }
}
