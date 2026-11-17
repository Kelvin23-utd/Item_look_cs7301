package edu.cs6301.findback.trace

import kotlin.math.abs

/*
 * v0.7 detailed trace (docs/PLAN_v0.7_trace.md, section 4): CSV line formatting. Pure Kotlin (no
 * Android imports), unit-tested in CsvLineTest.
 *
 * Numbers are written by hand rather than with String.format: no Locale is involved (the decimal
 * point is always '.'), and nothing but the line's own String is allocated, so the GL thread and
 * the sensor thread can format lines at 100 Hz without garbage.
 */

/**
 * Builds one CSV line in a reused [StringBuilder]. Not thread-safe: each thread owns its own.
 *
 * The first character of the returned String is the target file's tag ([TraceFile.tag]); the
 * writer strips it. The line ends with '\n'. A missing value (null, NaN or infinite) is an empty
 * field.
 */
class CsvLine(capacity: Int = 256) {
  private val sb = StringBuilder(capacity)
  private var first = true

  fun begin(file: TraceFile): CsvLine {
    sb.setLength(0)
    sb.append(file.tag)
    first = true
    return this
  }

  private fun sep() {
    if (first) first = false else sb.append(',')
  }

  fun long(v: Long): CsvLine {
    sep()
    sb.append(v)
    return this
  }

  fun int(v: Int): CsvLine {
    sep()
    sb.append(v)
    return this
  }

  /** 1 or 0. */
  fun bool(v: Boolean): CsvLine {
    sep()
    sb.append(if (v) '1' else '0')
    return this
  }

  /** [v] rounded to [decimals] places, trailing zeros removed; empty for NaN and infinities. */
  fun num(v: Double, decimals: Int): CsvLine {
    sep()
    appendFixed(sb, v, decimals)
    return this
  }

  fun num(v: Float, decimals: Int): CsvLine = num(v.toDouble(), decimals)

  /** The text, quoted when it holds a comma, a quote or a line break; empty for null. */
  fun str(v: String?): CsvLine {
    sep()
    if (v != null) appendEscaped(sb, v)
    return this
  }

  fun empty(): CsvLine {
    sep()
    return this
  }

  fun end(): String {
    sb.append('\n')
    return sb.toString()
  }

  companion object {
    private val POW10 = LongArray(10).also { p -> p[0] = 1; for (i in 1 until p.size) p[i] = p[i - 1] * 10 }

    /** Above this magnitude the fixed-point path could overflow; Double.toString is used instead. */
    private const val FIXED_LIMIT = 1e9

    fun appendFixed(sb: StringBuilder, v: Double, decimals: Int) {
      if (v.isNaN() || v.isInfinite()) return
      val d = decimals.coerceIn(0, POW10.size - 1)
      if (abs(v) >= FIXED_LIMIT) {
        sb.append(v) // Double.toString never uses a locale
        return
      }
      val scale = POW10[d]
      val r = Math.round(abs(v) * scale)
      if (r == 0L) {
        sb.append('0') // never "-0"
        return
      }
      if (v < 0) sb.append('-')
      sb.append(r / scale)
      var frac = r % scale
      if (d == 0 || frac == 0L) return
      // Drop trailing zeros, then pad the rest with leading zeros to its place.
      var digits = d
      while (frac % 10 == 0L) {
        frac /= 10
        digits--
      }
      sb.append('.')
      var lead = digits - 1
      var f = frac
      while (f >= 10) {
        f /= 10
        lead--
      }
      while (lead-- > 0) sb.append('0')
      sb.append(frac)
    }

    fun appendEscaped(sb: StringBuilder, s: String) {
      var needs = false
      for (c in s) {
        if (c == ',' || c == '"' || c == '\n' || c == '\r') {
          needs = true
          break
        }
      }
      if (!needs) {
        sb.append(s)
        return
      }
      sb.append('"')
      for (c in s) {
        if (c == '"') sb.append('"')
        sb.append(c)
      }
      sb.append('"')
    }
  }
}

/** The four trace files of a session (section 4), with their header rows. */
enum class TraceFile(val tag: Char, val fileName: String, val header: String) {
  POSE(
    '0',
    "pose.csv",
    "t,ns,tracking,failure,wx,wy,wz,qx,qy,qz,qw,located,mx,my,mz,myaw,metersSinceFix,speed,turnRate,screen,room,query",
  ),
  SENSORS('1', "sensors.csv", "t,ns,type,x,y,z,accuracy"),
  DEPTH(
    '2',
    "depth.csv",
    "t,ns,ok,w,h,validShare,usableShare,median,p10,p90,centre,points,planes,planeArea,lightIntensity,speed,turnRate",
  ),
  GUIDANCE(
    '3',
    "guidance.csv",
    "t,ns,query,mode,resolvable,distance,bearingDeg,elevationDeg,routeM,straightM,inView,ringX,ringY",
  );

  val columns: Int
    get() = header.count { it == ',' } + 1

  companion object {
    fun ofTag(c: Char): TraceFile? =
      when (c) {
        '0' -> POSE
        '1' -> SENSORS
        '2' -> DEPTH
        '3' -> GUIDANCE
        else -> null
      }
  }
}
