package edu.cs6301.findback.trace

import java.io.File
import java.nio.file.Files
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

/** v0.7 plan section 4.5 and section 6, item 1: the bounded queue's drop counting and the writer. */
class TraceWriterTest {
  private val tmp: File = Files.createTempDirectory("fbtrace").toFile()

  @After
  fun cleanUp() {
    tmp.deleteRecursively()
  }

  @Test
  fun queueDropsAndCountsWhenFull() {
    val q = LineQueue(3)
    assertTrue(q.offer("a"))
    assertTrue(q.offer("b"))
    assertTrue(q.offer("c"))
    assertFalse(q.offer("d"))
    assertFalse(q.offer("e"))
    assertEquals(3, q.size)
    assertEquals(2L, q.dropped)
    assertEquals("a", q.poll())
    assertTrue(q.offer("f")) // room again
    assertEquals(listOf("b", "c", "f"), generateSequence { q.poll() }.toList())
    assertEquals(0, q.size)
    assertEquals(2L, q.dropped)
  }

  @Test
  fun queueCountsDropsFromManyThreads() {
    val q = LineQueue(1000)
    val threads = (1..4).map { Thread { repeat(500) { q.offer("x") } } }
    threads.forEach { it.start() }
    threads.forEach { it.join() }
    assertEquals(1000, q.size)
    assertEquals(1000L, q.dropped)
  }

  private fun line(f: TraceFile, vararg fields: String) = CsvLine().begin(f).also { l -> fields.forEach { l.str(it) } }.end()

  @Test
  fun writerSendsEachLineToItsFileUnderAHeader() {
    val dir = File(tmp, "trace/s1")
    val errors = ArrayList<String>()
    val w = TraceWriter(dir, { site, e -> errors.add("$site $e") })
    w.offer(line(TraceFile.POSE, "p1"))
    w.offer(line(TraceFile.SENSORS, "s1"))
    w.offer(line(TraceFile.POSE, "p2"))
    w.offer(line(TraceFile.GUIDANCE, "g1"))
    w.offer("") // ignored
    w.offer("9bad\n") // unknown tag: ignored
    w.close()
    assertEquals(emptyList<String>(), errors)
    assertEquals(listOf(TraceFile.POSE.header, "p1", "p2"), File(dir, "pose.csv").readLines())
    assertEquals(listOf(TraceFile.SENSORS.header, "s1"), File(dir, "sensors.csv").readLines())
    assertEquals(listOf(TraceFile.DEPTH.header), File(dir, "depth.csv").readLines())
    assertEquals(listOf(TraceFile.GUIDANCE.header, "g1"), File(dir, "guidance.csv").readLines())
    assertEquals(dir.listFiles()!!.sumOf { it.length() }, w.bytesWritten)
    assertEquals(4L, w.linesWritten)
    // Closed: further lines are refused, not queued.
    assertFalse(w.offer(line(TraceFile.POSE, "late")))
  }

  @Test
  fun writerFlushesWithoutClosing() {
    val dir = File(tmp, "s2")
    val w = TraceWriter(dir, { _, _ -> }, flushMs = 60_000L)
    w.offer(line(TraceFile.DEPTH, "d1"))
    w.flushAsync()
    val deadline = System.currentTimeMillis() + 5000
    while (System.currentTimeMillis() < deadline && File(dir, "depth.csv").readLinesOrEmpty().size < 2) Thread.sleep(20)
    assertEquals(listOf(TraceFile.DEPTH.header, "d1"), File(dir, "depth.csv").readLines())
    w.close()
  }

  private fun File.readLinesOrEmpty(): List<String> = if (exists()) readLines() else emptyList()

  @Test
  fun writerCountsLinesDroppedWhileTheQueueIsFull() {
    // A folder that cannot be created: the writer reports it and discards every line, but the
    // queue still bounds what is held and counts what it drops.
    val blocker = File(tmp, "file").apply { writeText("x") }
    val errors = ArrayList<String>()
    val w = TraceWriter(File(blocker, "sub"), { site, _ -> synchronized(errors) { errors.add(site) } }, capacity = 4)
    var refused = 0
    repeat(10_000) { if (!w.offer(line(TraceFile.POSE, "p"))) refused++ }
    w.close()
    assertEquals(refused.toLong(), w.dropped)
    assertTrue(w.dropped > 0)
    assertEquals(listOf("dir"), errors)
    assertEquals(0L, w.linesWritten)
  }

  @Test
  fun anExistingFileIsAppendedWithoutASecondHeader() {
    val dir = File(tmp, "s3")
    TraceWriter(dir, { _, _ -> }).apply {
      offer(line(TraceFile.POSE, "a"))
      close()
    }
    TraceWriter(dir, { _, _ -> }).apply {
      offer(line(TraceFile.POSE, "b"))
      close()
    }
    assertEquals(listOf(TraceFile.POSE.header, "a", "b"), File(dir, "pose.csv").readLines())
  }
}
