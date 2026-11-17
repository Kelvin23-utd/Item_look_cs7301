package edu.cs6301.findback.trace

import java.io.BufferedWriter
import java.io.File
import java.io.FileOutputStream
import java.io.OutputStreamWriter
import java.util.concurrent.ConcurrentLinkedQueue
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicInteger
import java.util.concurrent.atomic.AtomicLong
import java.util.concurrent.locks.LockSupport

/**
 * Bounded multi-producer queue of trace lines (section 4.5). [offer] never blocks and never waits
 * for space: when [capacity] lines are queued the line is dropped and counted. Lock-free, so the
 * GL thread and the sensor thread never wait for the writer.
 */
class LineQueue(val capacity: Int) {
  private val q = ConcurrentLinkedQueue<String>()
  private val count = AtomicInteger()
  private val droppedCount = AtomicLong()

  fun offer(line: String): Boolean {
    if (count.incrementAndGet() > capacity) {
      count.decrementAndGet()
      droppedCount.incrementAndGet()
      return false
    }
    q.offer(line)
    return true
  }

  fun poll(): String? {
    val s = q.poll() ?: return null
    count.decrementAndGet()
    return s
  }

  val size: Int
    get() = count.get()

  val dropped: Long
    get() = droppedCount.get()
}

/**
 * Owns the four CSV files of one session's trace folder (section 4.5): a single background thread
 * drains a [LineQueue] into [BufferedWriter]s, flushed every [flushMs] and on [flushAsync]. Lines
 * come from [CsvLine]: their first character names the file. Pure JVM (java.io and
 * java.util.concurrent), unit-tested in TraceWriterTest.
 *
 * [onError] is called (on the writer thread) once per file that cannot be opened or written; the
 * lines of such a file are discarded from then on.
 */
class TraceWriter(
  val dir: File,
  private val onError: (site: String, error: String) -> Unit,
  capacity: Int = DEFAULT_CAPACITY,
  private val flushMs: Long = DEFAULT_FLUSH_MS,
) {
  companion object {
    const val DEFAULT_CAPACITY = 8192
    const val DEFAULT_FLUSH_MS = 2000L
    /** How long the writer sleeps when the queue is empty. */
    private const val IDLE_PARK_MS = 50L
  }

  private val queue = LineQueue(capacity)
  private val writers = arrayOfNulls<BufferedWriter>(TraceFile.values().size)
  private val failed = BooleanArray(TraceFile.values().size)
  private val bytes = AtomicLong()
  private val lines = AtomicLong()
  @Volatile private var running = true
  @Volatile private var flushRequested = false
  private val thread = Thread({ loop() }, "FindBackTraceWriter").apply {
    isDaemon = true
    priority = Thread.NORM_PRIORITY - 1
  }

  init {
    thread.start()
  }

  /**
   * Any thread: queue one line from [CsvLine.end]. False when it was dropped (queue full or
   * closed). Never blocks and never wakes the writer, which polls every [IDLE_PARK_MS].
   */
  fun offer(line: String): Boolean {
    if (!running) return false
    return queue.offer(line)
  }

  /** Bytes written to the files so far (headers included). */
  val bytesWritten: Long
    get() = bytes.get()

  val linesWritten: Long
    get() = lines.get()

  val dropped: Long
    get() = queue.dropped

  val queued: Int
    get() = queue.size

  /** Any thread: ask the writer to flush its files soon (on pause). Does not wait. */
  fun flushAsync() {
    flushRequested = true
    LockSupport.unpark(thread)
  }

  /** Write what is queued, flush and close the files, and stop the thread (waits up to [timeoutMs]). */
  fun close(timeoutMs: Long = 2000L) {
    running = false
    LockSupport.unpark(thread)
    try {
      thread.join(timeoutMs)
    } catch (e: InterruptedException) {
      Thread.currentThread().interrupt()
    }
  }

  private fun loop() {
    openAll()
    var lastFlush = System.nanoTime()
    val flushNs = TimeUnit.MILLISECONDS.toNanos(flushMs)
    while (true) {
      val s = queue.poll()
      if (s != null) {
        write(s)
      } else {
        if (!running) break
      }
      val now = System.nanoTime()
      if (flushRequested || now - lastFlush >= flushNs) {
        flushRequested = false
        flushAll()
        lastFlush = now
      }
      if (s == null) LockSupport.parkNanos(this, TimeUnit.MILLISECONDS.toNanos(IDLE_PARK_MS))
    }
    // Lines offered while stopping are written too.
    while (true) write(queue.poll() ?: break)
    flushAll()
    for (i in writers.indices) {
      try {
        writers[i]?.close()
      } catch (e: Exception) {
        // already reported by flushAll if it matters
      }
      writers[i] = null
    }
  }

  private fun openAll() {
    try {
      if (!dir.isDirectory && !dir.mkdirs()) throw java.io.IOException("cannot create $dir")
    } catch (e: Exception) {
      for (f in TraceFile.values()) failed[f.ordinal] = true
      onError("dir", e.toString())
      return
    }
    for (f in TraceFile.values()) {
      try {
        val file = File(dir, f.fileName)
        val existed = file.exists() && file.length() > 0
        val w = BufferedWriter(OutputStreamWriter(FileOutputStream(file, true), Charsets.UTF_8), 16 * 1024)
        if (!existed) {
          w.write(f.header)
          w.write("\n")
          bytes.addAndGet(f.header.length + 1L)
        }
        writers[f.ordinal] = w
      } catch (e: Exception) {
        failed[f.ordinal] = true
        onError("open:" + f.fileName, e.toString())
      }
    }
  }

  private fun write(s: String) {
    if (s.length < 2) return
    val f = TraceFile.ofTag(s[0]) ?: return
    val w = writers[f.ordinal] ?: return
    try {
      w.write(s, 1, s.length - 1)
      bytes.addAndGet(s.length - 1L) // the lines are ASCII
      lines.incrementAndGet()
    } catch (e: Exception) {
      fail(f, "write", e)
    }
  }

  private fun flushAll() {
    for (f in TraceFile.values()) {
      val w = writers[f.ordinal] ?: continue
      try {
        w.flush()
      } catch (e: Exception) {
        fail(f, "flush", e)
      }
    }
  }

  private fun fail(f: TraceFile, what: String, e: Exception) {
    if (!failed[f.ordinal]) {
      failed[f.ordinal] = true
      onError(what + ":" + f.fileName, e.toString())
    }
    try {
      writers[f.ordinal]?.close()
    } catch (ignored: Exception) {
    }
    writers[f.ordinal] = null
  }
}
