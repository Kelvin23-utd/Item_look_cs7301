package edu.cs6301.findback.trace

import android.app.ActivityManager
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.os.BatteryManager
import android.os.Build
import android.os.Debug
import android.os.Handler
import android.os.HandlerThread
import android.os.PowerManager
import android.os.SystemClock
import android.util.Log
import androidx.lifecycle.DefaultLifecycleObserver
import androidx.lifecycle.LifecycleOwner
import edu.cs6301.findback.memory.EventLog
import java.io.File
import kotlin.math.abs

/**
 * v0.7 detailed trace (docs/PLAN_v0.7_trace.md): everything that runs only while "Detailed trace"
 * is on and is not tied to the GL thread. Created in `MainActivity.onCreate` when the setting is
 * on; with it off there is no instance, so no folder, no thread and no listener exist.
 *
 *  - the [TraceWriter] of `trace/<sessionId>/`;
 *  - one background [HandlerThread] for the sensor callbacks ([SensorRecorder], registered in
 *    onResume, removed in onPause) and the `device_state` samples (start, every 30 s while
 *    resumed, on pause);
 *  - `app_pause` and `app_resume`.
 */
class TraceSession(
  private val context: Context,
  private val events: EventLog,
  root: File,
  sessionId: String,
  /** `SystemClock.elapsedRealtime()` at session start, for `sessionMs`. */
  private val startElapsedMs: Long,
) : DefaultLifecycleObserver {
  companion object {
    private const val TAG = "FindBackTrace"
    const val DIR = "trace"
    const val DEVICE_STATE_MS = 30_000L

    /** Total size of every session's trace folder under [root], in bytes (Settings line). Any thread. */
    fun storageBytes(root: File): Long {
      val dir = File(root, DIR)
      if (!dir.isDirectory) return 0L
      var total = 0L
      dir.walkTopDown().forEach { if (it.isFile) total += it.length() }
      return total
    }
  }

  private val filesRoot = root
  val writer = TraceWriter(File(File(root, DIR), sessionId), onError = { site, message -> error(site, message) })
  private val thread = HandlerThread("FindBackTrace").apply { start() }
  private val handler = Handler(thread.looper)
  private val sensors = SensorRecorder(context, writer) { site, message -> error(site, message) }
  @Volatile private var closed = false

  private val deviceTick =
    object : Runnable {
      override fun run() {
        logDeviceState("periodic")
        handler.postDelayed(this, DEVICE_STATE_MS)
      }
    }

  init {
    handler.post { logDeviceState("start") }
  }

  /** `trace_error` (any thread). */
  fun error(site: String, error: String) {
    Log.w(TAG, "trace error at $site: $error")
    events.log("trace_error", mapOf("site" to site, "error" to error))
  }

  private fun sessionMs() = SystemClock.elapsedRealtime() - startElapsedMs

  override fun onResume(owner: LifecycleOwner) {
    if (closed) return
    events.log("app_resume", mapOf("sessionMs" to sessionMs()))
    // Registered here, on the UI thread; the callbacks run on [thread].
    sensors.start(handler)
    handler.removeCallbacks(deviceTick)
    handler.postDelayed(deviceTick, DEVICE_STATE_MS)
  }

  override fun onPause(owner: LifecycleOwner) {
    if (closed) return
    handler.removeCallbacks(deviceTick)
    sensors.stop()
    handler.post { logDeviceState("pause") }
    writer.flushAsync()
    events.log(
      "app_pause",
      mapOf("sessionMs" to sessionMs(), "traceBytes" to writer.bytesWritten, "traceDropped" to writer.dropped),
    )
  }

  override fun onDestroy(owner: LifecycleOwner) = close()

  fun close() {
    if (closed) return
    closed = true
    handler.removeCallbacks(deviceTick)
    sensors.stop()
    thread.quitSafely()
    writer.close()
  }

  // ------------------------------------------------------------------ device_state

  /** Background thread. Every value is read without a permission; a value that fails is null. */
  private fun logDeviceState(trigger: String) {
    val fields = LinkedHashMap<String, Any?>()
    fields["trigger"] = trigger
    fields["sessionMs"] = sessionMs()
    try {
      // The sticky battery broadcast: no receiver is registered, nothing needs a permission.
      val filter = IntentFilter(Intent.ACTION_BATTERY_CHANGED)
      val battery =
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) context.registerReceiver(null, filter, Context.RECEIVER_NOT_EXPORTED)
        else context.registerReceiver(null, filter)
      if (battery != null) {
        val level = battery.getIntExtra(BatteryManager.EXTRA_LEVEL, -1)
        val scale = battery.getIntExtra(BatteryManager.EXTRA_SCALE, -1)
        fields["batteryPct"] = if (level >= 0 && scale > 0) 100.0 * level / scale else null
        val temp = battery.getIntExtra(BatteryManager.EXTRA_TEMPERATURE, Int.MIN_VALUE)
        fields["batteryTempC"] = if (temp != Int.MIN_VALUE) temp / 10.0 else null
        val status = battery.getIntExtra(BatteryManager.EXTRA_STATUS, -1)
        val plugged = battery.getIntExtra(BatteryManager.EXTRA_PLUGGED, 0)
        fields["charging"] =
          status == BatteryManager.BATTERY_STATUS_CHARGING || status == BatteryManager.BATTERY_STATUS_FULL || plugged != 0
      }
      val bm = context.getSystemService(BatteryManager::class.java)
      if (bm != null) {
        val now = bm.getIntProperty(BatteryManager.BATTERY_PROPERTY_CURRENT_NOW)
        val charge = bm.getIntProperty(BatteryManager.BATTERY_PROPERTY_CHARGE_COUNTER)
        // Integer.MIN_VALUE (or 0 on some devices) means "not supported".
        fields["currentNowMa"] = if (now != Int.MIN_VALUE && now != 0) now / 1000.0 else null
        fields["chargeCounterMah"] = if (charge != Int.MIN_VALUE && charge > 0) charge / 1000.0 else null
      }
    } catch (e: Exception) {
      fields["batteryError"] = e.toString()
    }
    try {
      val pm = context.getSystemService(PowerManager::class.java)
      if (pm != null && Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
        fields["thermalStatus"] = pm.currentThermalStatus
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
          val h = pm.getThermalHeadroom(10)
          fields["thermalHeadroom"] = if (h.isNaN()) null else h.toDouble()
        }
      }
    } catch (e: Exception) {
      fields["thermalError"] = e.toString()
    }
    try {
      val am = context.getSystemService(ActivityManager::class.java)
      if (am != null) {
        val mi = ActivityManager.MemoryInfo()
        am.getMemoryInfo(mi)
        fields["availMemMb"] = mi.availMem / (1024 * 1024)
        fields["lowMemory"] = mi.lowMemory
      }
      val rt = Runtime.getRuntime()
      fields["javaHeapMb"] = (rt.totalMemory() - rt.freeMemory()) / (1024 * 1024)
      fields["nativeHeapMb"] = Debug.getNativeHeapAllocatedSize() / (1024 * 1024)
    } catch (e: Exception) {
      fields["memoryError"] = e.toString()
    }
    try {
      fields["freeStorageMb"] = filesRoot.usableSpace / (1024 * 1024)
    } catch (e: Exception) {
      fields["storageError"] = e.toString()
    }
    fields["traceBytes"] = writer.bytesWritten
    fields["traceDropped"] = writer.dropped
    events.log("device_state", fields)
  }
}

/**
 * The motion and environment sensors of `sensors.csv` (section 4.2). The app's logic never reads
 * them. Callbacks run on the trace's background thread; each kept event is formatted into a
 * reused [CsvLine] and offered to the writer. Android may deliver events faster than asked (other
 * apps, ARCore itself, keep the IMU running fast), so each type is thinned by its sensor clock to
 * its rate.
 */
class SensorRecorder(
  context: Context,
  private val writer: TraceWriter,
  private val onError: (site: String, error: String) -> Unit,
) : SensorEventListener {
  private class Spec(val name: String, val type: Int, val periodUs: Int, val decimals: Int, val values: Int) {
    val gate = RateGate(periodUs * 1000L)
  }

  private val specs =
    arrayOf(
      Spec("acc", Sensor.TYPE_ACCELEROMETER, 20_000, 4, 3),
      Spec("gyro", Sensor.TYPE_GYROSCOPE, 20_000, 5, 3),
      Spec("mag", Sensor.TYPE_MAGNETIC_FIELD, 100_000, 2, 3),
      Spec("baro", Sensor.TYPE_PRESSURE, 1_000_000, 3, 1),
      // On change; at most 5 Hz.
      Spec("light", Sensor.TYPE_LIGHT, 200_000, 1, 1),
    )

  private val sm: SensorManager? = context.getSystemService(SensorManager::class.java)
  private val line = CsvLine(96)
  private val reported = HashSet<String>()
  private var registered = false

  /** UI thread (onResume): register every listener with [handler], whose thread runs the callbacks. */
  fun start(handler: Handler) {
    if (registered) return
    val manager = sm
    if (manager == null) {
      report("sensors", "no SensorManager")
      return
    }
    for (s in specs) {
      val sensor = manager.getDefaultSensor(s.type)
      if (sensor == null) {
        report("sensor:" + s.name, "not available on this device")
        continue
      }
      s.gate.reset()
      val ok =
        try {
          manager.registerListener(this, sensor, s.periodUs, handler)
        } catch (e: Exception) {
          report("sensor:" + s.name, e.toString())
          false
        }
      if (!ok) report("sensor:" + s.name, "registerListener returned false")
    }
    registered = true
  }

  /** UI thread (onPause, onDestroy). */
  fun stop() {
    if (!registered) return
    registered = false
    try {
      sm?.unregisterListener(this)
    } catch (e: Exception) {
      Log.w("FindBackTrace", "unregisterListener failed", e)
    }
  }

  /** A missing or failing sensor is reported once per session. */
  private fun report(site: String, error: String) {
    if (reported.add(site)) onError(site, error)
  }

  override fun onSensorChanged(e: SensorEvent) {
    val s =
      when (e.sensor.type) {
        Sensor.TYPE_ACCELEROMETER -> specs[0]
        Sensor.TYPE_GYROSCOPE -> specs[1]
        Sensor.TYPE_MAGNETIC_FIELD -> specs[2]
        Sensor.TYPE_PRESSURE -> specs[3]
        Sensor.TYPE_LIGHT -> specs[4]
        else -> return
      }
    if (!s.gate.accept(e.timestamp)) return
    // Wall time of the event from its own timestamp (elapsedRealtimeNanos base on this phone);
    // fall back to now if the clocks disagree by more than 10 s.
    val nowWall = System.currentTimeMillis()
    val ageNs = SystemClock.elapsedRealtimeNanos() - e.timestamp
    val t = if (abs(ageNs) < 10_000_000_000L) nowWall - ageNs / 1_000_000L else nowWall
    val v = e.values
    line.begin(TraceFile.SENSORS).long(t).long(e.timestamp).str(s.name)
    line.num(v[0], s.decimals)
    if (s.values >= 3 && v.size >= 3) line.num(v[1], s.decimals).num(v[2], s.decimals) else line.empty().empty()
    line.int(e.accuracy)
    writer.offer(line.end())
  }

  override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {}
}
