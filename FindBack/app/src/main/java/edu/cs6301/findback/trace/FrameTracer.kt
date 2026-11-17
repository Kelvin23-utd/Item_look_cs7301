package edu.cs6301.findback.trace

import com.google.ar.core.Camera
import com.google.ar.core.Frame
import com.google.ar.core.LightEstimate
import com.google.ar.core.Plane
import com.google.ar.core.Session
import com.google.ar.core.TrackingFailureReason
import com.google.ar.core.TrackingState
import com.google.ar.core.exceptions.NotYetAvailableException
import edu.cs6301.findback.map.Vec3
import edu.cs6301.findback.memory.EventLog
import edu.cs6301.findback.scan.ScanConfig
import java.nio.ByteOrder
import kotlin.math.atan2

/**
 * v0.7 detailed trace, the GL-thread part (docs/PLAN_v0.7_trace.md, sections 3 and 4): samples
 * `pose.csv` (10 Hz), `depth.csv` (2 Hz) and `guidance.csv` (5 Hz while a query is active), and
 * keeps the frame times for `perf` (every 10 s). GL thread only.
 *
 * Every line is formatted into one reused [CsvLine] and offered to the [TraceWriter], which never
 * blocks. The depth image is acquired, read on a subsample and closed in the same call.
 */
class FrameTracer(
  private val writer: TraceWriter,
  private val events: EventLog,
  private val onError: (site: String, error: String) -> Unit,
) {
  companion object {
    const val POSE_HZ = 10.0
    const val DEPTH_HZ = 2.0
    const val GUIDANCE_HZ = 5.0
    const val PERF_WINDOW_MS = 10_000L
    /** A gap this long between frames (the app was paused) restarts the `perf` window. */
    const val PERF_GAP_MS = 2_000L
    /** Depth statistics use every DEPTH_STEP-th pixel in both directions. */
    const val DEPTH_STEP = 4
  }

  private val line = CsvLine(320)
  private val poseGate = RateGate.hz(POSE_HZ)
  private val depthGate = RateGate.hz(DEPTH_HZ)
  private val guidanceGate = RateGate.hz(GUIDANCE_HZ)
  private val depthStats = DepthStats(minMm = (ScanConfig.DEPTH_MIN_M * 1000).toInt(), maxMm = (ScanConfig.DEPTH_MAX_M * 1000).toInt())

  // Previous tracking pose sample, for speed and turn rate.
  private var havePrev = false
  private var prevNs = 0L
  private var px = 0.0
  private var py = 0.0
  private var pz = 0.0
  private var pqx = 0.0
  private var pqy = 0.0
  private var pqz = 0.0
  private var pqw = 1.0
  /** Speed (m/s) and turn rate (deg/s) of the latest pose sample; NaN when unknown. */
  var speed = Double.NaN
    private set
  var turnRate = Double.NaN
    private set

  private val perf = FrameStats()
  private var perfStartNs = 0L
  private var lastFrameEndNs = 0L
  private val reported = HashSet<String>()

  /** Time spent tracing in the current frame (added to the `perf` statistics at [frameEnd]). */
  private var traceNsThisFrame = 0L

  // ------------------------------------------------------------------ gates

  fun poseDue(ns: Long) = poseGate.accept(ns)

  fun depthDue(ns: Long) = depthGate.accept(ns)

  fun guidanceDue(ns: Long) = guidanceGate.accept(ns)

  /** Adds tracing time measured by the caller to this frame's share. */
  fun addTraceNs(ns: Long) {
    traceNsThisFrame += ns
  }

  // ------------------------------------------------------------------ pose.csv

  /**
   * One `pose.csv` row. [userMap], [userForwardMap]: the camera in the map frame (from the
   * localizer) when located and tracking, else null. Heading `myaw` is in degrees with the
   * convention of `Pose4.yaw` for a frame whose -Z is the viewing direction.
   */
  fun pose(
    ns: Long,
    camera: Camera,
    located: Boolean,
    userMap: Vec3?,
    userForwardMap: Vec3?,
    metersSinceFix: Double?,
    screen: String?,
    room: String?,
    query: String?,
  ) {
    val state = camera.trackingState
    val tracking = state == TrackingState.TRACKING
    val failure = camera.trackingFailureReason
    line.begin(TraceFile.POSE).long(System.currentTimeMillis()).long(ns).str(state.name)
    line.str(if (failure == TrackingFailureReason.NONE) null else failure.name)
    if (tracking) {
      val p = camera.pose
      val x = p.tx().toDouble()
      val y = p.ty().toDouble()
      val z = p.tz().toDouble()
      val qx = p.qx().toDouble()
      val qy = p.qy().toDouble()
      val qz = p.qz().toDouble()
      val qw = p.qw().toDouble()
      if (havePrev) {
        val dt = ns - prevNs
        speed = Motion.speed(px, py, pz, x, y, z, dt)
        turnRate = Motion.turnRate(pqx, pqy, pqz, pqw, qx, qy, qz, qw, dt)
      } else {
        speed = Double.NaN
        turnRate = Double.NaN
      }
      havePrev = true
      prevNs = ns
      px = x
      py = y
      pz = z
      pqx = qx
      pqy = qy
      pqz = qz
      pqw = qw
      line.num(x, 4).num(y, 4).num(z, 4).num(qx, 5).num(qy, 5).num(qz, 5).num(qw, 5)
    } else {
      // No speed across a tracking gap.
      havePrev = false
      speed = Double.NaN
      turnRate = Double.NaN
      line.empty().empty().empty().empty().empty().empty().empty()
    }
    line.bool(located)
    if (tracking && located && userMap != null) {
      line.num(userMap.x, 4).num(userMap.y, 4).num(userMap.z, 4)
      if (userForwardMap != null) line.num(Math.toDegrees(atan2(-userForwardMap.x, -userForwardMap.z)), 2) else line.empty()
    } else {
      line.empty().empty().empty().empty()
    }
    line.num(metersSinceFix ?: Double.NaN, 3)
    line.num(speed, 3).num(turnRate, 2)
    line.str(screen).str(room).str(query)
    writer.offer(line.end())
  }

  // ------------------------------------------------------------------ depth.csv

  /** One `depth.csv` row: the depth image (acquired and closed here), the scene and the light. */
  fun depth(ns: Long, frame: Frame, session: Session) {
    line.begin(TraceFile.DEPTH).long(System.currentTimeMillis()).long(ns)
    val image =
      try {
        frame.acquireDepthImage16Bits()
      } catch (e: NotYetAvailableException) {
        null
      } catch (e: Exception) {
        reportOnce("depth_acquire", e.toString()) // depth unsupported or disabled
        null
      }
    if (image != null) {
      var w = 0
      var h = 0
      try {
        w = image.width
        h = image.height
        val plane = image.planes[0]
        val buf = plane.buffer.duplicate().order(ByteOrder.LITTLE_ENDIAN)
        val rowStride = plane.rowStride
        val pixelStride = plane.pixelStride
        depthStats.compute(w, h, DEPTH_STEP) { x, y -> buf.getShort(y * rowStride + x * pixelStride).toInt() and 0xffff }
      } finally {
        image.close()
      }
      line.int(1).int(w).int(h)
      line.num(depthStats.validShare, 4).num(depthStats.usableShare, 4)
      line.num(depthStats.medianM, 3).num(depthStats.p10M, 3).num(depthStats.p90M, 3).num(depthStats.centreM, 3)
    } else {
      line.int(0).empty().empty().empty().empty().empty().empty().empty().empty()
    }
    // Feature points of this frame's point cloud.
    var points = -1
    try {
      frame.acquirePointCloud().use { pc -> points = pc.points.remaining() / 4 }
    } catch (e: Exception) {
      reportOnce("point_cloud", e.toString())
    }
    if (points >= 0) line.int(points) else line.empty()
    var planes = 0
    var area = 0.0
    for (p in session.getAllTrackables(Plane::class.java)) {
      if (p.trackingState != TrackingState.TRACKING || p.subsumedBy != null) continue
      planes++
      area += p.extentX.toDouble() * p.extentZ
    }
    line.int(planes).num(area, 3)
    val light = frame.lightEstimate
    line.num(if (light.state == LightEstimate.State.VALID) light.pixelIntensity.toDouble() else Double.NaN, 4)
    line.num(speed, 3).num(turnRate, 2)
    writer.offer(line.end())
  }

  // ------------------------------------------------------------------ guidance.csv

  /**
   * One `guidance.csv` row. Distance and angles from the camera (display-oriented) to the item's
   * remembered position, NaN when it has no position or tracking is lost; [inView] null then.
   * [ringX], [ringY]: the item's screen position in view pixels as the overlay computes it.
   */
  fun guidance(
    ns: Long,
    query: String,
    mode: String?,
    resolvable: Boolean,
    distance: Double,
    bearingDeg: Double,
    elevationDeg: Double,
    routeM: Double?,
    straightM: Double?,
    inView: Boolean?,
    ringX: Double,
    ringY: Double,
  ) {
    line.begin(TraceFile.GUIDANCE).long(System.currentTimeMillis()).long(ns).str(query).str(mode).bool(resolvable)
    line.num(distance, 3).num(bearingDeg, 1).num(elevationDeg, 1)
    line.num(routeM ?: Double.NaN, 2).num(straightM ?: Double.NaN, 2)
    if (inView == null) line.empty() else line.bool(inView)
    line.num(ringX, 0).num(ringY, 0)
    writer.offer(line.end())
  }

  // ------------------------------------------------------------------ perf

  /**
   * End of a GL frame that started at [startNs] (System.nanoTime). Every [PERF_WINDOW_MS] this
   * logs `perf`, at most 0.1 Hz.
   */
  fun frameEnd(startNs: Long, mlBusy: Boolean) {
    val end = System.nanoTime()
    if (lastFrameEndNs != 0L && startNs - lastFrameEndNs > PERF_GAP_MS * 1_000_000L) {
      perf.reset() // the app was paused: start a new window
      perfStartNs = 0L
    }
    if (perfStartNs == 0L) perfStartNs = startNs
    perf.add((end - startNs) / 1e6, mlBusy, traceNsThisFrame / 1e6)
    traceNsThisFrame = 0L
    lastFrameEndNs = end
    val windowNs = end - perfStartNs
    if (windowNs >= PERF_WINDOW_MS * 1_000_000L) {
      val seconds = windowNs / 1e9
      events.log(
        "perf",
        mapOf(
          "frames" to perf.frames,
          "fps" to round(perf.frames / seconds, 2),
          "frameMsMean" to round(perf.meanMs, 3),
          "frameMsP95" to round(perf.p95Ms(), 3),
          "frameMsMax" to round(perf.maxMs, 3),
          "mlBusyShare" to round(perf.busyShare, 4),
          "traceQueue" to writer.queued,
          "traceDropped" to writer.dropped,
          // v0.7 extra: the GL time the trace itself took per frame (included in frameMs).
          "traceMsMean" to round(perf.traceMeanMs, 4),
          "traceMsMax" to round(perf.traceMaxMs, 3),
          "windowS" to round(seconds, 2),
        ),
      )
      perf.reset()
      perfStartNs = end
    }
  }

  private fun round(v: Double, d: Int): Double? {
    if (v.isNaN() || v.isInfinite()) return null
    val s = Math.pow(10.0, d.toDouble())
    return Math.round(v * s) / s
  }

  private fun reportOnce(site: String, error: String) {
    if (reported.add(site)) onError(site, error)
  }
}
