package edu.cs6301.findback

import android.annotation.SuppressLint
import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Path
import android.graphics.RectF
import android.util.AttributeSet
import android.view.MotionEvent
import android.view.View
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min

/**
 * Top-down x-z sketch of the home map in a fixed map orientation (+x right, +z down, as seen from
 * above). Shows walked edges, room names, fix points, items, the target, the route and the user with
 * heading. Fits the map's bounds. Tap toggles small/large. Consumes every touch, so taps on it never
 * reach the camera view (no accidental registrations).
 *
 * [setData] is called on the UI thread with an immutable snapshot built on the GL thread.
 */
class MiniMapView @JvmOverloads constructor(context: Context, attrs: AttributeSet? = null) : View(context, attrs) {
  data class Mark(val x: Float, val z: Float, val label: String)

  data class Label(val name: String, val x: Float, val z: Float)

  /** Map-frame snapshot. [nodes] is x0, z0, x1, z1...; [edges] is index pairs into it. */
  data class Data(
    val nodes: FloatArray,
    val edges: IntArray,
    val rooms: List<Label>,
    val fixPoints: List<Mark>,
    val items: List<Mark>,
    val target: Mark?,
    /** x0, z0, x1, z1... */
    val route: FloatArray?,
    /** Null when not located (no user dot). */
    val user: Mark?,
    /** Horizontal heading (x, z) in the map frame. */
    val userForward: FloatArray?,
  )

  companion object {
    const val SMALL_DP = 120
    const val LARGE_MAX_DP = 400
    /** Smallest area shown (metres), so a short first walk is not blown up. */
    private const val MIN_SPAN_M = 3f
  }

  private var data: Data? = null
  private var expanded = false
  private val density = resources.displayMetrics.density

  private fun dp(v: Float) = v * density

  private val bgPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = Color.argb(185, 18, 18, 18) }
  private val edgePaint =
    Paint(Paint.ANTI_ALIAS_FLAG).apply {
      color = Color.argb(200, 170, 170, 170)
      strokeWidth = dp(2f)
      strokeCap = Paint.Cap.ROUND
    }
  private val routePaint =
    Paint(Paint.ANTI_ALIAS_FLAG).apply {
      color = Color.rgb(0x4F, 0xC3, 0xF7)
      strokeWidth = dp(3.5f)
      strokeCap = Paint.Cap.ROUND
      strokeJoin = Paint.Join.ROUND
      style = Paint.Style.STROKE
    }
  private val fill = Paint(Paint.ANTI_ALIAS_FLAG).apply { style = Paint.Style.FILL }
  private val ring =
    Paint(Paint.ANTI_ALIAS_FLAG).apply {
      style = Paint.Style.STROKE
      strokeWidth = dp(2.5f)
    }
  private val text =
    Paint(Paint.ANTI_ALIAS_FLAG).apply {
      color = Color.WHITE
      textAlign = Paint.Align.CENTER
      isFakeBoldText = true
      setShadowLayer(dp(2f), 0f, 0f, Color.BLACK)
    }
  private val path = Path()
  private val rect = RectF()

  init {
    isClickable = true
    contentDescription = "Home map. Tap to enlarge or shrink."
  }

  fun setData(d: Data?) {
    data = d
    visibility = if (d == null) GONE else VISIBLE
    invalidate()
  }

  @SuppressLint("ClickableViewAccessibility")
  override fun onTouchEvent(event: MotionEvent): Boolean {
    if (event.actionMasked == MotionEvent.ACTION_UP) performClick()
    return true // never pass touches through to the camera view
  }

  override fun performClick(): Boolean {
    super.performClick()
    expanded = !expanded
    val parentW = (parent as? View)?.width ?: 0
    val large = min(dp(LARGE_MAX_DP.toFloat()).toInt(), if (parentW > 0) parentW - dp(24f).toInt() else Int.MAX_VALUE)
    val size = if (expanded) large else dp(SMALL_DP.toFloat()).toInt()
    layoutParams = layoutParams.apply {
      width = size
      height = size
    }
    return true
  }

  override fun onDraw(canvas: Canvas) {
    super.onDraw(canvas)
    val d = data ?: return
    val w = width.toFloat()
    val h = height.toFloat()
    rect.set(0f, 0f, w, h)
    canvas.drawRoundRect(rect, dp(12f), dp(12f), bgPaint)

    // Fit bounds over everything drawn.
    var minX = Float.MAX_VALUE
    var maxX = -Float.MAX_VALUE
    var minZ = Float.MAX_VALUE
    var maxZ = -Float.MAX_VALUE
    fun include(x: Float, z: Float) {
      minX = min(minX, x)
      maxX = max(maxX, x)
      minZ = min(minZ, z)
      maxZ = max(maxZ, z)
    }
    for (i in 0 until d.nodes.size / 2) include(d.nodes[2 * i], d.nodes[2 * i + 1])
    d.fixPoints.forEach { include(it.x, it.z) }
    d.items.forEach { include(it.x, it.z) }
    d.target?.let { include(it.x, it.z) }
    d.user?.let { include(it.x, it.z) }
    if (minX > maxX) return
    var spanX = maxX - minX
    var spanZ = maxZ - minZ
    if (spanX < MIN_SPAN_M) {
      minX -= (MIN_SPAN_M - spanX) / 2
      spanX = MIN_SPAN_M
    }
    if (spanZ < MIN_SPAN_M) {
      minZ -= (MIN_SPAN_M - spanZ) / 2
      spanZ = MIN_SPAN_M
    }
    val pad = dp(if (expanded) 22f else 12f)
    val scale = min((w - 2 * pad) / spanX, (h - 2 * pad) / spanZ)
    val ox = (w - spanX * scale) / 2 - minX * scale
    val oz = (h - spanZ * scale) / 2 - minZ * scale
    fun sx(x: Float) = ox + x * scale
    fun sz(z: Float) = oz + z * scale

    // Walked edges.
    for (k in 0 until d.edges.size / 2) {
      val a = d.edges[2 * k]
      val b = d.edges[2 * k + 1]
      canvas.drawLine(sx(d.nodes[2 * a]), sz(d.nodes[2 * a + 1]), sx(d.nodes[2 * b]), sz(d.nodes[2 * b + 1]), edgePaint)
    }
    // Route.
    d.route?.let { r ->
      if (r.size >= 4) {
        path.reset()
        path.moveTo(sx(r[0]), sz(r[1]))
        for (i in 1 until r.size / 2) path.lineTo(sx(r[2 * i]), sz(r[2 * i + 1]))
        canvas.drawPath(path, routePaint)
      }
    }
    // Room names at the centroid of their nodes.
    text.textSize = dp(if (expanded) 14f else 10f)
    text.color = Color.argb(230, 255, 255, 255)
    for (l in d.rooms) canvas.drawText(l.name, sx(l.x), sz(l.z), text)
    // Fix points: picture = magenta square, spot = yellow diamond.
    val fs = dp(if (expanded) 6f else 4f)
    for (f in d.fixPoints) {
      val x = sx(f.x)
      val y = sz(f.z)
      fill.color = if (f.label == "IMAGE") Color.rgb(0xE0, 0x40, 0xFB) else Color.rgb(0xFF, 0xEB, 0x3B)
      if (f.label == "IMAGE") canvas.drawRect(x - fs, y - fs, x + fs, y + fs, fill)
      else {
        path.reset()
        path.moveTo(x, y - fs * 1.3f)
        path.lineTo(x + fs * 1.3f, y)
        path.lineTo(x, y + fs * 1.3f)
        path.lineTo(x - fs * 1.3f, y)
        path.close()
        canvas.drawPath(path, fill)
      }
    }
    // Items.
    fill.color = Color.rgb(0xFF, 0xC1, 0x07)
    text.textSize = dp(11f)
    for (it in d.items) {
      canvas.drawCircle(sx(it.x), sz(it.z), dp(if (expanded) 4f else 2.5f), fill)
      if (expanded) canvas.drawText(it.label, sx(it.x), sz(it.z) - dp(7f), text)
    }
    // Target.
    d.target?.let {
      ring.color = Color.rgb(0x34, 0xD3, 0x99)
      canvas.drawCircle(sx(it.x), sz(it.z), dp(if (expanded) 9f else 6f), ring)
    }
    // User with heading.
    d.user?.let { u ->
      val x = sx(u.x)
      val y = sz(u.z)
      d.userForward?.let { f ->
        val len = hypot(f[0], f[1])
        if (len > 1e-3f) {
          val fx = f[0] / len
          val fz = f[1] / len
          val r = dp(if (expanded) 16f else 11f)
          val s = dp(if (expanded) 6f else 4.5f)
          path.reset()
          path.moveTo(x + fx * r, y + fz * r)
          path.lineTo(x - fz * s, y + fx * s)
          path.lineTo(x + fz * s, y - fx * s)
          path.close()
          fill.color = Color.argb(200, 0x4F, 0xC3, 0xF7)
          canvas.drawPath(path, fill)
        }
      }
      fill.color = Color.WHITE
      canvas.drawCircle(x, y, dp(if (expanded) 6f else 4.5f), fill)
      fill.color = Color.rgb(0x21, 0x96, 0xF3)
      canvas.drawCircle(x, y, dp(if (expanded) 4.5f else 3.2f), fill)
    }
    // Scale bar (1 m) when large.
    if (expanded) {
      val y = h - dp(10f)
      canvas.drawLine(dp(12f), y, dp(12f) + scale, y, edgePaint)
      text.textSize = dp(10f)
      text.textAlign = Paint.Align.LEFT
      canvas.drawText("1 m", dp(14f) + scale, y + dp(3f), text)
      text.textAlign = Paint.Align.CENTER
    }
  }
}
