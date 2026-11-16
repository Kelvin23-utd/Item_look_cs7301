package edu.cs6301.findback

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Path
import android.graphics.PointF
import android.os.Build
import android.os.SystemClock
import android.util.AttributeSet
import android.view.HapticFeedbackConstants
import android.view.View
import kotlin.math.abs
import kotlin.math.atan2
import kotlin.math.hypot
import kotlin.math.min
import kotlin.math.sin

/**
 * Screen-space guidance drawn over the AR view.
 *  - Target on screen: pulsing ring + distance, one haptic tick when it reaches the centre.
 *  - Waypoint (v0.3 FAR_ROUTE) on screen: a chevron, never the ring, with the walk length.
 *  - Target or waypoint off screen or behind: arrow on the screen edge pointing the way to turn.
 * Never consumes touches, so taps still reach the GLSurfaceView.
 */
class GuidanceOverlay @JvmOverloads constructor(context: Context, attrs: AttributeSet? = null) :
  View(context, attrs) {

  /** Snapshot produced on the GL thread each frame. */
  data class Target(
    /** Normalised device coordinates; meaningful only when [inFront]. */
    val ndcX: Float,
    val ndcY: Float,
    val inFront: Boolean,
    /** Camera-local position (x right, y up, metres) for choosing a turn direction. */
    val localX: Float,
    val localY: Float,
    val distance: Float,
    val state: State,
    /** RING: the item itself (NEAR). CHEVRON: a route waypoint 1.5 m ahead (FAR_ROUTE). */
    val kind: Kind = Kind.RING,
    /** Text under the marker instead of the straight distance (e.g. "7 m walk"). */
    val label: String? = null,
  )

  enum class State { UNVERIFIED, CONFIRMED, NOT_CONFIRMED }

  enum class Kind { RING, CHEVRON }

  companion object {
    private val CHEVRON_COLOR = Color.rgb(0xFF, 0xFF, 0xFF)
  }

  @Volatile private var target: Target? = null
  private var centeredOnce = false
  private var tapPoint: PointF? = null
  private var tapTime = 0L

  private val density = resources.displayMetrics.density
  private fun dp(v: Float) = v * density

  /** Keep these clear of the status panel (top) and guide card + buttons (bottom). */
  private val topInset = dp(150f)
  private val bottomInset = dp(290f)
  private val sideInset = dp(40f)

  private val ringPaint =
    Paint(Paint.ANTI_ALIAS_FLAG).apply {
      style = Paint.Style.STROKE
      strokeWidth = dp(4f)
    }
  private val haloPaint =
    Paint(Paint.ANTI_ALIAS_FLAG).apply {
      style = Paint.Style.STROKE
      strokeWidth = dp(10f)
      color = Color.argb(90, 0, 0, 0)
    }
  private val fillPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply { style = Paint.Style.FILL }
  private val textPaint =
    Paint(Paint.ANTI_ALIAS_FLAG).apply {
      color = Color.WHITE
      textSize = dp(16f)
      textAlign = Paint.Align.CENTER
      isFakeBoldText = true
      setShadowLayer(dp(3f), 0f, 0f, Color.BLACK)
    }
  private val arrowPath = Path()

  init {
    isClickable = false
    isFocusable = false
  }

  fun setTarget(t: Target?) {
    if (t == null) centeredOnce = false
    target = t
    postInvalidateOnAnimation()
  }

  fun flashTap(x: Float, y: Float) {
    tapPoint = PointF(x, y)
    tapTime = SystemClock.uptimeMillis()
    postInvalidateOnAnimation()
  }

  private fun colorFor(s: State) =
    when (s) {
      State.UNVERIFIED -> Color.rgb(0x4F, 0xC3, 0xF7)
      State.CONFIRMED -> Color.rgb(0x34, 0xD3, 0x99)
      State.NOT_CONFIRMED -> Color.rgb(0xFF, 0xC1, 0x07)
    }

  override fun onDraw(canvas: Canvas) {
    super.onDraw(canvas)
    val now = SystemClock.uptimeMillis()
    drawTapFlash(canvas, now)
    val t = target ?: return
    val w = width.toFloat()
    val h = height.toFloat()
    val color = if (t.kind == Kind.CHEVRON) CHEVRON_COLOR else colorFor(t.state)
    val sx = (t.ndcX + 1f) / 2f * w
    val sy = (1f - t.ndcY) / 2f * h
    val onScreen =
      t.inFront && sx in sideInset..(w - sideInset) && sy in topInset..(h - bottomInset)

    if (onScreen && t.kind == Kind.CHEVRON) {
      drawChevron(canvas, sx, sy, now, color)
      canvas.drawText(t.label ?: "%.1f m".format(t.distance), sx, sy + dp(48f), textPaint)
    } else if (onScreen) {
      val pulse = 1f + 0.12f * sin(now / 180.0).toFloat()
      val r = dp(38f) * pulse
      ringPaint.color = color
      canvas.drawCircle(sx, sy, r, haloPaint)
      canvas.drawCircle(sx, sy, r, ringPaint)
      fillPaint.color = color
      canvas.drawCircle(sx, sy, dp(5f), fillPaint)
      canvas.drawText("%.1f m".format(t.distance), sx, sy + r + dp(22f), textPaint)

      val centered = hypot(sx - w / 2f, sy - h / 2f) < w * 0.15f
      if (centered && !centeredOnce) {
        centeredOnce = true
        performHapticFeedback(
          if (Build.VERSION.SDK_INT >= 30) HapticFeedbackConstants.CONFIRM
          else HapticFeedbackConstants.VIRTUAL_KEY
        )
      }
    } else {
      drawEdgeArrow(canvas, t, sx, sy, w, h, color)
    }
    postInvalidateOnAnimation()
  }

  private fun drawEdgeArrow(canvas: Canvas, t: Target, sx: Float, sy: Float, w: Float, h: Float, color: Int) {
    val left = sideInset + dp(24f)
    val right = w - sideInset - dp(24f)
    val top = topInset + dp(24f)
    val bottom = h - bottomInset - dp(24f)
    val cx = w / 2f
    val cy = (top + bottom) / 2f

    // Direction from the screen centre toward the target. Behind the camera, the projection is
    // mirrored, so fall back to a pure left/right "turn around" cue from the camera-local x.
    var dx: Float
    var dy: Float
    if (t.inFront) {
      dx = sx - cx
      dy = sy - cy
    } else {
      dx = if (t.localX >= 0f) 1f else -1f
      dy = 0f
    }
    val len = hypot(dx, dy)
    if (len < 1e-3f) return
    dx /= len
    dy /= len

    val tx = if (dx > 0) (right - cx) / dx else if (dx < 0) (left - cx) / dx else Float.MAX_VALUE
    val ty = if (dy > 0) (bottom - cy) / dy else if (dy < 0) (top - cy) / dy else Float.MAX_VALUE
    val s = min(tx, ty)
    val px = cx + dx * s
    val py = cy + dy * s

    val size = dp(26f)
    canvas.save()
    canvas.translate(px, py)
    canvas.rotate(Math.toDegrees(atan2(dy, dx).toDouble()).toFloat())
    arrowPath.reset()
    arrowPath.moveTo(size, 0f)
    arrowPath.lineTo(-size * 0.7f, size * 0.75f)
    arrowPath.lineTo(-size * 0.35f, 0f)
    arrowPath.lineTo(-size * 0.7f, -size * 0.75f)
    arrowPath.close()
    fillPaint.color = Color.argb(110, 0, 0, 0)
    canvas.drawCircle(0f, 0f, size * 1.25f, fillPaint)
    fillPaint.color = color
    canvas.drawPath(arrowPath, fillPaint)
    canvas.restore()

    val dist = t.label ?: "%.1f m".format(t.distance)
    val label = if (!t.inFront) "behind you · $dist" else dist
    // Keep the label inside the screen, on the inner side of the arrow.
    val lx = (px - dx * dp(56f)).coerceIn(dp(70f), w - dp(70f))
    val ly = (py - dy * dp(56f) + dp(6f)).coerceIn(top, bottom)
    canvas.drawText(label, lx, ly, textPaint)
    if (abs(dx) > 0.2f) {
      canvas.drawText(if (dx > 0) "turn right" else "turn left", lx, ly + dp(20f), textPaint)
    }
  }

  /** Two stacked upward chevrons ("walk this way"), gently bobbing. */
  private fun drawChevron(canvas: Canvas, cx: Float, cy: Float, now: Long, color: Int) {
    val size = dp(22f)
    val bob = dp(4f) * sin(now / 220.0).toFloat()
    fillPaint.color = Color.argb(110, 0, 0, 0)
    canvas.drawCircle(cx, cy, size * 1.7f, fillPaint)
    fillPaint.color = color
    for (k in 0..1) {
      val y = cy + bob + (k - 0.5f) * size * 0.9f
      arrowPath.reset()
      arrowPath.moveTo(cx, y - size * 0.55f)
      arrowPath.lineTo(cx + size, y + size * 0.45f)
      arrowPath.lineTo(cx + size * 0.6f, y + size * 0.45f)
      arrowPath.lineTo(cx, y - size * 0.1f)
      arrowPath.lineTo(cx - size * 0.6f, y + size * 0.45f)
      arrowPath.lineTo(cx - size, y + size * 0.45f)
      arrowPath.close()
      canvas.drawPath(arrowPath, fillPaint)
    }
  }

  private fun drawTapFlash(canvas: Canvas, now: Long) {
    val p = tapPoint ?: return
    val age = now - tapTime
    if (age > 500) {
      tapPoint = null
      return
    }
    val f = age / 500f
    ringPaint.color = Color.argb(((1f - f) * 255).toInt(), 255, 255, 255)
    canvas.drawCircle(p.x, p.y, dp(16f) + dp(28f) * f, ringPaint)
    postInvalidateOnAnimation()
  }
}
