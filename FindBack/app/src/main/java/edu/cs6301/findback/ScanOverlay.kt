package edu.cs6301.findback

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.os.SystemClock
import android.util.AttributeSet
import android.view.View
import edu.cs6301.findback.scan.OverlayBox
import edu.cs6301.findback.scan.ScanConfig

/**
 * v0.5 scan mode: the boxes and labels of the latest processed keyframe, in view pixels. Entries
 * that already existed are green, new ones amber. The boxes are where the things were when the
 * keyframe was taken, so they lag the live view by the processing time; they disappear after
 * [ScanConfig.OVERLAY_HOLD_MS] without a newer keyframe. Never consumes touches.
 */
class ScanOverlay @JvmOverloads constructor(context: Context, attrs: AttributeSet? = null) : View(context, attrs) {
  private var boxes: List<OverlayBox> = emptyList()
  private var shownAtMs = 0L
  private val density = resources.displayMetrics.density

  private val boxPaint =
    Paint(Paint.ANTI_ALIAS_FLAG).apply {
      style = Paint.Style.STROKE
      strokeWidth = 3f * density
    }
  private val labelBg = Paint(Paint.ANTI_ALIAS_FLAG).apply { style = Paint.Style.FILL }
  private val textPaint =
    Paint(Paint.ANTI_ALIAS_FLAG).apply {
      color = Color.BLACK
      textSize = 13f * density
    }

  /** UI thread. An empty list clears the overlay. */
  fun setBoxes(b: List<OverlayBox>) {
    boxes = b
    shownAtMs = SystemClock.elapsedRealtime()
    invalidate()
    if (b.isNotEmpty()) postDelayed({ invalidate() }, ScanConfig.OVERLAY_HOLD_MS + 50)
  }

  override fun onDraw(canvas: Canvas) {
    super.onDraw(canvas)
    if (boxes.isEmpty() || SystemClock.elapsedRealtime() - shownAtMs > ScanConfig.OVERLAY_HOLD_MS) return
    val pad = 4f * density
    for (b in boxes) {
      val color = if (b.known) MainView.GREEN else MainView.AMBER
      boxPaint.color = color
      labelBg.color = color
      canvas.drawRect(b.left, b.top, b.right, b.bottom, boxPaint)
      val text = if (b.known) b.label else b.label + " · new"
      val w = textPaint.measureText(text) + 2 * pad
      val h = textPaint.textSize + 2 * pad
      val top = if (b.top - h >= 0f) b.top - h else b.top
      canvas.drawRect(b.left, top, b.left + w, top + h, labelBg)
      canvas.drawText(text, b.left + pad, top + h - pad - textPaint.descent() / 2, textPaint)
    }
  }
}
