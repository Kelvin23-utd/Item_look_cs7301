package edu.cs6301.findback.ml

import android.graphics.Bitmap
import android.graphics.Matrix
import android.graphics.PointF
import android.graphics.Rect
import android.graphics.RectF
import android.media.Image
import kotlin.math.max
import kotlin.math.min

/**
 * One CPU camera frame from ARCore, converted to RGB.
 *
 * Geometry (hit tests, projections, crop rectangles) is expressed in *sensor* image pixels, which is
 * what ARCore's Coordinates2d.IMAGE_PIXELS uses. ML models receive *upright* bitmaps, rotated by
 * [rotation] degrees clockwise so the scene looks the way the user sees it on screen.
 */
class CameraFrame(val sensor: Bitmap, val rotation: Int) {
  val width: Int get() = sensor.width
  val height: Int get() = sensor.height

  val upright: Bitmap by lazy { rotate(sensor, rotation) }

  private val uprightWidth: Int get() = if (rotation % 180 == 0) width else height
  private val uprightHeight: Int get() = if (rotation % 180 == 0) height else width

  /** Map a point in upright-bitmap pixels back to sensor pixels. */
  fun uprightToSensor(x: Float, y: Float): PointF =
    when (rotation) {
      0 -> PointF(x, y)
      90 -> PointF(y, uprightWidth - x)
      180 -> PointF(uprightWidth - x, uprightHeight - y)
      270 -> PointF(uprightHeight - y, x)
      else -> error("Unsupported rotation $rotation")
    }

  /** Map a point in sensor pixels to upright-bitmap pixels (inverse of [uprightToSensor]). */
  fun sensorToUpright(x: Float, y: Float): PointF =
    when (rotation) {
      0 -> PointF(x, y)
      90 -> PointF(height - y, x)
      180 -> PointF(width - x, height - y)
      270 -> PointF(y, width - x)
      else -> error("Unsupported rotation $rotation")
    }

  fun uprightRectToSensor(r: Rect): RectF {
    val a = uprightToSensor(r.left.toFloat(), r.top.toFloat())
    val b = uprightToSensor(r.right.toFloat(), r.bottom.toFloat())
    return RectF(min(a.x, b.x), min(a.y, b.y), max(a.x, b.x), max(a.y, b.y))
  }

  /** Clamp a sensor-space rectangle to the image bounds. */
  fun clamp(r: RectF): RectF =
    RectF(
      r.left.coerceIn(0f, width - 1f),
      r.top.coerceIn(0f, height - 1f),
      r.right.coerceIn(1f, width.toFloat()),
      r.bottom.coerceIn(1f, height.toFloat()),
    )

  /** Crop a sensor-space rectangle and return it rotated upright (what the embedder sees). */
  fun cropUpright(r: RectF): Bitmap {
    val c = clamp(r)
    val x = c.left.toInt()
    val y = c.top.toInt()
    val w = max(1, c.width().toInt().coerceAtMost(width - x))
    val h = max(1, c.height().toInt().coerceAtMost(height - y))
    val crop = Bitmap.createBitmap(sensor, x, y, w, h)
    return rotate(crop, rotation)
  }

  companion object {
    fun from(image: Image, rotation: Int) = CameraFrame(YuvConverter.toBitmap(image), rotation)

    fun rotate(bitmap: Bitmap, rotation: Int): Bitmap {
      if (rotation == 0) return bitmap
      val m = Matrix().apply { postRotate(rotation.toFloat()) }
      return Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, m, true)
    }

    /** Square rectangle of side [side] centred at ([cx], [cy]). */
    fun square(cx: Float, cy: Float, side: Float) =
      RectF(cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2)

    /** Expand [r] by [fraction] of its size on every side. */
    fun pad(r: RectF, fraction: Float): RectF {
      val dx = r.width() * fraction
      val dy = r.height() * fraction
      return RectF(r.left - dx, r.top - dy, r.right + dx, r.bottom + dy)
    }
  }
}

/**
 * The three planes of a YUV_420_888 image copied out of the [Image], so the image can be closed at
 * once (v0.5 scan keyframes copy on the GL thread and convert later on a worker).
 */
class YuvPlanes(
  val width: Int,
  val height: Int,
  val y: ByteArray,
  val u: ByteArray,
  val v: ByteArray,
  val yRowStride: Int,
  val yPixelStride: Int,
  /** Row and pixel stride of the U plane, used for V too (as before v0.5). */
  val uvRowStride: Int,
  val uvPixelStride: Int,
) {
  companion object {
    fun copy(image: Image): YuvPlanes {
      val yPlane = image.planes[0]
      val uPlane = image.planes[1]
      val vPlane = image.planes[2]
      return YuvPlanes(
        image.width,
        image.height,
        yPlane.buffer.duplicate().let { b -> ByteArray(b.remaining()).also { b.get(it) } },
        uPlane.buffer.duplicate().let { b -> ByteArray(b.remaining()).also { b.get(it) } },
        vPlane.buffer.duplicate().let { b -> ByteArray(b.remaining()).also { b.get(it) } },
        yPlane.rowStride,
        yPlane.pixelStride,
        uPlane.rowStride,
        uPlane.pixelStride,
      )
    }
  }
}

/** Plain-Kotlin YUV_420_888 to ARGB conversion (replaces the deprecated RenderScript path). */
object YuvConverter {
  fun toBitmap(image: Image): Bitmap = toBitmap(YuvPlanes.copy(image))

  /** The conversion itself, on planes already copied out of the image. */
  fun toBitmap(p: YuvPlanes): Bitmap {
    val w = p.width
    val h = p.height
    val yBytes = p.y
    val uBytes = p.u
    val vBytes = p.v
    val yRow = p.yRowStride
    val yPix = p.yPixelStride
    val uvRow = p.uvRowStride
    val uvPix = p.uvPixelStride

    val argb = IntArray(w * h)
    for (j in 0 until h) {
      val yOff = j * yRow
      val uvOff = (j shr 1) * uvRow
      val outOff = j * w
      for (i in 0 until w) {
        val y = yBytes[yOff + i * yPix].toInt() and 0xff
        val uvIdx = uvOff + (i shr 1) * uvPix
        val u = (uBytes[uvIdx].toInt() and 0xff) - 128
        val v = (vBytes[uvIdx].toInt() and 0xff) - 128
        // BT.601 full-range, 16.16 fixed point.
        val r = (y + ((91881 * v) shr 16)).coerceIn(0, 255)
        val g = (y - ((22554 * u + 46802 * v) shr 16)).coerceIn(0, 255)
        val b = (y + ((116130 * u) shr 16)).coerceIn(0, 255)
        argb[outOff + i] = (0xff shl 24) or (r shl 16) or (g shl 8) or b
      }
    }
    return Bitmap.createBitmap(argb, w, h, Bitmap.Config.ARGB_8888)
  }
}
