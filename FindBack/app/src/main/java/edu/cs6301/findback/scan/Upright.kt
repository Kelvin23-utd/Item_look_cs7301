package edu.cs6301.findback.scan

import kotlin.math.max
import kotlin.math.min

/**
 * Between sensor image pixels (what ARCore, the boxes and the depth mapping use) and the upright
 * image the detector sees: the sensor image turned [rotation] degrees clockwise, optionally scaled.
 * The same formulas as `CameraFrame.uprightToSensor` and `sensorToUpright`, without Android types,
 * so they are unit-tested. Pure Kotlin.
 */
object Upright {
  /** Sensor pixel of upright pixel ([x], [y]) in an upright image of the full sensor size. */
  fun toSensor(x: Double, y: Double, rotation: Int, sensorW: Int, sensorH: Int): DoubleArray {
    val uprightW = if (rotation % 180 == 0) sensorW else sensorH
    val uprightH = if (rotation % 180 == 0) sensorH else sensorW
    return when (rotation) {
      0 -> doubleArrayOf(x, y)
      90 -> doubleArrayOf(y, uprightW - x)
      180 -> doubleArrayOf(uprightW - x, uprightH - y)
      270 -> doubleArrayOf(uprightH - y, x)
      else -> error("Unsupported rotation $rotation")
    }
  }

  /** Upright pixel of sensor pixel ([x], [y]) (inverse of [toSensor]). */
  fun fromSensor(x: Double, y: Double, rotation: Int, sensorW: Int, sensorH: Int): DoubleArray =
    when (rotation) {
      0 -> doubleArrayOf(x, y)
      90 -> doubleArrayOf(sensorH - y, x)
      180 -> doubleArrayOf(sensorW - x, sensorH - y)
      270 -> doubleArrayOf(y, sensorW - x)
      else -> error("Unsupported rotation $rotation")
    }

  /**
   * A box found in an upright image that was scaled by [scale] (0.5: half size) back in sensor
   * pixels: both corners mapped, then sorted.
   */
  fun boxToSensor(
    left: Double, top: Double, right: Double, bottom: Double,
    scale: Double, rotation: Int, sensorW: Int, sensorH: Int,
  ): Box {
    val a = toSensor(left / scale, top / scale, rotation, sensorW, sensorH)
    val b = toSensor(right / scale, bottom / scale, rotation, sensorW, sensorH)
    return Box(
      min(a[0], b[0]).toFloat(),
      min(a[1], b[1]).toFloat(),
      max(a[0], b[0]).toFloat(),
      max(a[1], b[1]).toFloat(),
    )
  }

  /** Bounding rectangle of [box] through [m] (sensor pixels to view pixels, say): l, t, r, b. */
  fun mapBox(box: Box, m: Affine2): FloatArray {
    val xs = DoubleArray(4)
    val ys = DoubleArray(4)
    var k = 0
    for (x in doubleArrayOf(box.left.toDouble(), box.right.toDouble())) {
      for (y in doubleArrayOf(box.top.toDouble(), box.bottom.toDouble())) {
        xs[k] = m.mapX(x, y)
        ys[k] = m.mapY(x, y)
        k++
      }
    }
    return floatArrayOf(xs.min().toFloat(), ys.min().toFloat(), xs.max().toFloat(), ys.max().toFloat())
  }
}
