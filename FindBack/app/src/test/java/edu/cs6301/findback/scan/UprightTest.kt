package edu.cs6301.findback.scan

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Test

/** The detector's upright, half-size input and the overlay mapping, back to sensor pixels. */
class UprightTest {
  @Test
  fun toSensorInvertsFromSensorForEveryRotation() {
    for (rot in listOf(0, 90, 180, 270)) {
      for ((x, y) in listOf(0.0 to 0.0, 1919.0 to 1079.0, 812.5 to 402.25)) {
        val up = Upright.fromSensor(x, y, rot, 1920, 1080)
        val back = Upright.toSensor(up[0], up[1], rot, 1920, 1080)
        assertEquals("rot $rot x", x, back[0], 1e-9)
        assertEquals("rot $rot y", y, back[1], 1e-9)
      }
    }
    // Portrait phone (90° clockwise): the sensor's top-left corner is the upright top-right one.
    assertArrayEquals(doubleArrayOf(1080.0, 0.0), Upright.fromSensor(0.0, 0.0, 90, 1920, 1080), 1e-12)
  }

  @Test
  fun aBoxInTheHalfSizeUprightImageComesBackInSensorPixels() {
    // A sensor box, seen by the detector in the upright image at half size...
    val sensor = Box(800f, 300f, 1000f, 420f)
    val a = Upright.fromSensor(sensor.left.toDouble(), sensor.top.toDouble(), 90, 1920, 1080)
    val b = Upright.fromSensor(sensor.right.toDouble(), sensor.bottom.toDouble(), 90, 1920, 1080)
    val l = minOf(a[0], b[0]) * 0.5
    val t = minOf(a[1], b[1]) * 0.5
    val r = maxOf(a[0], b[0]) * 0.5
    val bt = maxOf(a[1], b[1]) * 0.5
    // ...is the same box once mapped back.
    assertEquals(sensor, Upright.boxToSensor(l, t, r, bt, 0.5, 90, 1920, 1080))
    assertEquals(sensor, Upright.boxToSensor(sensor.left * 0.5, sensor.top * 0.5, sensor.right * 0.5, sensor.bottom * 0.5, 0.5, 0, 1920, 1080))
  }

  @Test
  fun mapBoxIsTheBoundingRectangleThroughAnAffineMap() {
    // A quarter turn (sensor to portrait view) and a scale.
    val m = Affine2(0.0, -2.0, 2160.0, 2.0, 0.0, 0.0)
    val v = Upright.mapBox(Box(100f, 200f, 300f, 260f), m)
    assertArrayEquals(floatArrayOf(1640f, 200f, 1760f, 600f), v, 1e-3f)
  }

  @Test
  fun depthBytesAreLittleEndianRowMajor() {
    val d = DepthMap(2, 1, shortArrayOf(0x1234, 40000.toShort()), Affine2.IDENTITY)
    val b = d.littleEndianBytes()
    assertArrayEquals(byteArrayOf(0x34, 0x12, (40000 and 0xff).toByte(), (40000 shr 8).toByte()), b)
  }
}
