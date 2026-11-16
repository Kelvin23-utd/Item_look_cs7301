package edu.cs6301.findback.map

import org.junit.Assert.assertArrayEquals
import org.junit.Test

class GhostGeometryTest {
  /** Same formulas as CameraFrame.sensorToUpright (pixels). */
  private fun sensorToUpright(x: Float, y: Float, w: Int, h: Int, rotation: Int): FloatArray =
    when (rotation) {
      0 -> floatArrayOf(x, y)
      90 -> floatArrayOf(h - y, x)
      180 -> floatArrayOf(w - x, h - y)
      270 -> floatArrayOf(y, w - x)
      else -> error("rotation")
    }

  @Test
  fun normalizedMappingMatchesCameraFrame() {
    val w = 1920
    val h = 1080
    for (rot in listOf(0, 90, 180, 270)) {
      val uw = if (rot % 180 == 0) w else h
      val uh = if (rot % 180 == 0) h else w
      for ((x, y) in listOf(0f to 0f, 480f to 270f, 1920f to 1080f, 1000f to 20f)) {
        val px = sensorToUpright(x, y, w, h, rot)
        val n = GhostGeometry.imageToUpright(x / w, y / h, rot)
        assertArrayEquals("rot $rot", floatArrayOf(px[0] / uw, px[1] / uh), n, 1e-5f)
      }
    }
  }

  @Test
  fun ghostPointsScaleToTheBitmap() {
    // Portrait phone: view corners land on a central band of the landscape sensor image.
    val corners = floatArrayOf(0.1f, 1f, 0.1f, 0f, 0.9f, 1f)
    val pts = GhostGeometry.ghostPointsForViewCorners(corners, 90, 540, 960)
    assertArrayEquals(floatArrayOf(0f, 96f, 540f, 96f, 0f, 864f), pts, 1e-3f)
  }
}
