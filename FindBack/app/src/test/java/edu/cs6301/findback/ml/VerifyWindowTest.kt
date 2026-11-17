package edu.cs6301.findback.ml

import org.junit.Assert.assertEquals
import org.junit.Test

/** The helper extracted from handleVerify must compute what the original lines computed. */
class VerifyWindowTest {
  /** The expressions as they stood in FindBackRenderer.handleVerify before v0.5 (left, top, right, bottom, cw, ch). */
  private fun original(
    captureDistance: Float, cropWidthPx: Float, cropHeightPx: Float, imageWidth: Int,
    d1: Float, cfWidth: Int, cfHeight: Int, px0: Float, px1: Float,
  ): FloatArray {
    val scale = (captureDistance / d1) * (cfWidth.toFloat() / imageWidth)
    val cw = (cropWidthPx * scale).coerceIn(64f, cfWidth.toFloat())
    val ch = (cropHeightPx * scale).coerceIn(64f, cfHeight.toFloat())
    return floatArrayOf(px0 - cw / 2, px1 - ch / 2, px0 + cw / 2, px1 + ch / 2, cw, ch)
  }

  @Test
  fun sameNumbersAsTheOriginalExpressions() {
    val cases =
      listOf(
        floatArrayOf(0.92f, 310.5f, 287.25f, 1920f, 0.61f, 1920f, 1080f, 812.3f, 402.7f),
        floatArrayOf(2.6753116f, 160f, 160f, 1920f, 0.1f, 1920f, 1080f, 5f, 1075f), // very near: clamped to the frame
        floatArrayOf(0.5f, 120f, 90f, 1920f, 4.9f, 1920f, 1080f, 960f, 540f), // far: clamped to 64 px
        floatArrayOf(1.3f, 333.3f, 211.1f, 640f, 1.7f, 1920f, 1080f, 1500.5f, 77.25f), // another image size
      )
    for (c in cases) {
      val o = original(c[0], c[1], c[2], c[3].toInt(), c[4], c[5].toInt(), c[6].toInt(), c[7], c[8])
      val w = VerifyWindow.expected(c[0], c[1], c[2], c[3].toInt(), c[4], c[5].toInt(), c[6].toInt(), c[7], c[8])
      assertEquals(o[0], w.left, 0f)
      assertEquals(o[1], w.top, 0f)
      assertEquals(o[2], w.right, 0f)
      assertEquals(o[3], w.bottom, 0f)
      assertEquals(o[4], w.width, 0f)
      assertEquals(o[5], w.height, 0f)
    }
    val far = VerifyWindow.expected(0.5f, 120f, 90f, 1920, 4.9f, 1920, 1080, 960f, 540f)
    assertEquals(64f, far.width, 0f)
    assertEquals(64f, far.height, 0f)
    val near = VerifyWindow.expected(2.6753116f, 160f, 160f, 1920, 0.1f, 1920, 1080, 5f, 1075f)
    assertEquals(1920f, near.width, 0f)
    assertEquals(1080f, near.height, 0f)
  }
}
