package edu.cs6301.findback.ml

/**
 * The expected crop window of verification: the stored crop rescaled by distance and image
 * resolution, i.e. the same physical window around the remembered point, centred on its pixel.
 *
 * Extracted unchanged from `FindBackRenderer.handleVerify` (v0.2 crop logic) so that scan evidence
 * (v0.5) looks at exactly the window verification would. Pure Kotlin, unit-tested against the
 * original expressions. The caller clamps the rectangle to the image (`CameraFrame.clamp`).
 */
object VerifyWindow {
  /** [width] × [height] is the window before clamping; the rectangle is in sensor image pixels. */
  class Window(
    val width: Float,
    val height: Float,
    val left: Float,
    val top: Float,
    val right: Float,
    val bottom: Float,
  )

  /**
   * [captureDistance], [cropWidthPx], [cropHeightPx], [recordImageWidth]: as stored at capture.
   * [distance]: camera to the remembered point now. ([px], [py]): its pixel in the current frame.
   */
  fun expected(
    captureDistance: Float,
    cropWidthPx: Float,
    cropHeightPx: Float,
    recordImageWidth: Int,
    distance: Float,
    frameWidth: Int,
    frameHeight: Int,
    px: Float,
    py: Float,
  ): Window {
    val scale = (captureDistance / distance) * (frameWidth.toFloat() / recordImageWidth)
    val cw = (cropWidthPx * scale).coerceIn(64f, frameWidth.toFloat())
    val ch = (cropHeightPx * scale).coerceIn(64f, frameHeight.toFloat())
    return Window(cw, ch, px - cw / 2, py - ch / 2, px + cw / 2, py + ch / 2)
  }
}
