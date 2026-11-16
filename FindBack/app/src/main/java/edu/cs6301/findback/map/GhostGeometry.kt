package edu.cs6301.findback.map

/**
 * Placement of a spot's ghost image over the live view.
 *
 * The ghost is the CPU camera image of the saved pose, stored upright (rotated like
 * `CameraFrame.upright`). To overlay it, the renderer maps three view corners into CPU-image
 * normalized coordinates with `Frame.transformCoordinates2d(VIEW, ..., IMAGE_NORMALIZED, ...)`, the
 * same mapping that places the camera background and that turns taps into crop centres. Those
 * corners are converted here to ghost-bitmap pixels; the UI then fits an affine matrix
 * (`Matrix.setPolyToPoly`, 3 points) from ghost pixels to view pixels.
 */
object GhostGeometry {
  /**
   * CPU-image normalized (u, v) to upright-image normalized, for [rotation] degrees clockwise.
   * Matches `CameraFrame.sensorToUpright` divided by the upright size.
   */
  fun imageToUpright(u: Float, v: Float, rotation: Int): FloatArray =
    when (((rotation % 360) + 360) % 360) {
      0 -> floatArrayOf(u, v)
      90 -> floatArrayOf(1f - v, u)
      180 -> floatArrayOf(1f - u, 1f - v)
      270 -> floatArrayOf(v, 1f - u)
      else -> error("Unsupported rotation $rotation")
    }

  /**
   * Ghost-bitmap pixel positions of the view corners (0,0), (w,0), (0,h), given those corners in
   * CPU-image normalized coordinates ([cornersImageNorm]: u0, v0, u1, v1, u2, v2).
   */
  fun ghostPointsForViewCorners(cornersImageNorm: FloatArray, rotation: Int, ghostW: Int, ghostH: Int): FloatArray {
    val out = FloatArray(6)
    for (i in 0 until 3) {
      val up = imageToUpright(cornersImageNorm[2 * i], cornersImageNorm[2 * i + 1], rotation)
      out[2 * i] = up[0] * ghostW
      out[2 * i + 1] = up[1] * ghostH
    }
    return out
  }

  fun viewCorners(viewW: Int, viewH: Int) = floatArrayOf(0f, 0f, viewW.toFloat(), 0f, 0f, viewH.toFloat())
}
