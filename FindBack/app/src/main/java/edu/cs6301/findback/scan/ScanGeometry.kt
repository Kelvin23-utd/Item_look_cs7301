package edu.cs6301.findback.scan

import edu.cs6301.findback.map.Pose4
import edu.cs6301.findback.map.Vec3
import kotlin.math.abs
import kotlin.math.acos
import kotlin.math.ceil
import kotlin.math.floor
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

/*
 * Scan geometry: from a pixel and a depth to a point in the home map, and back. Pure Kotlin (no
 * Android or ARCore imports), unit-tested.
 *
 * Conventions, each taken from the ARCore reference (see the README, v0.5):
 *  - Image pixels (u, v) are CPU image pixels (`Coordinates2d.IMAGE_PIXELS`): u to the right, v
 *    down, in the sensor's readout order. The intrinsics are `Camera.getImageIntrinsics()`
 *    ("unrotated camera intrinsics for the CPU image").
 *  - The camera frame is that of `Camera.getPose()`: "+X pointing right, +Y pointing right up, and
 *    -Z pointing in the direction the camera is looking, with right and up being relative to the
 *    image readout in the usual left-to-right, top-to-bottom order". So u grows along +X and v grows
 *    along -Y.
 *  - A depth sample is "the distance in millimeters to the camera plane": distance along the optical
 *    axis, not along the ray.
 *  - Depth pixel = TEXTURE_NORMALIZED coordinate × depth image size; the CPU image maps to it
 *    through `Frame.transformCoordinates2d(IMAGE_PIXELS, .., TEXTURE_NORMALIZED, ..)`, captured here
 *    as an affine map because the frame is gone when a keyframe is processed.
 */

/** Rectangle in CPU (sensor) image pixels. */
data class Box(val left: Float, val top: Float, val right: Float, val bottom: Float) {
  val width: Float
    get() = right - left

  val height: Float
    get() = bottom - top

  val area: Float
    get() = max(0f, width) * max(0f, height)

  val centerX: Float
    get() = (left + right) / 2f

  val centerY: Float
    get() = (top + bottom) / 2f

  fun intersectionArea(o: Box): Float {
    val w = min(right, o.right) - max(left, o.left)
    val h = min(bottom, o.bottom) - max(top, o.top)
    return if (w <= 0f || h <= 0f) 0f else w * h
  }

  /** Intersection over union, 0 for empty boxes. */
  fun iou(o: Box): Float {
    val i = intersectionArea(o)
    val u = area + o.area - i
    return if (u <= 0f) 0f else i / u
  }

  /** The same centre with [fraction] of the width and of the height. */
  fun central(fraction: Float): Box {
    val w = width * fraction / 2f
    val h = height * fraction / 2f
    return Box(centerX - w, centerY - h, centerX + w, centerY + h)
  }

  /** Grown by [fraction] of its size on every side (as `CameraFrame.pad`). */
  fun padded(fraction: Float): Box {
    val dx = width * fraction
    val dy = height * fraction
    return Box(left - dx, top - dy, right + dx, bottom + dy)
  }

  fun toList(): List<Double> = listOf(left.toDouble(), top.toDouble(), right.toDouble(), bottom.toDouble())
}

/** Pinhole intrinsics of the CPU image, in pixels. */
data class Intrinsics(val fx: Double, val fy: Double, val cx: Double, val cy: Double, val width: Int, val height: Int) {
  /** The same camera for an image of another size (ARCore reports them for its own dimensions). */
  fun scaledTo(w: Int, h: Int): Intrinsics {
    if (w == width && h == height) return this
    val sx = w.toDouble() / width
    val sy = h.toDouble() / height
    return Intrinsics(fx * sx, fy * sy, cx * sx, cy * sy, w, h)
  }
}

/** A point of the image with its distance along the optical axis (metres). */
data class Projection(val u: Double, val v: Double, val depthM: Double)

object Pinhole {
  /**
   * Camera-frame point of pixel ([u], [v]) at [depthM] along the optical axis:
   * (d·(u − cx)/fx, d·(cy − v)/fy, −d).
   */
  fun unproject(u: Double, v: Double, depthM: Double, k: Intrinsics): Vec3 =
    Vec3(depthM * (u - k.cx) / k.fx, depthM * (k.cy - v) / k.fy, -depthM)

  /** Inverse of [unproject]; null for a point at or behind the camera plane. */
  fun project(p: Vec3, k: Intrinsics): Projection? {
    val d = -p.z
    if (d <= 1e-6) return null
    return Projection(k.cx + k.fx * p.x / d, k.cy - k.fy * p.y / d, d)
  }
}

/**
 * Rigid pose of the physical camera in the ARCore world frame: a rotation (unit quaternion in
 * ARCore's order x, y, z, w) followed by a translation, as `Pose.transformPoint`.
 */
class CameraPose(val position: Vec3, val qx: Double, val qy: Double, val qz: Double, val qw: Double) {
  /** Camera frame to world frame. */
  fun toWorld(p: Vec3): Vec3 = rotate(p, qx, qy, qz, qw) + position

  /** World frame to camera frame. */
  fun toCamera(p: Vec3): Vec3 = rotate(p - position, -qx, -qy, -qz, qw)

  /** Angle of the rotation from this orientation to [o], in degrees (0 to 180). */
  fun angleTo(o: CameraPose): Double = quaternionAngleDeg(qx, qy, qz, qw, o.qx, o.qy, o.qz, o.qw)

  fun quaternion(): List<Double> = listOf(qx, qy, qz, qw)

  companion object {
    fun of(translation: FloatArray, quaternion: FloatArray) =
      CameraPose(
        Vec3.of(translation),
        quaternion[0].toDouble(),
        quaternion[1].toDouble(),
        quaternion[2].toDouble(),
        quaternion[3].toDouble(),
      )

    /** v' = v + 2w (q × v) + 2 q × (q × v), for a unit quaternion (x, y, z, w). */
    private fun rotate(v: Vec3, x: Double, y: Double, z: Double, w: Double): Vec3 {
      val cx = y * v.z - z * v.y
      val cy = z * v.x - x * v.z
      val cz = x * v.y - y * v.x
      val dx = y * cz - z * cy
      val dy = z * cx - x * cz
      val dz = x * cy - y * cx
      return Vec3(v.x + 2 * (w * cx + dx), v.y + 2 * (w * cy + dy), v.z + 2 * (w * cz + dz))
    }

    fun quaternionAngleDeg(
      ax: Double, ay: Double, az: Double, aw: Double,
      bx: Double, by: Double, bz: Double, bw: Double,
    ): Double {
      val na = sqrt(ax * ax + ay * ay + az * az + aw * aw)
      val nb = sqrt(bx * bx + by * by + bz * bz + bw * bw)
      if (na < 1e-9 || nb < 1e-9) return 0.0
      val dot = abs(ax * bx + ay * by + az * bz + aw * bw) / (na * nb)
      return Math.toDegrees(2.0 * acos(dot.coerceIn(0.0, 1.0)))
    }
  }
}

/** 2D affine map: x' = a·x + b·y + c, y' = d·x + e·y + f. */
data class Affine2(val a: Double, val b: Double, val c: Double, val d: Double, val e: Double, val f: Double) {
  fun mapX(x: Double, y: Double): Double = a * x + b * y + c

  fun mapY(x: Double, y: Double): Double = d * x + e * y + f

  fun toList(): List<Double> = listOf(a, b, c, d, e, f)

  companion object {
    val IDENTITY = Affine2(1.0, 0.0, 0.0, 0.0, 1.0, 0.0)

    /**
     * The affine map that takes three source points to three destination points ([src] and [dst]
     * are x0, y0, x1, y1, x2, y2), or null when the source points lie on a line.
     */
    fun fromThreePoints(src: DoubleArray, dst: DoubleArray): Affine2? {
      val x0 = src[0]
      val y0 = src[1]
      val x1 = src[2]
      val y1 = src[3]
      val x2 = src[4]
      val y2 = src[5]
      val det = x0 * (y1 - y2) - y0 * (x1 - x2) + (x1 * y2 - x2 * y1)
      if (abs(det) < 1e-9) return null
      fun solve(v0: Double, v1: Double, v2: Double): DoubleArray =
        doubleArrayOf(
          (v0 * (y1 - y2) - y0 * (v1 - v2) + (v1 * y2 - v2 * y1)) / det,
          (x0 * (v1 - v2) - v0 * (x1 - x2) + (x1 * v2 - x2 * v1)) / det,
          (x0 * (y1 * v2 - y2 * v1) - y0 * (x1 * v2 - x2 * v1) + v0 * (x1 * y2 - x2 * y1)) / det,
        )
      val px = solve(dst[0], dst[2], dst[4])
      val py = solve(dst[1], dst[3], dst[5])
      return Affine2(px[0], px[1], px[2], py[0], py[1], py[2])
    }
  }
}

/**
 * One 16-bit depth image, copied out of ARCore: [mm] is row-major, unsigned millimetres along the
 * optical axis, 0 for "no estimate". [fromImage] maps CPU image pixels to depth pixel coordinates,
 * where pixel (i, j) covers [i, i+1) × [j, j+1).
 */
class DepthMap(val width: Int, val height: Int, private val mm: ShortArray, val fromImage: Affine2) {
  init {
    require(mm.size >= width * height) { "Depth buffer of ${mm.size} values is too small for ${width}x$height" }
  }

  /** Millimetres at depth pixel ([x], [y]); 0 means no estimate. */
  fun mmAt(x: Int, y: Int): Int = mm[y * width + x].toInt() and 0xffff

  /** The image as raw little-endian uint16, row-major, [width] × [height] (the saved-frame file). */
  fun littleEndianBytes(): ByteArray {
    val n = width * height
    val out = ByteArray(n * 2)
    for (i in 0 until n) {
      val v = mm[i].toInt()
      out[2 * i] = (v and 0xff).toByte()
      out[2 * i + 1] = ((v shr 8) and 0xff).toByte()
    }
    return out
  }

  /** Share of depth pixels that carry an estimate. */
  fun validShare(): Double {
    val n = width * height
    if (n == 0) return 0.0
    var valid = 0
    for (i in 0 until n) if (mm[i].toInt() != 0) valid++
    return valid.toDouble() / n
  }

  /**
   * Median depth (metres) of the valid depth pixels whose centres lie in the region that [box]
   * (CPU image pixels) maps to; at least the pixel under the box centre is used. Null when the
   * region is outside the depth image or has no valid pixel.
   */
  fun medianInBox(box: Box): Double? {
    val xs = DoubleArray(4)
    val ys = DoubleArray(4)
    var k = 0
    for (cx in doubleArrayOf(box.left.toDouble(), box.right.toDouble())) {
      for (cy in doubleArrayOf(box.top.toDouble(), box.bottom.toDouble())) {
        xs[k] = fromImage.mapX(cx, cy)
        ys[k] = fromImage.mapY(cx, cy)
        k++
      }
    }
    val xRange = pixelRange(xs.min(), xs.max(), width) ?: return null
    val yRange = pixelRange(ys.min(), ys.max(), height) ?: return null
    return median(xRange.first, xRange.second, yRange.first, yRange.second)
  }

  /**
   * Median depth (metres) of the valid pixels in the [size]×[size] patch around the depth pixel
   * that CPU image pixel ([u], [v]) falls in. Null when that pixel is outside the depth image or
   * the patch has no valid pixel.
   */
  fun medianPatch(u: Double, v: Double, size: Int = ScanConfig.DEPTH_PATCH): Double? {
    val x = floor(fromImage.mapX(u, v)).toInt()
    val y = floor(fromImage.mapY(u, v)).toInt()
    if (x < 0 || y < 0 || x >= width || y >= height) return null
    val r = size / 2
    return median(max(0, x - r), min(width - 1, x + r), max(0, y - r), min(height - 1, y + r))
  }

  /** Indices of the pixels whose centres lie in [lo, hi], clamped to the image; null when outside. */
  private fun pixelRange(lo: Double, hi: Double, n: Int): Pair<Int, Int>? {
    var i0 = ceil(lo - 0.5).toInt()
    var i1 = floor(hi - 0.5).toInt()
    if (i0 > i1) {
      // Narrower than one depth pixel: the pixel under the middle.
      i0 = floor((lo + hi) / 2).toInt()
      i1 = i0
    }
    if (i1 < 0 || i0 > n - 1) return null
    return max(0, i0) to min(n - 1, i1)
  }

  private fun median(x0: Int, x1: Int, y0: Int, y1: Int): Double? {
    val values = IntArray((x1 - x0 + 1) * (y1 - y0 + 1))
    var n = 0
    for (y in y0..y1) {
      for (x in x0..x1) {
        val d = mmAt(x, y)
        if (d != 0) values[n++] = d
      }
    }
    if (n == 0) return null
    values.sort(0, n)
    return values[n / 2] / 1000.0
  }
}

/**
 * Everything needed to place one keyframe's pixels in the home map: the CPU image intrinsics, the
 * camera pose in the world frame, the map transform T_WM (map frame to world frame) and the depth.
 */
class KeyframeGeometry(
  val intrinsics: Intrinsics,
  val cameraPose: CameraPose,
  val mapToWorld: Pose4,
  val depth: DepthMap,
) {
  private val worldToMap: Pose4 = mapToWorld.inverse()

  /** Camera position in the map frame: where the user stood. */
  val cameraMap: Vec3
    get() = worldToMap.transformPoint(cameraPose.position)

  /** World point of pixel ([u], [v]) at [depthM] along the optical axis. */
  fun worldPoint(u: Double, v: Double, depthM: Double): Vec3 =
    cameraPose.toWorld(Pinhole.unproject(u, v, depthM, intrinsics))

  /** Map point of pixel ([u], [v]) at [depthM] along the optical axis. */
  fun mapPoint(u: Double, v: Double, depthM: Double): Vec3 = worldToMap.transformPoint(worldPoint(u, v, depthM))

  fun worldToMapPoint(world: Vec3): Vec3 = worldToMap.transformPoint(world)

  /** Where a map point falls in this image, or null when it is behind the camera. */
  fun projectMap(p: Vec3): Projection? =
    Pinhole.project(cameraPose.toCamera(mapToWorld.transformPoint(p)), intrinsics)

  /** Where a world point falls in this image, or null when it is behind the camera. */
  fun projectWorld(p: Vec3): Projection? = Pinhole.project(cameraPose.toCamera(p), intrinsics)

  /** Whether a projection is inside the image, less [margin] of each side. */
  fun inImage(p: Projection, margin: Double): Boolean {
    val w = intrinsics.width.toDouble()
    val h = intrinsics.height.toDouble()
    return p.u >= margin * w && p.u <= (1 - margin) * w && p.v >= margin * h && p.v <= (1 - margin) * h
  }
}
