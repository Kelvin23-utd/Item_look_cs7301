package edu.cs6301.findback.map

import kotlin.math.PI
import kotlin.math.atan2
import kotlin.math.cos
import kotlin.math.hypot
import kotlin.math.sin
import kotlin.math.sqrt

/** Plain 3D point or direction in metres. Doubles keep multi-step round trips well under 1 mm. */
data class Vec3(val x: Double, val y: Double, val z: Double) {
  operator fun plus(o: Vec3) = Vec3(x + o.x, y + o.y, z + o.z)

  operator fun minus(o: Vec3) = Vec3(x - o.x, y - o.y, z - o.z)

  operator fun times(s: Double) = Vec3(x * s, y * s, z * s)

  fun length() = sqrt(x * x + y * y + z * z)

  /** Length of the projection onto the horizontal (x-z) plane. +Y is up in both W and M. */
  fun horizontalLength() = hypot(x, z)

  fun toFloatArray() = floatArrayOf(x.toFloat(), y.toFloat(), z.toFloat())

  companion object {
    val ZERO = Vec3(0.0, 0.0, 0.0)

    fun of(a: FloatArray) = Vec3(a[0].toDouble(), a[1].toDouble(), a[2].toDouble())
  }
}

fun distance(a: Vec3, b: Vec3) = (a - b).length()

/** Distance in the floor plane; the flat is treated as one floor. */
fun horizontalDistance(a: Vec3, b: Vec3) = hypot(a.x - b.x, a.z - b.z)

/**
 * Gravity-aligned rigid transform with 4 degrees of freedom: a rotation by [yaw] radians about +Y,
 * followed by a translation ([x], [y], [z]). It maps points from a local frame into its parent.
 *
 * The rotation is right-handed about +Y, the same sense as an ARCore quaternion (0, sin(yaw/2), 0,
 * cos(yaw/2)): yaw = +90 degrees turns local +X into parent -Z.
 *
 * World W and map M share +Y (both gravity-aligned), so they differ by exactly one Pose4. A full
 * 6-DoF transform is never used between them: tilt error in an image pose would be multiplied by the
 * distance to far rooms.
 */
data class Pose4(val x: Double, val y: Double, val z: Double, val yaw: Double) {
  val translation: Vec3
    get() = Vec3(x, y, z)

  fun rotate(v: Vec3): Vec3 {
    val c = cos(yaw)
    val s = sin(yaw)
    return Vec3(c * v.x + s * v.z, v.y, -s * v.x + c * v.z)
  }

  fun transformPoint(p: Vec3): Vec3 = rotate(p) + translation

  fun transformPoint(p: FloatArray): FloatArray = transformPoint(Vec3.of(p)).toFloatArray()

  fun inverse(): Pose4 {
    val t = Pose4(0.0, 0.0, 0.0, -yaw).rotate(translation)
    return Pose4(-t.x, -t.y, -t.z, wrapAngle(-yaw))
  }

  /** `this ∘ other`: apply [other] first, then this. */
  fun compose(other: Pose4): Pose4 {
    val t = rotate(other.translation) + translation
    return Pose4(t.x, t.y, t.z, wrapAngle(yaw + other.yaw))
  }

  /** Local -Z (the "forward" direction of a camera-like frame) in the parent frame, horizontal. */
  fun forward(): Vec3 = rotate(Vec3(0.0, 0.0, -1.0))

  /** Yaw rotation as an ARCore-ordered quaternion (x, y, z, w). */
  fun quaternion(): FloatArray = floatArrayOf(0f, sin(yaw / 2).toFloat(), 0f, cos(yaw / 2).toFloat())

  companion object {
    val IDENTITY = Pose4(0.0, 0.0, 0.0, 0.0)

    /**
     * Yaw that turns local +X into the horizontal projection of [axis]. The caller checks
     * [Vec3.horizontalLength] first: a near-vertical axis has no stable heading.
     */
    fun yawOfAxis(axis: Vec3): Double = atan2(-axis.z, axis.x)

    /** Pose4 at [position] whose local +X heading follows the reference [axis]. */
    fun fromAxis(position: Vec3, axis: Vec3): Pose4 = Pose4(position.x, position.y, position.z, yawOfAxis(axis))

    fun fromAxis(position: FloatArray, axis: FloatArray): Pose4 = fromAxis(Vec3.of(position), Vec3.of(axis))

    /** Wrap to (-π, π]. */
    fun wrapAngle(a: Double): Double {
      var r = a % (2 * PI)
      if (r <= -PI) r += 2 * PI
      if (r > PI) r -= 2 * PI
      return r
    }

    /** Signed smallest difference a - b, in (-π, π]. */
    fun angleDiff(a: Double, b: Double) = wrapAngle(a - b)
  }
}
