package edu.cs6301.findback.map

import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.sin
import kotlin.math.sqrt
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class Pose4Test {
  private val eps = 1e-9

  private fun assertVec(expected: Vec3, actual: Vec3, tol: Double = eps) {
    assertEquals("x", expected.x, actual.x, tol)
    assertEquals("y", expected.y, actual.y, tol)
    assertEquals("z", expected.z, actual.z, tol)
  }

  private fun assertPose(expected: Pose4, actual: Pose4, tol: Double = eps) {
    assertVec(expected.translation, actual.translation, tol)
    assertEquals("yaw", 0.0, Pose4.angleDiff(expected.yaw, actual.yaw), tol)
  }

  private val a = Pose4(1.5, -0.3, 2.0, 0.7)
  private val b = Pose4(-4.0, 1.2, 0.25, -2.4)
  private val pts = listOf(Vec3(0.0, 0.0, 0.0), Vec3(1.0, 2.0, 3.0), Vec3(-5.5, 0.1, 7.25))

  @Test
  fun inverseRoundTrip() {
    for (p in pts) assertVec(p, a.inverse().transformPoint(a.transformPoint(p)))
    assertPose(Pose4.IDENTITY, a.compose(a.inverse()))
    assertPose(Pose4.IDENTITY, a.inverse().compose(a))
    assertPose(a, a.inverse().inverse())
  }

  @Test
  fun composeAppliesRightOperandFirst() {
    for (p in pts) assertVec(a.transformPoint(b.transformPoint(p)), a.compose(b).transformPoint(p))
    val c = Pose4(0.3, 0.0, -0.9, 3.0)
    assertPose(a.compose(b).compose(c), a.compose(b.compose(c)))
  }

  @Test
  fun rotationKeepsHeightAndMatchesArcoreQuaternion() {
    val v = Vec3(1.0, 0.5, 0.0)
    val r = Pose4(0.0, 0.0, 0.0, PI / 2).rotate(v)
    // Right-handed about +Y: +X turns into -Z.
    assertVec(Vec3(0.0, 0.5, -1.0), r)
    // Same result as rotating by the quaternion we hand to ARCore.
    for (p in pts) assertVec(a.rotate(p), quatRotate(a.quaternion(), p), 1e-6)
  }

  @Test
  fun yawFromAxis() {
    assertEquals(0.0, Pose4.yawOfAxis(Vec3(1.0, 0.0, 0.0)), eps)
    assertEquals(PI / 2, Pose4.yawOfAxis(Vec3(0.0, 0.0, -1.0)), eps)
    assertEquals(-PI / 2, Pose4.yawOfAxis(Vec3(0.0, 0.0, 1.0)), eps)
    assertEquals(0.0, Pose4.angleDiff(Pose4.yawOfAxis(Vec3(-1.0, 0.0, 0.0)), PI), eps) // ±π
    // Tilt does not change the heading: only the horizontal projection matters.
    assertEquals(0.0, Pose4.yawOfAxis(Vec3(0.8, 0.6, 0.0)), eps)
    // The resulting pose's +X points along the projected axis.
    val axis = Vec3(0.3, -0.4, -0.5)
    val p = Pose4.fromAxis(Vec3(1.0, 1.0, 1.0), axis)
    val x = p.rotate(Vec3(1.0, 0.0, 0.0))
    val h = axis.horizontalLength()
    assertVec(Vec3(axis.x / h, 0.0, axis.z / h), x)
  }

  @Test
  fun wrapAngle() {
    assertEquals(PI, Pose4.wrapAngle(PI), eps)
    assertEquals(PI, Pose4.wrapAngle(-PI), eps)
    assertEquals(-PI / 2, Pose4.wrapAngle(3 * PI / 2), eps)
    assertEquals(0.1, Pose4.angleDiff(PI - 0.05, -PI + 0.05) * -1, eps)
  }

  /**
   * Session 1 creates the map from a tilted picture, registers items and adds a spot. Session 2 has
   * a different world frame (ARCore picks a new origin and heading on every launch); locating from
   * either fix point must put every item back within 1 mm.
   */
  @Test
  fun secondSessionRecoversItemsWithin1mm() {
    // --- session 1 (world W1)
    val imageCenter1 = Vec3(0.4, 1.1, -2.0)
    val imageX1 = tiltedAxis(yaw = 0.35, tiltDeg = 12.0) // picture propped up, not quite level
    val fix1 = Pose4.fromAxis(imageCenter1, imageX1)
    val imageMapPose = Pose4.IDENTITY // first fix point defines M
    val tWM1 = fix1.compose(imageMapPose.inverse())
    val itemsW1 = listOf(Vec3(0.5, 0.8, -1.6), Vec3(3.2, 0.4, -6.1), Vec3(-7.5, 1.9, 4.4), Vec3(12.0, 0.0, 9.0))
    val itemsM = itemsW1.map { tWM1.inverse().transformPoint(it) }
    // A spot added while located: mapPose = inverse(T_WM) ∘ worldPose.
    val spotW1 = Pose4.fromAxis(Vec3(-3.0, 1.3, 2.5), tiltedAxis(yaw = -1.9, tiltDeg = 0.0))
    val spotMapPose = tWM1.inverse().compose(spotW1)

    // --- session 2 (world W2 = G(W1), any 4-DoF change)
    val g = Pose4(5.7, -0.4, -12.3, 2.6)
    val expectedW2 = itemsW1.map { g.transformPoint(it) }

    // Locate from the picture: same physical picture, seen in the new world frame.
    val imageCenter2 = g.transformPoint(imageCenter1)
    val imageX2 = g.rotate(imageX1)
    val tWM2 = Pose4.fromAxis(imageCenter2, imageX2).compose(imageMapPose.inverse())
    for ((m, e) in itemsM.zip(expectedW2)) assertTrue(distance(tWM2.transformPoint(m), e) < 1e-3)

    // Locate from the spot instead.
    val spotW2 = g.compose(spotW1)
    val tWM2b = spotW2.compose(spotMapPose.inverse())
    for ((m, e) in itemsM.zip(expectedW2)) assertTrue(distance(tWM2b.transformPoint(m), e) < 1e-3)
  }

  private fun tiltedAxis(yaw: Double, tiltDeg: Double): Vec3 {
    val t = Math.toRadians(tiltDeg)
    return Vec3(cos(yaw) * cos(t), sin(t), -sin(yaw) * cos(t))
  }

  /** v' = q v q*, quaternion (x, y, z, w). */
  private fun quatRotate(q: FloatArray, v: Vec3): Vec3 {
    val qx = q[0].toDouble()
    val qy = q[1].toDouble()
    val qz = q[2].toDouble()
    val qw = q[3].toDouble()
    val n = sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    val x = qx / n
    val y = qy / n
    val z = qz / n
    val w = qw / n
    // t = 2 * cross(q.xyz, v); v' = v + w * t + cross(q.xyz, t)
    val tx = 2 * (y * v.z - z * v.y)
    val ty = 2 * (z * v.x - x * v.z)
    val tz = 2 * (x * v.y - y * v.x)
    return Vec3(
      v.x + w * tx + (y * tz - z * ty),
      v.y + w * ty + (z * tx - x * tz),
      v.z + w * tz + (x * ty - y * tx),
    )
  }
}
