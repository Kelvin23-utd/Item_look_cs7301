package edu.cs6301.findback.scan

import edu.cs6301.findback.map.Pose4
import edu.cs6301.findback.map.Vec3
import kotlin.math.cos
import kotlin.math.sin
import kotlin.math.sqrt

/**
 * A synthetic camera like the Pixel 8's in the study logs (1920×1080 CPU image, fx ≈ 1387.5) with a
 * 160×90 depth image, and helpers to build poses, depth images and embeddings for the scan tests.
 */
object ScanFixtures {
  const val W = 1920
  const val H = 1080
  const val DW = 160
  const val DH = 90
  val K = Intrinsics(1387.5, 1387.5, 960.0, 540.0, W, H)

  /** CPU image pixels to depth pixels when the depth image covers the whole CPU image. */
  val FULL = Affine2(DW.toDouble() / W, 0.0, 0.0, 0.0, DH.toDouble() / H, 0.0)

  /** A map frame that differs from the world frame by a yaw and a translation. */
  val T_WM = Pose4(1.2, -0.4, 3.1, 0.6)

  private fun mul(a: DoubleArray, b: DoubleArray): DoubleArray =
    doubleArrayOf(
      a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1],
      a[3] * b[1] - a[0] * b[2] + a[1] * b[3] + a[2] * b[0],
      a[3] * b[2] + a[0] * b[1] - a[1] * b[0] + a[2] * b[3],
      a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2],
    )

  private fun axisAngle(x: Double, y: Double, z: Double, deg: Double): DoubleArray {
    val h = Math.toRadians(deg) / 2
    return doubleArrayOf(x * sin(h), y * sin(h), z * sin(h), cos(h))
  }

  /**
   * Camera at [position] (world frame), turned [yawDeg] about world +Y, then pitched [pitchDeg]
   * about its own +X (positive looks up), then rolled [rollDeg] about its own +Z. With all three
   * zero it looks along world -Z with +X to the right and +Y up, as an ARCore camera pose does.
   */
  fun pose(position: Vec3, yawDeg: Double = 0.0, pitchDeg: Double = 0.0, rollDeg: Double = 0.0): CameraPose {
    val q = mul(mul(axisAngle(0.0, 1.0, 0.0, yawDeg), axisAngle(1.0, 0.0, 0.0, pitchDeg)), axisAngle(0.0, 0.0, 1.0, rollDeg))
    return CameraPose(position, q[0], q[1], q[2], q[3])
  }

  /** A depth image with the same depth everywhere (a wall facing the camera). */
  fun flatDepth(metres: Double, fromImage: Affine2 = FULL): DepthMap =
    DepthMap(DW, DH, ShortArray(DW * DH) { (metres * 1000).toInt().toShort() }, fromImage)

  /** A depth image from a function of the depth pixel (metres; 0 for "no estimate"). */
  fun depthOf(fromImage: Affine2 = FULL, f: (x: Int, y: Int) -> Double): DepthMap =
    DepthMap(DW, DH, ShortArray(DW * DH) { i -> (f(i % DW, i / DW) * 1000).toInt().toShort() }, fromImage)

  fun geometry(pose: CameraPose, depth: DepthMap, tWM: Pose4 = T_WM) = KeyframeGeometry(K, pose, tWM, depth)

  /**
   * A unit embedding of [dim] values that is [cos] similar to [unit] (index [axis], 1 elsewhere 0):
   * cos on that axis, the rest on axis [other].
   */
  fun embedding(axis: Int, cos: Double = 1.0, other: Int = 7, dim: Int = 8): FloatArray {
    val e = FloatArray(dim)
    e[axis] = cos.toFloat()
    e[other] = sqrt(1.0 - cos * cos).toFloat()
    return e
  }

  /** A box of [sidePx] centred on a pixel. */
  fun boxAt(u: Double, v: Double, sidePx: Double) =
    Box((u - sidePx / 2).toFloat(), (v - sidePx / 2).toFloat(), (u + sidePx / 2).toFloat(), (v + sidePx / 2).toFloat())

  /**
   * The detection a perfect detector and a perfect depth image would give for an item of
   * [sizeM] at map point [p], seen from [pose]: null when it is not in the image.
   */
  fun observe(
    p: Vec3,
    pose: CameraPose,
    label: String,
    embedding: FloatArray,
    sizeM: Double = 0.10,
    score: Float = 0.8f,
    tWM: Pose4 = T_WM,
  ): Observation? {
    // Detect in a depth image that has the item's own depth everywhere.
    val probe = geometry(pose, flatDepth(1.0), tWM)
    val pr = probe.projectMap(p) ?: return null
    if (!probe.inImage(pr, 0.0)) return null
    val geom = geometry(pose, flatDepth(pr.depthM), tWM)
    val side = sizeM * K.fx / pr.depthM
    val report = DetectionFilter.apply(listOf(SceneDetection(boxAt(pr.u, pr.v, side), label, score)), geom).single()
    if (!report.kept) return null
    val box = report.detection.box
    val crop = box.padded(ScanConfig.CROP_PAD)
    return Observation(
      label = report.detection.label,
      score = score,
      box = box,
      mapPoint = report.mapPoint!!,
      worldPoint = report.worldPoint!!,
      depthM = report.depthM!!,
      distanceM = report.distanceM!!,
      sizeM = report.sizeM!!,
      embedding = embedding,
      cropWidthPx = crop.width,
      cropHeightPx = crop.height,
    )
  }

  fun keyframe(index: Int, tMs: Long, pose: CameraPose, room: String? = "room1", tWM: Pose4 = T_WM) =
    KeyframeInfo(index, tMs, room, tWM.inverse().transformPoint(pose.position), W, H)

  /** Camera standing [back] metres in front of map point [p] (on its world -Z side), shifted sideways. */
  fun poseLookingAt(p: Vec3, back: Double = 1.0, side: Double = 0.0, tWM: Pose4 = T_WM): CameraPose {
    val w = tWM.transformPoint(p)
    return pose(Vec3(w.x + side, w.y, w.z + back))
  }
}
