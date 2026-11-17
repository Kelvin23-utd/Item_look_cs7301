package edu.cs6301.findback.scan

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Matrix
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.objectdetector.ObjectDetector
import edu.cs6301.findback.ml.CameraFrame

/**
 * A detector that names many objects per frame (plan section 3). Boxes are in sensor image pixels,
 * like every other rectangle in the app. Called from the scan worker thread only.
 */
interface SceneDetector {
  /** Logged with every scan: "effdet0" (more later: "effdet2", "mlkit"). */
  val id: String
  /** Model file name, for `model_loaded`. */
  val modelName: String

  fun detect(frame: CameraFrame): List<SceneDetection>

  fun close()
}

/**
 * `effdet0`: MediaPipe Tasks ObjectDetector with EfficientDet-Lite0 (80 COCO classes, 320×320).
 *
 * Model: `assets/models/efficientdet_lite0.tflite`, the int8 variant from Google's MediaPipe model
 * storage, pinned to version 1 (the same bytes as `latest` on 2026-11-16):
 *   https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/int8/1/efficientdet_lite0.tflite
 *   4 602 795 bytes, sha256 0720bf247bd76e6594ea28fa9c6f7c5242be774818997dbbeffc4da460c723bb
 * The MediaPipe Object Detector page lists this file but states no licence, and the file's
 * metadata carries none. The upstream EfficientDet-Lite0 checkpoints, published by TensorFlow on
 * Kaggle (formerly TF Hub, `tensorflow/efficientdet`), are Apache 2.0 in every variant.
 *
 * The detector sees the upright image (as ML Kit does in `BoxDetector`) at half size: it resizes to
 * 320×320 itself, so the full 1920×1080 frame would only cost a larger rotation.
 */
class EffDet0Detector(context: Context) : SceneDetector {
  companion object {
    const val MODEL = "efficientdet_lite0"
    private const val INPUT_SCALE = ScanConfig.DETECTOR_INPUT_SCALE
  }

  override val id = ScanConfig.DETECTOR_EFFDET0
  override val modelName = MODEL

  private val detector: ObjectDetector =
    ObjectDetector.createFromOptions(
      context,
      ObjectDetector.ObjectDetectorOptions.builder()
        .setBaseOptions(BaseOptions.builder().setModelAssetPath("models/$MODEL.tflite").build())
        .setRunningMode(RunningMode.IMAGE)
        .setScoreThreshold(ScanConfig.RAW_SCORE_MIN)
        .setMaxResults(ScanConfig.RAW_MAX_RESULTS)
        .build()
    )

  override fun detect(frame: CameraFrame): List<SceneDetection> {
    val sensor = frame.sensor
    val m = Matrix().apply {
      postScale(INPUT_SCALE.toFloat(), INPUT_SCALE.toFloat())
      postRotate(frame.rotation.toFloat())
    }
    val input = Bitmap.createBitmap(sensor, 0, 0, sensor.width, sensor.height, m, true)
    val argb = if (input.config == Bitmap.Config.ARGB_8888) input else input.copy(Bitmap.Config.ARGB_8888, false)
    val result = detector.detect(BitmapImageBuilder(argb).build())
    val out = ArrayList<SceneDetection>()
    for (d in result.detections()) {
      val best = d.categories().maxByOrNull { it.score() } ?: continue
      val r = d.boundingBox()
      val box =
        Upright.boxToSensor(
          r.left.toDouble(), r.top.toDouble(), r.right.toDouble(), r.bottom.toDouble(),
          INPUT_SCALE, frame.rotation, sensor.width, sensor.height,
        )
      out.add(SceneDetection(box, best.categoryName() ?: "", best.score()))
    }
    return out
  }

  override fun close() = detector.close()
}
