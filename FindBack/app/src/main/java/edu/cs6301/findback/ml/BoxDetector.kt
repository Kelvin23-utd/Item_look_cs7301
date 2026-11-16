package edu.cs6301.findback.ml

import android.graphics.RectF
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.objects.ObjectDetection
import com.google.mlkit.vision.objects.defaults.ObjectDetectorOptions
import kotlinx.coroutines.tasks.await

/** One detector proposal. [label] is ML Kit's coarse category, or "" when it gave none. */
data class Detection(val sensorRect: RectF, val label: String, val confidence: Float)

/**
 * ML Kit object detector. We use it for its class-agnostic boxes; its five coarse categories
 * (Fashion good, Food, Home good, Place, Plant) are recorded as raw predictions only.
 */
class BoxDetector {
  private val detector =
    ObjectDetection.getClient(
      ObjectDetectorOptions.Builder()
        .setDetectorMode(ObjectDetectorOptions.SINGLE_IMAGE_MODE)
        .enableMultipleObjects()
        .enableClassification()
        .build()
    )

  suspend fun detect(frame: CameraFrame): List<Detection> {
    val input = InputImage.fromBitmap(frame.upright, 0)
    return detector.process(input).await().map { obj ->
      val best = obj.labels.maxByOrNull { it.confidence }
      Detection(frame.uprightRectToSensor(obj.boundingBox), best?.text ?: "", best?.confidence ?: 0f)
    }
  }

  fun close() = detector.close()
}
