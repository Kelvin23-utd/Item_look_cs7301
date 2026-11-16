package edu.cs6301.findback.ml

import android.content.Context
import android.graphics.Bitmap
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.imageembedder.ImageEmbedder
import kotlin.math.sqrt

/**
 * Frozen appearance model shared by every experimental condition (MediaPipe MobileNet-V3 small).
 * Produces L2-normalised embeddings, so cosine similarity is the dot product.
 */
class Embedder(context: Context) {
  val modelName = "mobilenet_v3_small"

  private val embedder: ImageEmbedder =
    ImageEmbedder.createFromOptions(
      context,
      ImageEmbedder.ImageEmbedderOptions.builder()
        .setBaseOptions(
          BaseOptions.builder().setModelAssetPath("models/$modelName.tflite").build()
        )
        .setL2Normalize(true)
        .setQuantize(false)
        .setRunningMode(RunningMode.IMAGE)
        .build()
    )

  @Synchronized
  fun embed(bitmap: Bitmap): FloatArray {
    val argb = if (bitmap.config == Bitmap.Config.ARGB_8888) bitmap
    else bitmap.copy(Bitmap.Config.ARGB_8888, false)
    val result = embedder.embed(BitmapImageBuilder(argb).build())
    return result.embeddingResult().embeddings()[0].floatEmbedding()
  }

  fun close() = embedder.close()

  companion object {
    fun cosine(a: FloatArray, b: FloatArray): Float {
      if (a.size != b.size || a.isEmpty()) return 0f
      var dot = 0f
      var na = 0f
      var nb = 0f
      for (i in a.indices) {
        dot += a[i] * b[i]
        na += a[i] * a[i]
        nb += b[i] * b[i]
      }
      val denom = sqrt(na) * sqrt(nb)
      return if (denom == 0f) 0f else dot / denom
    }
  }
}
