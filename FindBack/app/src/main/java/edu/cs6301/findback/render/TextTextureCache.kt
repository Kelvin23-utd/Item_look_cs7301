/*
 * Adapted from the ARCore ml_kotlin sample (Copyright 2021 Google LLC, Apache License 2.0).
 * Change: labels draw a dot at the exact anchor point, and support a highlight color.
 */
package edu.cs6301.findback.render

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Paint
import android.graphics.Typeface
import android.opengl.GLES30
import com.google.ar.core.examples.java.common.samplerender.GLError
import com.google.ar.core.examples.java.common.samplerender.SampleRender
import com.google.ar.core.examples.java.common.samplerender.Texture
import java.nio.ByteBuffer

/** Generates and caches GL textures for label names. */
class TextTextureCache {
  private val cacheMap = mutableMapOf<String, Texture>()

  fun get(render: SampleRender, string: String, highlight: Boolean = false): Texture {
    val key = (if (highlight) "H|" else "N|") + string
    return cacheMap.getOrPut(key) { generateTexture(render, string, highlight) }
  }

  private fun generateTexture(render: SampleRender, string: String, highlight: Boolean): Texture {
    val texture = Texture(render, Texture.Target.TEXTURE_2D, Texture.WrapMode.CLAMP_TO_EDGE)
    val bitmap = generateBitmapFromString(string, highlight)
    val buffer = ByteBuffer.allocateDirect(bitmap.byteCount)
    bitmap.copyPixelsToBuffer(buffer)
    buffer.rewind()

    GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, texture.textureId)
    GLError.maybeThrowGLException("Failed to bind texture", "glBindTexture")
    GLES30.glTexImage2D(
      GLES30.GL_TEXTURE_2D, 0, GLES30.GL_RGBA8, bitmap.width, bitmap.height, 0,
      GLES30.GL_RGBA, GLES30.GL_UNSIGNED_BYTE, buffer
    )
    GLError.maybeThrowGLException("Failed to populate texture data", "glTexImage2D")
    GLES30.glGenerateMipmap(GLES30.GL_TEXTURE_2D)
    GLError.maybeThrowGLException("Failed to generate mipmaps", "glGenerateMipmap")
    return texture
  }

  private fun fillPaint(highlight: Boolean) =
    Paint().apply {
      textSize = 28f
      if (highlight) setARGB(0xff, 0x34, 0xd3, 0x99) else setARGB(0xff, 0xff, 0xc1, 0x07)
      style = Paint.Style.FILL
      isAntiAlias = true
      textAlign = Paint.Align.CENTER
      typeface = Typeface.DEFAULT_BOLD
    }

  private fun strokePaint(highlight: Boolean) =
    Paint(fillPaint(highlight)).apply {
      setARGB(0xff, 0x00, 0x00, 0x00)
      style = Paint.Style.STROKE
      strokeWidth = 4f
    }

  private fun generateBitmapFromString(string: String, highlight: Boolean): Bitmap {
    val w = 256
    val h = 256
    val fill = fillPaint(highlight)
    val stroke = strokePaint(highlight)
    return Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888).apply {
      eraseColor(0)
      Canvas(this).apply {
        // Dot marks the exact remembered point (texture center = anchor point).
        val r = if (highlight) 12f else 8f
        drawCircle(w / 2f, h / 2f, r, stroke)
        drawCircle(w / 2f, h / 2f, r, fill)
        drawText(string, w / 2f, h / 2f - 28f, stroke)
        drawText(string, w / 2f, h / 2f - 28f, fill)
      }
    }
  }
}
