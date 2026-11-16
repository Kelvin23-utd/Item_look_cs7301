package edu.cs6301.findback.memory

import android.util.Log
import java.io.File
import java.util.concurrent.Executors
import org.json.JSONObject

/**
 * Append-only JSONL log (events.jsonl). Every line has t (epoch ms), session and type. This is the
 * raw material for the offline evaluator: registrations, queries, policy decisions, verification
 * scores, ground truth and latencies.
 */
class EventLog(root: File, private val sessionId: String) {
  private val file = File(root, "events.jsonl")
  private val writer = Executors.newSingleThreadExecutor()

  fun log(type: String, fields: Map<String, Any?> = emptyMap()) {
    val o = JSONObject()
    o.put("t", System.currentTimeMillis())
    o.put("session", sessionId)
    o.put("type", type)
    for ((k, v) in fields) o.put(k, v ?: JSONObject.NULL)
    val line = o.toString()
    Log.i("FindBackEvent", line)
    writer.execute { file.appendText(line + "\n") }
  }
}
