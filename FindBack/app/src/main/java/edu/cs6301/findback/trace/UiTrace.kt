package edu.cs6301.findback.trace

import android.content.Context
import android.content.DialogInterface
import android.os.SystemClock
import androidx.appcompat.app.AlertDialog
import edu.cs6301.findback.memory.EventLog

/**
 * v0.7 detailed trace (docs/PLAN_v0.7_trace.md, section 3): the UI events `ui_screen`,
 * `ui_action`, `ui_search` and `ui_message`. UI thread only. With the trace off ([on] false) every
 * method does nothing and [dialog] returns a plain [AlertDialog.Builder].
 *
 * [screen] gives the screen as last drawn ([edu.cs6301.findback.ui.Screen] name), for the
 * `screen` field.
 */
class UiTrace(private val events: EventLog, val on: Boolean, private val screen: () -> String?) {
  private var screenSinceMs = 0L

  /** `ui_action`: a button press or a choice. [detail]: the answer, the filter, and so on. */
  fun action(action: String, detail: String? = null) {
    if (!on) return
    events.log("ui_action", mapOf("action" to action, "screen" to screen(), "detail" to detail))
  }

  /** `ui_message`: a banner message as shown. */
  fun message(text: String) {
    if (!on) return
    events.log("ui_message", mapOf("text" to text, "screen" to screen()))
  }

  /** `ui_screen`: the screen changed from [from] (null before the first) to [to]. */
  fun screenChanged(from: String?, to: String) {
    if (!on) return
    val now = SystemClock.elapsedRealtime()
    val ms = if (from == null) null else now - screenSinceMs
    screenSinceMs = now
    events.log("ui_screen", mapOf("from" to from, "to" to to, "msInPrevious" to ms))
  }

  /**
   * `ui_search`: the Items search settled. The typed text itself is never logged. v0.9:
   * [extraNames] the setting, [namesBound] items with bound extra names, [viaExtra] shown rows that
   * are results only because of extra names, [lastScanOnly] the "Last scan" chip.
   */
  fun search(
    chars: Int,
    results: Int,
    total: Int,
    roomFilter: String?,
    extraNames: Boolean,
    namesBound: Int,
    viaExtra: Int,
    lastScanOnly: Boolean,
  ) {
    if (!on) return
    events.log(
      "ui_search",
      mapOf(
        "chars" to chars,
        "results" to results,
        "total" to total,
        "roomFilter" to roomFilter,
        "extraNames" to extraNames,
        "namesBound" to namesBound,
        "viaExtra" to viaExtra,
        "lastScanOnly" to lastScanOnly,
      ),
    )
  }

  /**
   * v0.9 `ui_action` `items_row`: a row opened. [kind] ("record", "entry") is the detail; [rank]
   * its place in the shown list (1 = top), [shown] the rows shown, [chars] the query length,
   * [matchClass] 0 to 3 (null for an empty query), [viaExtra] a result only because of extra names.
   */
  fun itemsRow(kind: String, rank: Int, shown: Int, chars: Int, matchClass: Int?, viaExtra: Boolean) {
    if (!on) return
    events.log(
      "ui_action",
      mapOf(
        "action" to "items_row",
        "screen" to screen(),
        "detail" to kind,
        "rank" to rank,
        "shown" to shown,
        "chars" to chars,
        "matchClass" to matchClass,
        "viaExtra" to viaExtra,
      ),
    )
  }

  /**
   * An [AlertDialog.Builder] whose button, list and Back answers are logged as `ui_action` with
   * action `dialog:<name>` and the answer's label as detail ("back" for Back or a tap outside).
   */
  fun dialog(context: Context, name: String): AlertDialog.Builder =
    if (on) TracedDialogBuilder(context, "dialog:$name", ::action) else AlertDialog.Builder(context)
}

/**
 * Wraps each listener so the answer is logged first, then handled exactly as before. A button
 * given without a listener gets one that only logs: AlertDialog dismisses after any button press
 * whether or not it has a listener, so nothing else changes. A dialog without its own cancel
 * listener gets one that only logs "back".
 */
private class TracedDialogBuilder(
  context: Context,
  private val name: String,
  private val log: (String, String?) -> Unit,
) : AlertDialog.Builder(context) {
  init {
    super.setOnCancelListener { log(name, BACK) }
  }

  private fun wrap(label: CharSequence?, l: DialogInterface.OnClickListener?) =
    DialogInterface.OnClickListener { d, which ->
      log(name, label?.toString())
      l?.onClick(d, which)
    }

  override fun setPositiveButton(text: CharSequence?, listener: DialogInterface.OnClickListener?): AlertDialog.Builder =
    super.setPositiveButton(text, wrap(text, listener))

  override fun setNegativeButton(text: CharSequence?, listener: DialogInterface.OnClickListener?): AlertDialog.Builder =
    super.setNegativeButton(text, wrap(text, listener))

  override fun setNeutralButton(text: CharSequence?, listener: DialogInterface.OnClickListener?): AlertDialog.Builder =
    super.setNeutralButton(text, wrap(text, listener))

  override fun setItems(items: Array<out CharSequence>?, listener: DialogInterface.OnClickListener?): AlertDialog.Builder =
    super.setItems(
      items,
      DialogInterface.OnClickListener { d, which ->
        log(name, items?.getOrNull(which)?.toString() ?: "item $which")
        listener?.onClick(d, which)
      },
    )

  override fun setOnCancelListener(onCancelListener: DialogInterface.OnCancelListener?): AlertDialog.Builder =
    super.setOnCancelListener { d ->
      log(name, BACK)
      onCancelListener?.onCancel(d)
    }

  companion object {
    const val BACK = "back"
  }
}
