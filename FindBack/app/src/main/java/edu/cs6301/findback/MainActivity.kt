package edu.cs6301.findback

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.Build
import android.os.Bundle
import android.util.Log
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import com.google.ar.core.AugmentedImageDatabase
import com.google.ar.core.CameraConfig
import com.google.ar.core.CameraConfigFilter
import com.google.ar.core.Config
import com.google.ar.core.examples.java.common.helpers.FullScreenHelper
import com.google.ar.core.examples.kotlin.common.helpers.ARCoreSessionLifecycleHelper
import com.google.ar.core.exceptions.CameraNotAvailableException
import com.google.ar.core.exceptions.UnavailableApkTooOldException
import com.google.ar.core.exceptions.UnavailableArcoreNotInstalledException
import com.google.ar.core.exceptions.UnavailableDeviceNotCompatibleException
import com.google.ar.core.exceptions.UnavailableSdkTooOldException
import com.google.ar.core.exceptions.UnavailableUserDeclinedInstallationException
import edu.cs6301.findback.map.HomeMap
import edu.cs6301.findback.map.MapStore
import edu.cs6301.findback.memory.EventLog
import edu.cs6301.findback.memory.MemoryStore
import edu.cs6301.findback.study.CaptureMode
import edu.cs6301.findback.study.StudySettings
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.UUID

class MainActivity : AppCompatActivity() {
  companion object {
    private const val TAG = "FindBack"

    /** Name of the room reference image in the ARCore augmented image database. */
    const val REF_NAME = "room_reference"

    /** Default displayed width of assets/reference.jpg (printed A4 sheet), in metres. */
    const val DEFAULT_REF_WIDTH_M = 0.20f
    private const val PREFS = "findback"
    private const val KEY_REF_WIDTH = "refWidthM"
    private const val KEY_STUDY_TAG = "studyTag"
    private const val KEY_CAPTURE_MODE = "captureMode"
  }

  /** New id per app launch; same-session anchors are only valid within one id. */
  val sessionId: String = UUID.randomUUID().toString().substring(0, 8)

  lateinit var store: MemoryStore
  lateinit var events: EventLog
  lateinit var mapStore: MapStore
  /** map.json as loaded at start-up (null: no map yet). The renderer's Localizer owns it after that. */
  var loadedMap: HomeMap? = null
    private set
  private lateinit var filesRoot: File
  lateinit var sessionHelper: ARCoreSessionLifecycleHelper
  lateinit var renderer: FindBackRenderer
  lateinit var view: MainView

  private lateinit var referenceBitmap: Bitmap

  /**
   * Measured width of the reference picture as shown in the room (printout or iPad screen).
   * ARCore uses it to size the room frame, so set it before registering objects and keep it fixed.
   */
  val refWidthM: Float
    get() = getSharedPreferences(PREFS, MODE_PRIVATE).getFloat(KEY_REF_WIDTH, DEFAULT_REF_WIDTH_M)

  /** Save a new reference width and restart the activity so ARCore rebuilds its image database. */
  fun setRefWidthM(widthM: Float) {
    events.log("ref_width_set", mapOf("from" to refWidthM, "to" to widthM))
    getSharedPreferences(PREFS, MODE_PRIVATE).edit().putFloat(KEY_REF_WIDTH, widthM).apply()
    recreate()
  }

  // Study settings (spec v0.4, S1), stored in preferences and read once at start. Volatile: the
  // GL thread and the capture coroutines read them when they build log events.
  /** Free text that labels the logs, by convention `key=value;key=value`. Empty by default. */
  @Volatile
  var studyTag: String = ""
    private set

  @Volatile
  var captureMode: CaptureMode = CaptureMode.FREE
    private set

  /** `studyTag` and `captureMode` as log fields. */
  fun studyFields(): Map<String, Any?> = mapOf("studyTag" to studyTag, "captureMode" to captureMode.name)

  /** Location sheet → Study settings: store and log a change. Takes effect at once, no restart. */
  fun setStudySettings(tag: String, mode: CaptureMode) {
    val clean = StudySettings.cleanTag(tag)
    if (clean == studyTag && mode == captureMode) return
    events.log(
      "study_settings",
      mapOf(
        "studyTag" to clean,
        "captureMode" to mode.name,
        "prevStudyTag" to studyTag,
        "prevCaptureMode" to captureMode.name,
      )
    )
    getSharedPreferences(PREFS, MODE_PRIVATE)
      .edit()
      .putString(KEY_STUDY_TAG, clean)
      .putString(KEY_CAPTURE_MODE, mode.name)
      .apply()
    studyTag = clean
    captureMode = mode
  }

  /**
   * "New map" (confirmed in the location sheet): move records.json and map.json into
   * archive/<yyyyMMdd-HHmmss>/ and restart. Crops, verification frames, ghost images and
   * events.jsonl stay where they are. Nothing is deleted.
   */
  fun startNewMap(mapId: String?) {
    val stamp = SimpleDateFormat("yyyyMMdd-HHmmss", Locale.US).format(Date())
    val rel = "archive/$stamp"
    val dir = File(filesRoot, rel)
    val records = store.snapshot().size
    val recordsMoved = store.archiveTo(dir)
    val mapMoved = mapStore.archiveTo(dir)
    events.log(
      "map_archived",
      mapOf("mapId" to mapId, "dir" to rel, "records" to records, "recordsMoved" to recordsMoved, "mapMoved" to mapMoved)
    )
    recreate()
  }

  override fun onCreate(savedInstanceState: Bundle?) {
    super.onCreate(savedInstanceState)

    val root = getExternalFilesDir(null) ?: filesDir
    filesRoot = root
    getSharedPreferences(PREFS, MODE_PRIVATE).let { prefs ->
      studyTag = StudySettings.cleanTag(prefs.getString(KEY_STUDY_TAG, ""))
      captureMode = CaptureMode.parse(prefs.getString(KEY_CAPTURE_MODE, null))
    }
    store = MemoryStore(root, sessionId)
    events = EventLog(root, sessionId)
    mapStore =
      MapStore(
        root,
        onSaved = { m ->
          events.log(
            "map_saved",
            mapOf(
              "mapId" to m.mapId,
              "nodes" to m.nodes.size,
              "edges" to m.edges.size,
              "rooms" to m.rooms.size,
              "fixPoints" to m.fixPoints.size,
            )
          )
        },
        onError = { msg, e -> Log.e(TAG, msg, e) },
      )
    loadedMap = mapStore.load()
    mapStore.startAutosave()
    events.log(
      "session_start",
      mapOf(
        "records" to store.snapshot().size,
        "device" to Build.MODEL,
        "android" to Build.VERSION.RELEASE,
        "refWidthM" to refWidthM,
        // v0.3
        "mapId" to loadedMap?.mapId,
        "mapNodes" to (loadedMap?.nodes?.size ?: 0),
        "mapFixPoints" to (loadedMap?.fixPoints?.size ?: 0),
        "mapRooms" to (loadedMap?.rooms?.size ?: 0),
      ) + studyFields() // v0.4
    )

    referenceBitmap = assets.open("reference.jpg").use { BitmapFactory.decodeStream(it) }

    sessionHelper = ARCoreSessionLifecycleHelper(this)
    sessionHelper.exceptionCallback = { exception ->
      val message =
        when (exception) {
          is UnavailableArcoreNotInstalledException,
          is UnavailableUserDeclinedInstallationException -> "Please install Google Play Services for AR"
          is UnavailableApkTooOldException -> "Please update Google Play Services for AR"
          is UnavailableSdkTooOldException -> "Please update this app"
          is UnavailableDeviceNotCompatibleException -> "This device does not support AR"
          is CameraNotAvailableException -> "Camera not available. Try restarting the app."
          else -> "Failed to create AR session: $exception"
        }
      Log.e(TAG, message, exception)
      Toast.makeText(this, message, Toast.LENGTH_LONG).show()
    }

    sessionHelper.beforeSessionResume = { session ->
      val config = session.config
      config.focusMode = Config.FocusMode.AUTO
      if (session.isDepthModeSupported(Config.DepthMode.AUTOMATIC)) {
        config.depthMode = Config.DepthMode.AUTOMATIC
      }
      config.augmentedImageDatabase =
        AugmentedImageDatabase(session).apply { addImage(REF_NAME, referenceBitmap, refWidthM) }
      session.configure(config)

      // Highest CPU image resolution gives sharper object crops.
      val filter = CameraConfigFilter(session).setFacingDirection(CameraConfig.FacingDirection.BACK)
      val best =
        session
          .getSupportedCameraConfigs(filter)
          .sortedWith(
            compareByDescending<CameraConfig> { it.imageSize.width }.thenByDescending { it.imageSize.height }
          )
          .first()
      session.cameraConfig = best
      Log.i(TAG, "Camera config: CPU image ${best.imageSize}, GPU texture ${best.textureSize}")
    }
    lifecycle.addObserver(sessionHelper)

    renderer = FindBackRenderer(this)
    lifecycle.addObserver(renderer)
    view = MainView(this, renderer)
    setContentView(view.root)
    renderer.bindView(view)
    lifecycle.addObserver(view)
  }

  override fun onRequestPermissionsResult(
    requestCode: Int,
    permissions: Array<out String>,
    grantResults: IntArray
  ) {
    super.onRequestPermissionsResult(requestCode, permissions, grantResults)
    sessionHelper.onRequestPermissionsResult(requestCode, permissions, grantResults)
  }

  override fun onWindowFocusChanged(hasFocus: Boolean) {
    super.onWindowFocusChanged(hasFocus)
    FullScreenHelper.setFullScreenOnWindowFocusChanged(this, hasFocus)
  }
}
