package vn.edu.usth.objectdetectmobile;

import android.Manifest;
import android.annotation.SuppressLint;
import android.content.SharedPreferences;
import android.content.pm.PackageManager;
import android.hardware.camera2.CameraCharacteristics;
import android.os.Bundle;
import android.util.Log;
import android.util.Size;
import android.view.View;
import android.widget.SeekBar;
import android.widget.TextView;
import android.widget.Toast;

import androidx.activity.ComponentActivity;
import androidx.annotation.NonNull;
import androidx.annotation.OptIn;
import androidx.camera.camera2.interop.Camera2CameraInfo;
import androidx.camera.camera2.interop.ExperimentalCamera2Interop;
import androidx.camera.core.Camera;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageProxy;
import androidx.camera.core.Preview;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.camera.view.PreviewView;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import androidx.lifecycle.LifecycleOwner;

// Non-deprecated resolution selector API
import androidx.camera.core.resolutionselector.AspectRatioStrategy;
import androidx.camera.core.resolutionselector.ResolutionSelector;
import androidx.camera.core.resolutionselector.ResolutionStrategy;

import com.google.android.material.button.MaterialButton;
import com.google.android.material.switchmaterial.SwitchMaterial;
import vn.edu.usth.objectdetectmobile.utils.TTSWarning;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Set;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.atomic.AtomicBoolean;

import ai.onnxruntime.OrtException;


import android.os.Bundle;

import android.content.Intent;
import android.net.Uri;
import androidx.appcompat.app.AlertDialog;
import android.app.DownloadManager;
import android.os.Environment;
import android.database.Cursor;
import android.os.Build;
import android.content.IntentFilter;
import android.widget.ImageButton;
import android.os.SystemClock;
import android.content.Context;

public class MainActivity extends ComponentActivity {
    // ---- Latency logging ----
    private static final int LAT_LOG_EVERY_N_FRAMES = 15;
    private int latFrameCounter = 0;
    private boolean cameraTsIsRealtime = false;
    private long realtimeMinusUptimeOffsetNs = 0;
    // ---------------------------------------------------------------------------------------------
    //  Environment mode (indoor / outdoor)
    // ---------------------------------------------------------------------------------------------
    public enum EnvMode {
        INDOOR,
        OUTDOOR
    }
    // Depth model prefs live in a separate file
    private static final String DEPTH_MODEL_PREFS = "depth_models";
    private SharedPreferences depthModelPrefs;
    private static final String PREF_ENV_MODE = "pref_env_mode";
    private static final String PREF_BLUR_ENABLED = "pref_blur_enabled";
    private static final String PREF_DEPTH_MODE = "pref_depth_mode"; // MONO or STEREO
    private static final String PREF_DEPTH_ASYNC = "pref_depth_async";
    private static final boolean DEFAULT_DEPTH_ASYNC = true;

    private EnvMode envMode = EnvMode.OUTDOOR;  // default = Outdoor
    private SwitchMaterial environmentSwitch;
    // ---------------------------------------------------------------------------------------------
    //  Constants
    // ---------------------------------------------------------------------------------------------
    private static final int REQ = 42;
    private static final String TAG = "MainActivity";

    // Depth throttling / cache
    // Set interval to 0 to disable depth throttling and run depth continuously.
    private static final short DEPTH_INTERVAL_MS = 0;
    private static final short DEPTH_CACHE_MS = 3000;

    // Input blur
    private static final boolean ENABLE_INPUT_BLUR = true;
    private static final int BLUR_RADIUS = 1; // 1 => kernel 3x3
    private static final int ANALYSIS_INPUT_SIZE = 640;
    private static final int BLUR_INPUT_SIZE = 360;

    // ---------------------------------------------------------------------------------------------
    //  UI views
    // ---------------------------------------------------------------------------------------------
    private PreviewView previewView;
    private OverlayView overlay;
    private SwitchMaterial realtimeSwitch;
    private SwitchMaterial blurSwitch;
    private SwitchMaterial stereoSwitch;
    private MaterialButton detectOnceButton;
    private ImageButton quickSettingsButton;
    private ImageButton settingsButton;

    private View controlPanel;
    private TextView depthModeText;
    private SeekBar calibrationSeek;
    private TextView calibrationValue;
    private SeekBar zoomSeek;
    private TextView zoomValue;
    // ---------------------------------------------------------------------------------------------
    //  Core components
    // ---------------------------------------------------------------------------------------------
    private ObjectDetector detectorOd;
    private ObjectDetector detectorSeg;
    private volatile DepthEstimator depthEstimator;     //Marking the field volatile guarantees visibility of the latest reference across threads
    private StereoDepthProcessor stereoProcessor;
    private ProcessCameraProvider cameraProvider;
    private Camera currentCamera;

    // CameraX analysis executor (single thread)
    private ExecutorService exec;
    // Inference executor (YOLO + depth in parallel)
    private ExecutorService inferenceExec;

    // ---------------------------------------------------------------------------------------------
    //  Depth & stereo state
    // ---------------------------------------------------------------------------------------------
    // Reuse your existing helper state holder
    final DepthPipelineHelper.DepthState depthState =
            new DepthPipelineHelper.DepthState();
    // Prevent overlapping depth runs when we make depth async.
    private final AtomicBoolean depthBusy = new AtomicBoolean(false);

    private volatile boolean stereoFusionEnabled = false;
    private boolean stereoPipelineAvailable = false;
    private volatile boolean sequentialStereoRunning = false;
    private List<CameraUtils.CamInfo> backCameraInfos = new ArrayList<>();

    // ---------------------------------------------------------------------------------------------
    //  Detection / realtime state
    // ---------------------------------------------------------------------------------------------
    private volatile boolean realtimeEnabled = true;
    private volatile boolean blurEnabled = ENABLE_INPUT_BLUR;
    private volatile boolean depthAsyncEnabled = DEFAULT_DEPTH_ASYNC;
    private volatile boolean singleShotRequested = false;
    private volatile boolean singleShotRunning = false;

    // ---------------------------------------------------------------------------------------------
    //  Calibration & prefs
    // ---------------------------------------------------------------------------------------------
    private SharedPreferences prefs;
    private String calibrationPrefKey;
    private float calibrationScale = 1f;

    // ---------------------------------------------------------------------------------------------
    //  Zoom & camera facing
    // ---------------------------------------------------------------------------------------------
    private volatile int lensFacing = CameraSelector.LENS_FACING_BACK;
    private volatile float zoomMinRatio = 1f;
    private volatile float zoomMaxRatio = 1f;
    private boolean stereoSwitchInternalChange = false;
    private boolean envSwitchInternalChange = false;

    // class-level field (member variable), here it is for warning user system
    private java.util.List<String> cachedLabels = java.util.Collections.emptyList();
    private TTSWarning tts;

    // ---------------------------------------------------------------------------------------------
    //  Lifecycle & entry point
    // ---------------------------------------------------------------------------------------------

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        tts = TTSWarning.getInstance(this);

        depthModelPrefs = getSharedPreferences(DEPTH_MODEL_PREFS, MODE_PRIVATE);
        // Single-thread CameraX analyzer
        exec = Executors.newSingleThreadExecutor();
        // Three-thread inference pool: OD + seg + depth
        inferenceExec = Executors.newFixedThreadPool(3);

        initViews();
        initPreferencesAndCalibrationKey();

        initControls();

        // camera permission
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
                != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this,
                    new String[]{Manifest.permission.CAMERA}, REQ);
        } else {
            startPipelines();
        }

        // Lắng nghe thay đổi từ Settings/DepthEstimation
        prefs.registerOnSharedPreferenceChangeListener(prefListener);
    }

    // react to user choice permission
    @Override
    public void onRequestPermissionsResult(int c, @NonNull String[] p, @NonNull int[] r) {
        super.onRequestPermissionsResult(c, p, r);
        if (c == REQ && r.length > 0 && r[0] == PackageManager.PERMISSION_GRANTED) {
            startPipelines();
        }
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (exec != null) exec.shutdownNow();
        if (inferenceExec != null) inferenceExec.shutdownNow();
        if (detectorOd != null) {
            try {
                detectorOd.close();
            } catch (Exception e) {
                Log.e(TAG, "OD detector close failed", e);
            }
        }
        if (detectorSeg != null) {
            try {
                detectorSeg.close();
            } catch (Exception e) {
                Log.e(TAG, "Seg detector close failed", e);
            }
        }
        if (depthEstimator != null) {
            try {
                depthEstimator.close();
            } catch (Exception e) {
                Log.e(TAG, "DepthEstimator close failed", e);
            }
        }
        if (prefs != null) prefs.unregisterOnSharedPreferenceChangeListener(prefListener);
        stereoProcessor = null;
        // Clear depth cache state
        depthState.lastDepthMap = null;
        depthState.lastDepthMillis = 0L;
        depthState.lastDepthCacheTime = 0L;
        if (tts != null) {
            tts.shutdown();
        }
    }

    // Listener để reload khi Settings thay đổi
    private final SharedPreferences.OnSharedPreferenceChangeListener prefListener = (sharedPreferences, key) -> {
        if (PREF_ENV_MODE.equals(key)) {
            String modeName = sharedPreferences.getString(key, EnvMode.OUTDOOR.name());
            EnvMode newMode = EnvMode.valueOf(modeName);
            if (newMode != envMode) {
                envMode = newMode;
                runOnUiThread(() -> {
                    if (environmentSwitch != null) environmentSwitch.setChecked(envMode == EnvMode.OUTDOOR);
                    reloadPipelinesForEnvChange();
                    updateDepthModeLabel();
                });
            }
        } else if (PREF_BLUR_ENABLED.equals(key)) {
            blurEnabled = sharedPreferences.getBoolean(key, ENABLE_INPUT_BLUR);
            runOnUiThread(() -> {
                if (blurSwitch != null) blurSwitch.setChecked(blurEnabled);
                updateDepthModeLabel();
            });
        } else if (PREF_DEPTH_ASYNC.equals(key)) {
            depthAsyncEnabled = sharedPreferences.getBoolean(key, DEFAULT_DEPTH_ASYNC);
            runOnUiThread(this::updateDepthModeLabel);
        } else if (PREF_DEPTH_MODE.equals(key)) {
            String mode = sharedPreferences.getString(key, "MONO");
            boolean isStereo = "STEREO".equals(mode);
            // Chỉ cho phép bật stereo nếu phần cứng hỗ trợ
            if (isStereo && !stereoPipelineAvailable) {
                isStereo = false; // Fallback
            }
            stereoFusionEnabled = isStereo;
            runOnUiThread(() -> {
                if (stereoSwitch != null) stereoSwitch.setChecked(stereoFusionEnabled);
                updateDepthModeLabel();
            });
        }
    };

    // ---------------------------------------------------------------------------------------------
    //  UI init & listeners
    // ---------------------------------------------------------------------------------------------
    private void initViews() {
        previewView = findViewById(R.id.previewView);
        overlay = findViewById(R.id.overlay);
        realtimeSwitch = findViewById(R.id.switchRealtime);
        blurSwitch = findViewById(R.id.switchBlur);
        stereoSwitch = findViewById(R.id.switchStereo);
        detectOnceButton = findViewById(R.id.buttonDetectOnce);
        quickSettingsButton = findViewById(R.id.buttonToggleSettings);
        settingsButton = findViewById(R.id.buttonSettings);
        controlPanel = findViewById(R.id.controlPanel);
        depthModeText = findViewById(R.id.textDepthMode);
        calibrationSeek = findViewById(R.id.seekCalibration);
        calibrationValue = findViewById(R.id.textCalibrationValue);
        environmentSwitch = findViewById(R.id.switchEnvironment);

        // labels for OD + segmentation
        String[] odLabelsArr = LabelHelper.loadLabels(this, "labels_od.txt");
        if (odLabelsArr.length == 0) {
            odLabelsArr = LabelHelper.loadLabels(this, "labels.txt");
        }
        String[] segLabelsArr = LabelHelper.loadLabels(this, "labels_seg.txt");
        cachedLabels = java.util.Arrays.asList(odLabelsArr);
        overlay.setLabels(odLabelsArr, segLabelsArr);
    }

    private void initPreferencesAndCalibrationKey() {
        prefs = DepthCalibrationHelper.getPrefs(this);

        // Calibration
        calibrationPrefKey = DepthCalibrationHelper.buildCalibrationKey(this);
        calibrationScale = 1f;
        DepthCalibrationHelper.saveCalibration(prefs, calibrationPrefKey, calibrationScale);
        DepthEstimator.setUserScale(calibrationScale);

        // Environment mode (load from prefs, default = OUTDOOR)
        String savedEnv = prefs.getString(PREF_ENV_MODE, EnvMode.OUTDOOR.name());
        try {
            envMode = EnvMode.valueOf(savedEnv);
        } catch (IllegalArgumentException e) {
            envMode = EnvMode.OUTDOOR;
        }

        // Sync với UI switch (ON = OUTDOOR, OFF = INDOOR)
        if (environmentSwitch != null) {
            environmentSwitch.setChecked(envMode == EnvMode.OUTDOOR);
        }

        // Sync Blur
        blurEnabled = prefs.getBoolean(PREF_BLUR_ENABLED, ENABLE_INPUT_BLUR);
        depthAsyncEnabled = prefs.getBoolean(PREF_DEPTH_ASYNC, DEFAULT_DEPTH_ASYNC);
    }


    private void initControls() {
        initRealtimeSwitch();
        initDetectOnceButton();
        initBlurSwitch();
        initStereoSwitch();
        initEnvironmentSwitch();
        initQuickSettingsButton();
        initSettingsButton();
        setupCalibrationControls();
        updateDepthModeLabel();
        setupZoomControls();
    }

    private void initSettingsButton() {
        if (settingsButton == null) return;
        settingsButton.setOnClickListener(v -> {
            Intent intent = new Intent(MainActivity.this, Settings.class);
            // Truyền khả năng hỗ trợ Stereo sang Settings để Settings truyền tiếp cho DepthEstimation
            intent.putExtra("STEREO_AVAILABLE", stereoPipelineAvailable);
            startActivity(intent);
        });
    }

    private void initRealtimeSwitch() {
        if (realtimeSwitch == null) return;
        realtimeSwitch.setChecked(true);

        realtimeSwitch.setThumbTintList(ContextCompat.getColorStateList(this, R.color.switch_thumb1));
        realtimeSwitch.setTrackTintList(ContextCompat.getColorStateList(this, R.color.switch_track1));

        realtimeSwitch.setOnCheckedChangeListener((buttonView, isChecked) -> {
            realtimeEnabled = isChecked;
            if (detectOnceButton != null) {
                detectOnceButton.setVisibility(isChecked ? View.GONE : View.VISIBLE);
                detectOnceButton.setEnabled(true);
            }
            if (isChecked) {
                singleShotRequested = false;
            }
        });
    }

    private void initDetectOnceButton() {
        if (detectOnceButton == null) return;
        detectOnceButton.setVisibility(View.GONE);
        detectOnceButton.setOnClickListener(v -> {
            if (singleShotRunning) return;
            singleShotRequested = true;
            detectOnceButton.setEnabled(false);
        });
    }

    private void initBlurSwitch() {
        if (blurSwitch == null) return;
        // Ẩn Blur switch khỏi Quick Settings theo yêu cầu
        blurSwitch.setVisibility(View.GONE);
    }

    private void initStereoSwitch() {
        if (stereoSwitch == null) return;
        // Ẩn Stereo switch, chỉ hiển thị trạng thái qua text
        stereoSwitch.setVisibility(View.GONE);
    }

    private void initEnvironmentSwitch() {
        if (environmentSwitch == null) return;
        // Ẩn Environment switch, chỉ hiển thị trạng thái qua text
        environmentSwitch.setVisibility(View.GONE);
        // Vẫn sync trạng thái checked để logic nội bộ (nếu có dùng) không bị sai
        environmentSwitch.setChecked(envMode == EnvMode.OUTDOOR);
    }

    private void switchEnvironment(EnvMode newMode) {
        envMode = newMode;

        // Lưu vào prefs
        prefs.edit().putString(PREF_ENV_MODE, envMode.name()).apply();

        Toast.makeText(
                this,
                "Environment: " + (envMode == EnvMode.OUTDOOR ? "Outdoor" : "Indoor"),
                Toast.LENGTH_SHORT
        ).show();

        // Reload lại detector/depth cho mode mới
        reloadPipelinesForEnvChange();
    }


    private void reloadPipelinesForEnvChange() {
        // Pause realtime so we don't process frames while reloading depth
        realtimeEnabled = false;

        runOnUiThread(() -> {
            Log.i(TAG, "Reloading depth pipeline for envMode = " + envMode);

            // 1) Check if we have ANY model for this mode (asset or downloaded)
            boolean depthModelOk = DepthEstimator.isModelAvailable(this, envMode);
            if (!depthModelOk) {
                // No model yet -> show download dialog for this mode
                showMissingDepthModelDialog(envMode);

                // Keep depthEstimator = null, YOLO-only mode
                depthEstimator = null;
                synchronized (depthState) {
                    depthState.lastDepthMap = null;
                    depthState.lastDepthMillis = 0L;
                    depthState.lastDepthCacheTime = 0L;
                }

                // Re-enable realtime (but without depth)
                realtimeEnabled = true;
                return;
            }

            // 2) We DO have a model (asset or downloaded) -> try to create DepthEstimator
            try {
                DepthEstimator newDepth = new DepthEstimator(this, envMode);
                depthEstimator = newDepth;

                synchronized (depthState) {
                    depthState.lastDepthMap = null;
                    depthState.lastDepthMillis = 0L;
                    depthState.lastDepthCacheTime = 0L;
                }

                Toast.makeText(
                        this,
                        "Depth model loaded for " +
                                (envMode == EnvMode.OUTDOOR ? "Outdoor" : "Indoor"),
                        Toast.LENGTH_SHORT
                ).show();

            } catch (Throwable e) {
                Log.w(TAG, "Depth estimator re-init failed", e);
                depthEstimator = null;
                synchronized (depthState) {
                    depthState.lastDepthMap = null;
                    depthState.lastDepthMillis = 0L;
                    depthState.lastDepthCacheTime = 0L;
                }
                Toast.makeText(
                        this,
                        "Failed to init depth for " +
                                (envMode == EnvMode.OUTDOOR ? "Outdoor" : "Indoor"),
                        Toast.LENGTH_LONG
                ).show();
            }

            // Re-enable realtime (with or without depth depending on success)
            realtimeEnabled = true;
        });
    }



    private void initQuickSettingsButton() {
        if (quickSettingsButton == null) {
            applySettingsVisibility(true);
            return;
        }
        quickSettingsButton.setOnClickListener(v -> toggleSettingsPanel());
        applySettingsVisibility(false);
    }

    private void toggleSettingsPanel() {
        boolean visible = controlPanel != null
                && controlPanel.getVisibility() != View.VISIBLE;
        applySettingsVisibility(visible);
    }


    private void applySettingsVisibility(boolean visible) {
        if (controlPanel != null) {
            controlPanel.setVisibility(visible ? View.VISIBLE : View.GONE);
        }
        if (quickSettingsButton != null) {
            quickSettingsButton.setImageResource(
                    visible ? R.drawable.ic_closetab
                            : R.drawable.ic_quickset
            );
            quickSettingsButton.setContentDescription(
                    getString(visible ? R.string.settings_hide : R.string.settings_show)
            );
        }
    }

    // ---------------------------------------------------------------------------------------------
    //  Pipelines startup (detector + depth + camera)
    // ---------------------------------------------------------------------------------------------
    private void startPipelines() {
        initDetectorAndDepth();
        initCameraProvider();
    }

    private void initDetectorAndDepth() {
        try {
            detectorOd = new ObjectDetector(this, "best_OD_nano.onnx", ObjectDetector.Detection.SOURCE_OD);
        } catch (Throwable e) {
            detectorOd = null;
            Log.w(TAG, "OD detector init failed (best_OD_nano.onnx), trying fallback", e);
            try {
                detectorOd = new ObjectDetector(this, "best.onnx",
                        ObjectDetector.Detection.SOURCE_OD);
            } catch (Throwable secondErr) {
                detectorOd = null;
                Log.w(TAG, "OD detector init failed (best.onnx), trying fallback 2", secondErr);
                try {
                    detectorOd = new ObjectDetector(this, "yolov8m_compatible.onnx",
                        ObjectDetector.Detection.SOURCE_OD);
                } catch (Throwable fallbackErr) {
                    detectorOd = null;
                    Log.e(TAG, "OD detector init failed (fallback)", fallbackErr);
                }
            }
        }
        try {
            detectorSeg = new ObjectDetector(this, "best_seg_nano.onnx", ObjectDetector.Detection.SOURCE_SEG);
        } catch (Throwable e) {
            detectorSeg = null;
            Log.w(TAG, "Seg detector init failed (best_seg_nano.onnx), trying fallback", e);
            try {
                detectorSeg = new ObjectDetector(this, "bestseg.onnx", ObjectDetector.Detection.SOURCE_SEG);
            } catch (Throwable fallbackErr) {
                detectorSeg = null;
                Log.e(TAG, "Seg detector init failed (fallback)", fallbackErr);
            }
        }
        if (detectorOd == null && detectorSeg == null) {
            Toast.makeText(this, "Detector load failed: no model available",
                    Toast.LENGTH_LONG).show();
            return;
        }

        // --- NEW: kiểm tra depth model có trong assets hay chưa ---
        boolean depthModelOk = DepthEstimator.isModelAvailable(this, envMode);
        if (!depthModelOk) {
            // Không có model -> thông báo & gợi ý mở link download
            showMissingDepthModelDialog(envMode);
            // Không tạo depthEstimator, app vẫn chạy YOLO-only
            depthEstimator = null;
            depthState.lastDepthMap = null;
            depthState.lastDepthMillis = 0L;
            depthState.lastDepthCacheTime = 0L;
            stereoProcessor = null;
            updateStereoSwitchAvailability(false);
            return;
        }

        try {
            depthEstimator = new DepthEstimator(this, envMode);
            depthState.lastDepthMap = null;
            depthState.lastDepthMillis = 0L;
            depthState.lastDepthCacheTime = 0L;
        } catch (Throwable e) {
            Log.w(TAG, "Depth estimator disabled", e);
            depthEstimator = null;
            depthState.lastDepthMap = null;
            depthState.lastDepthMillis = 0L;
            depthState.lastDepthCacheTime = 0L;
        }
        stereoProcessor = null;
        updateStereoSwitchAvailability(false);
    }

    private void showMissingDepthModelDialog(EnvMode targetMode) {
        String modeLabel = (targetMode == EnvMode.OUTDOOR) ? "Outdoor" : "Indoor";

        new AlertDialog.Builder(this)
                .setTitle("Depth model missing")
                .setMessage(
                        "Depth model for " + modeLabel + " mode is not available inside the app.\n\n" +
                                "Please go to Settings > Model Package to download it."
                )
                .setPositiveButton("OK", null)
                .setCancelable(true)
                .show();
    }


    private void initCameraProvider() {
        ProcessCameraProvider.getInstance(this).addListener(() -> {
            try {
                ProcessCameraProvider provider =
                        ProcessCameraProvider.getInstance(this).get();
                cameraProvider = provider;
                provider.unbindAll();
                backCameraInfos = CameraUtils.cacheBackCameraInfos(provider);
                bindCameraUseCases();
            } catch (Throwable e) {
                Log.e(TAG, "Camera bind error", e);
                Toast.makeText(this, "Camera error: " + e.getMessage(),
                        Toast.LENGTH_LONG).show();
            }
        }, ContextCompat.getMainExecutor(this));
    }

    // ---------------------------------------------------------------------------------------------
    //  Camera pipeline (realtime analysis)
    // ---------------------------------------------------------------------------------------------
    private void bindCameraUseCases() {
        if (cameraProvider == null) return;
        try {
            cameraProvider.unbindAll();

            Preview preview = new Preview.Builder()
                    .setResolutionSelector(
                            new ResolutionSelector.Builder()
                                    .setAspectRatioStrategy(
                                            AspectRatioStrategy.RATIO_4_3_FALLBACK_AUTO_STRATEGY
                                    )
                                    .build()
                    )
                    .build();
            preview.setSurfaceProvider(previewView.getSurfaceProvider());

            ImageAnalysis analysis = new ImageAnalysis.Builder()
                    .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                    .setResolutionSelector(
                            new ResolutionSelector.Builder()
                                    .setResolutionStrategy(
                                            new ResolutionStrategy(
                                                    new Size(ANALYSIS_INPUT_SIZE, ANALYSIS_INPUT_SIZE),
                                                    ResolutionStrategy
                                                            .FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER
                                            )
                                    )
                                    .build()
                    )
                    .build();

            analysis.setAnalyzer(exec, this::analyzeFrame);

            CameraSelector selector =
                    CameraUtils.buildBackSelector(cameraProvider, lensFacing);

            Camera camera = cameraProvider.bindToLifecycle(
                    (LifecycleOwner) this, selector, preview, analysis);
            currentCamera = camera;

            observeZoom(camera);
            setupStereoProcessorForCurrentCamera(camera);

        } catch (Throwable e) {
            Log.e(TAG, "Camera bind error", e);
            Toast.makeText(this, "Camera error: " + e.getMessage(),
                    Toast.LENGTH_LONG).show();
        }
    }

    /**
     * Runs depth estimation if the interval has passed, otherwise returns cached depth.
     * Thread-safe; can be called from background threads.
     */
    private DepthEstimator.DepthMap maybeRunDepthSync(
            int[] argb,
            int width,
            int height,
            long nowMs
    ) {
        if (depthEstimator == null) return null;

        synchronized (depthState) {
            boolean hasDepth = (depthState.lastDepthMap != null);
            boolean tooSoon = (nowMs - depthState.lastDepthMillis) < DEPTH_INTERVAL_MS;
            boolean cacheValid = hasDepth &&
                    (nowMs - depthState.lastDepthCacheTime) <= DEPTH_CACHE_MS;

            if (tooSoon && cacheValid) {
                depthState.lastDepthCacheTime = nowMs;
                return depthState.lastDepthMap;
            }
        }

        try {
            DepthEstimator.DepthMap map =
                    depthEstimator.estimate(argb, width, height);
            long doneMs = SystemClock.elapsedRealtime();

            synchronized (depthState) {
                depthState.lastDepthMap = map;
                depthState.lastDepthMillis = doneMs;
                depthState.lastDepthCacheTime = doneMs;
            }
            return map;
        } catch (Exception e) {
            Log.e(TAG, "Depth estimation failed", e);
            return null;
        }
    }

    private DepthEstimator.DepthMap getCachedDepthMap(long nowMs) {
        synchronized (depthState) {
            if (depthState.lastDepthMap == null) return null;
            if ((nowMs - depthState.lastDepthCacheTime) > DEPTH_CACHE_MS) return null;
            return depthState.lastDepthMap;
        }
    }

    private void analyzeFrame(ImageProxy image) {
        boolean singleShotFrame = false;

        try {
            boolean shouldProcess = realtimeEnabled;

            if (!shouldProcess && singleShotRequested && !singleShotRunning) {
                singleShotRequested = false;
                singleShotRunning = true;
                singleShotFrame = true;
                shouldProcess = true;

                if (stereoFusionEnabled && !stereoPipelineAvailable) {
                    singleShotFrame = false;
                    handleSequentialDualShot();
                    return;
                }
            }

            if (!shouldProcess) return;

            // ---- Capture timestamp (convert to nanoTime base) ----
            long imgTsNs = image.getImageInfo().getTimestamp();   // camera timestamp
            long imgTsUptimeNs = cameraTsIsRealtime
                    ? (imgTsNs - realtimeMinusUptimeOffsetNs)
                    : imgTsNs;

            // Start of YOUR processing for this frame
            long analyzerStartNs = System.nanoTime();
            long captureToAnalyzerNs = analyzerStartNs - imgTsUptimeNs;
            // -----------------------------------------------------

            // Basic frame info
            int frameW = image.getWidth();
            int frameH = image.getHeight();
            int rotation = image.getImageInfo().getRotationDegrees();

            // YUV → ARGB (+ rotation)
            int[] argb = Yuv.toArgb(image);
            if (rotation != 0) {
                argb = Yuv.rotate(argb, frameW, frameH, rotation);
                if (rotation == 90 || rotation == 270) {
                    int tmp = frameW;
                    frameW = frameH;
                    frameH = tmp;
                }
            }

            if (stereoProcessor != null) {
                stereoProcessor.setReferenceSize(frameW, frameH);
            }

            int[] detectorInput = (blurEnabled && BLUR_RADIUS > 0)
                    ? ImageUtils.blurAtSize(argb, frameW, frameH, BLUR_INPUT_SIZE, BLUR_RADIUS)
                    : argb;

            // Run YOLO + depth in parallel on inferenceExec
            int finalFrameW1 = frameW;
            int finalFrameH1 = frameH;

            Future<List<ObjectDetector.Detection>> detFutureOd = null;
            Future<List<ObjectDetector.Detection>> detFutureSeg = null;
            if (detectorOd != null) {
                detFutureOd = inferenceExec.submit(() -> {
                    try {
                        return detectorOd.detect(detectorInput, finalFrameW1, finalFrameH1);
                    } catch (OrtException e) {
                        Log.e(TAG, "OD detect failed", e);
                        return null;
                    } catch (Throwable t) {
                        Log.e(TAG, "OD detect crashed", t);
                        return null;
                    }
                });
            }
            if (detectorSeg != null) {
                detFutureSeg = inferenceExec.submit(() -> {
                    try {
                        return detectorSeg.detect(detectorInput, finalFrameW1, finalFrameH1);
                    } catch (OrtException e) {
                        Log.e(TAG, "Seg detect failed", e);
                        return null;
                    } catch (Throwable t) {
                        Log.e(TAG, "Seg detect crashed", t);
                        return null;
                    }
                });
            }

            Future<DepthEstimator.DepthMap> depthFuture = null;
            if (depthEstimator != null) {
                if (depthAsyncEnabled) {
                    if (depthBusy.compareAndSet(false, true)) {
                        int[] finalArgb = argb;
                        int finalFrameW = frameW;
                        int finalFrameH = frameH;
                        inferenceExec.submit(() -> {
                            try {
                                maybeRunDepthSync(finalArgb, finalFrameW, finalFrameH,
                                        SystemClock.elapsedRealtime());
                            } finally {
                                depthBusy.set(false);
                            }
                        });
                    }
                } else {
                    int[] finalArgb = argb;
                    int finalFrameW = frameW;
                    int finalFrameH = frameH;
                    depthFuture = inferenceExec.submit(() ->
                            maybeRunDepthSync(finalArgb, finalFrameW, finalFrameH,
                                    SystemClock.elapsedRealtime())
                    );
                }
            }

            // Wait for results
            List<ObjectDetector.Detection> dets = new ArrayList<>();
            if (detFutureOd != null) {
                List<ObjectDetector.Detection> od = detFutureOd.get();
                if (od != null && !od.isEmpty()) dets.addAll(od);
            }
            if (detFutureSeg != null) {
                List<ObjectDetector.Detection> seg = detFutureSeg.get();
                if (seg != null && !seg.isEmpty()) dets.addAll(seg);
            }
            if (dets.isEmpty()) dets = null;

            DepthEstimator.DepthMap depthMap = null;
            if (depthEstimator != null) {
                if (depthAsyncEnabled) {
                    depthMap = getCachedDepthMap(SystemClock.elapsedRealtime());
                } else if (depthFuture != null) {
                    depthMap = depthFuture.get();
                }
            }

            if (depthMap != null && dets != null) {
                dets = depthEstimator.attachDepth(dets, depthMap);
            }

            if (stereoFusionEnabled && stereoProcessor != null
                    && depthMap != null && dets != null) {
                dets = stereoProcessor.fuseDepth(depthMap, dets, frameW, frameH);
            }

            // End of your compute work
            long inferenceDoneNs = System.nanoTime();
            long processingNs = inferenceDoneNs - analyzerStartNs;

            int finalW = frameW;
            int finalH = frameH;
            List<ObjectDetector.Detection> finalDets = dets;

            long imgTsUptimeNsFinal = imgTsUptimeNs;
            long captureToAnalyzerNsFinal = captureToAnalyzerNs;
            long processingNsFinal = processingNs;

            runOnUiThread(() -> {
                long uiCallbackStartNs = System.nanoTime();
                long captureToUiCallbackNs = uiCallbackStartNs - imgTsUptimeNsFinal;

                overlay.setDetections(finalDets, finalW, finalH);

                // This logs capture->TTS(begin) at the moment we START calling TTS
                processTTSWarning(finalDets,
                        finalW,
                        imgTsUptimeNsFinal,
                        captureToAnalyzerNsFinal,
                        processingNsFinal,
                        captureToUiCallbackNs
                );


                // Optional: capture -> next UI frame start (vsync)
                overlay.postInvalidateOnAnimation();
                if ((++latFrameCounter % LAT_LOG_EVERY_N_FRAMES) == 0) {
                    android.view.Choreographer.getInstance().postFrameCallback(frameTimeNanos -> {
                        long capToUiFrameNs = frameTimeNanos - imgTsUptimeNsFinal;
                        Log.i(TAG, String.format(
                                "Latency(ms): cap->UIframe=%.2f",
                                capToUiFrameNs / 1e6
                        ));
                    });
                }
            });

        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            Log.w(TAG, "analyzeFrame interrupted", e);
        } catch (Throwable t) {
            Log.e(TAG, "analyzer crash", t);
        } finally {
            image.close();
            if (singleShotFrame) {
                singleShotRunning = false;
                runOnUiThread(() -> {
                    if (detectOnceButton != null) detectOnceButton.setEnabled(true);
                });
            }
        }
    }

    @SuppressLint("RestrictedApi")
    @OptIn(markerClass = ExperimentalCamera2Interop.class)
    private void setupStereoProcessorForCurrentCamera(Camera camera) {
        try {
            CameraCharacteristics cc =
                    Camera2CameraInfo.extractCameraCharacteristics(camera.getCameraInfo());

            // ---- REQUIRED for correct capture timestamp conversion ----
            Integer src = cc.get(CameraCharacteristics.SENSOR_INFO_TIMESTAMP_SOURCE);
            cameraTsIsRealtime = (src != null
                    && src == CameraCharacteristics.SENSOR_INFO_TIMESTAMP_SOURCE_REALTIME);

            // Convert REALTIME (elapsedRealtimeNanos) -> nanoTime base
            realtimeMinusUptimeOffsetNs =
                    SystemClock.elapsedRealtimeNanos() - System.nanoTime();
            // ----------------------------------------------------------

            boolean isLogicalMultiCamera = false;
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
                Set<String> physicalIds = cc.getPhysicalCameraIds();
                if (physicalIds != null && physicalIds.size() >= 2) {
                    isLogicalMultiCamera = true;
                }
            }

            if (lensFacing == CameraSelector.LENS_FACING_BACK && isLogicalMultiCamera) {
                stereoProcessor = new StereoDepthProcessor(this, cc);
                updateStereoSwitchAvailability(true);
            } else {
                stereoProcessor = null;
                updateStereoSwitchAvailability(false);
            }
        } catch (Throwable processorErr) {
            Log.w(TAG, "Stereo processor init failed", processorErr);
            stereoProcessor = null;
            updateStereoSwitchAvailability(false);
        }
    }


    // ---------------------------------------------------------------------------------------------
    //  Stereo single-shot pipeline
    // ---------------------------------------------------------------------------------------------
    private void handleSequentialDualShot() {
        if (cameraProvider == null) {
            singleShotRunning = false;
            return;
        }
        if (lensFacing != CameraSelector.LENS_FACING_BACK) {
            sequentialStereoRunning = false;
            singleShotRunning = false;
            runOnUiThread(() -> {
                if (detectOnceButton != null) detectOnceButton.setEnabled(true);
                Toast.makeText(this, R.string.stereo_toggle_disabled_hint,
                        Toast.LENGTH_SHORT).show();
            });
            return;
        }
        if (sequentialStereoRunning) return;

        singleShotRunning = true;
        sequentialStereoRunning = true;

        runOnUiThread(() ->
                Toast.makeText(this, R.string.sequential_dual_shot,
                        Toast.LENGTH_SHORT).show());

        if (backCameraInfos == null || backCameraInfos.isEmpty()) {
            backCameraInfos = CameraUtils.cacheBackCameraInfos(cameraProvider);
        }

        SequentialStereoHelper.runSequentialStereoShot(
                this,
                cameraProvider,
                previewView,
                exec,
                backCameraInfos,
                detectorOd,
                detectorSeg,
                depthEstimator,
                blurEnabled,
                BLUR_RADIUS,
                stereoFusionEnabled,
                stereoProcessor,
                new SequentialStereoHelper.Callback() {
                    @Override
                    public void onResult(SequentialStereoHelper.Result result) {
                        if (result == null || result.detections == null) return;
                        overlay.setDetections(
                                result.detections,
                                result.width,
                                result.height
                        );
                    }

                    @Override
                    public void onError(Throwable t) {
                        Log.e(TAG, "Sequential stereo single shot failed", t);
                    }

                    @Override
                    public void onFinished() {
                        sequentialStereoRunning = false;
                        singleShotRunning = false;
                        if (detectOnceButton != null) {
                            detectOnceButton.setEnabled(true);
                        }
                        bindCameraUseCases();
                    }
                }
        );
    }

    // ---------------------------------------------------------------------------------------------
    //  Depth calibration & UI
    // ---------------------------------------------------------------------------------------------
    private void setupCalibrationControls() {
        if (calibrationSeek != null) {
            calibrationSeek.setThumbTintList(ContextCompat.getColorStateList(this, R.color.switch_thumb1));
            calibrationSeek.setProgressTintList(ContextCompat.getColorStateList(this, R.color.switch_track1));
        }
        DepthCalibrationHelper.bindCalibrationSeekBar(
                calibrationSeek,
                calibrationValue,
                calibrationScale,
                prefs,
                calibrationPrefKey
        );
    }

    private void updateDepthModeLabel() {
        if (depthModeText == null) return;
        boolean stereoActive = stereoFusionEnabled
                && stereoPipelineAvailable
                && stereoProcessor != null;
        String modeStr = getString(stereoActive ? R.string.depth_mode_stereo : R.string.depth_mode_mono);
        String envStr = (envMode == EnvMode.OUTDOOR) ? "Outdoor" : "Indoor";
        String blurStr = blurEnabled ? "Blur: On" : "Blur: Off";
        String depthAsyncStr = depthAsyncEnabled ? "Depth Async: On" : "Depth Async: Off";
        // Hiển thị kết hợp: "Mono • Indoor • Blur: On • Depth Async: On"
        depthModeText.setText(String.format("%s • %s • %s • %s",
                modeStr, envStr, blurStr, depthAsyncStr));
    }

    private void updateStereoSwitchAvailability(boolean available) {
        stereoPipelineAvailable = available;
        if (stereoSwitch == null) return;
        runOnUiThread(() -> {
            stereoSwitchInternalChange = true;
            if (!available) {
                stereoSwitch.setChecked(false);
                stereoSwitch.setEnabled(false);
                stereoSwitch.setText(R.string.stereo_toggle_disabled_hint);
                stereoFusionEnabled = false;
            } else {
                stereoSwitch.setText(R.string.stereo_toggle);
                stereoSwitch.setEnabled(true);
                
                // Đồng bộ trạng thái từ Prefs khi Stereo khả dụng (để khi mở app nó nhớ trạng thái cũ)
                String savedMode = prefs.getString(PREF_DEPTH_MODE, "MONO");
                stereoFusionEnabled = "STEREO".equals(savedMode);
                stereoSwitch.setChecked(stereoFusionEnabled);
            }
            stereoSwitchInternalChange = false;
            updateDepthModeLabel();
        });
    }

    // ---------------------------------------------------------------------------------------------
    //  Zoom & camera utils
    // ---------------------------------------------------------------------------------------------
    private void setupZoomControls() {
        if (zoomSeek == null || zoomValue == null) return;
        zoomSeek.setThumbTintList(ContextCompat.getColorStateList(this, R.color.switch_thumb1));
        zoomSeek.setProgressTintList(ContextCompat.getColorStateList(this, R.color.switch_track1));
        ZoomHelper.setupZoomSeekBar(
                zoomSeek,
                zoomValue,
                new ZoomHelper.ZoomControl() {
                    @Override
                    public float getMinZoomRatio() {
                        return zoomMinRatio;
                    }

                    @Override
                    public float getMaxZoomRatio() {
                        return zoomMaxRatio;
                    }

                    @Override
                    public void setZoomRatio(float ratio) {
                        if (currentCamera != null) {
                            currentCamera.getCameraControl().setZoomRatio(ratio);
                        }
                    }
                }
        );
    }

    private void observeZoom(Camera camera) {
        if (zoomSeek == null || zoomValue == null) return;
        camera.getCameraInfo().getZoomState().observe(this, state -> {
            if (state == null) return;
            zoomMinRatio = state.getMinZoomRatio();
            zoomMaxRatio = state.getMaxZoomRatio();
            int progress = ZoomHelper.zoomRatioToProgress(
                    state.getZoomRatio(),
                    zoomMinRatio,
                    zoomMaxRatio
            );
            zoomSeek.setProgress(progress);
            zoomValue.setText(getString(R.string.zoom_value, state.getZoomRatio()));
        });
    }

    private void switchCameraFacing() {
        lensFacing = (lensFacing == CameraSelector.LENS_FACING_BACK)
                ? CameraSelector.LENS_FACING_FRONT
                : CameraSelector.LENS_FACING_BACK;
        updateStereoSwitchAvailability(false);
        bindCameraUseCases();
        updateDepthModeLabel();
    }

    //process TTS warning
    private void processTTSWarning(
            java.util.List<ObjectDetector.Detection> results,
            int frameW,
            long imgTsUptimeNs,
            long captureToAnalyzerNs,
            long processingNs,
            long captureToUiCallbackNs
    ) {
        if (results == null || results.isEmpty() || tts == null) return;
        if (cachedLabels == null || cachedLabels.isEmpty()) return;

        java.util.List<vn.edu.usth.objectdetectmobile.utils.TTSWarning.Detection> ttsDetections =
                new java.util.ArrayList<>();

        final float invW = 1.0f / Math.max(1, frameW);

        for (ObjectDetector.Detection det : results) {
            if (det.source != ObjectDetector.Detection.SOURCE_OD) continue;
            if (Float.isNaN(det.depth) || det.depth <= 0) continue;

            // Your pipeline: det.depth printed as "cm" in OverlayView -> convert to meters
            float distanceMeters = det.depth / 100.0f;

            String label = (det.cls >= 0 && det.cls < cachedLabels.size())
                    ? cachedLabels.get(det.cls)
                    : "object";

            float xCenter = (det.x1 + det.x2) * 0.5f;

            float xCenterNorm = clamp01(xCenter * invW);

            ttsDetections.add(new vn.edu.usth.objectdetectmobile.utils.TTSWarning.Detection(
                    label, distanceMeters, xCenterNorm
            ));
        }

        if (ttsDetections.isEmpty()) return;

        // latency log (unchanged)
        long ttsBeginNs = System.nanoTime();
        long captureToTtsBeginNs = ttsBeginNs - imgTsUptimeNs;
        android.util.Log.i("LAT", String.format(
                java.util.Locale.US,
                "Latency(ms): cap->analyzer=%.2f, processing=%.2f, cap->UIcb=%.2f, cap->TTSbegin=%.2f (ttsDets=%d)",
                captureToAnalyzerNs / 1e6,
                processingNs / 1e6,
                captureToUiCallbackNs / 1e6,
                captureToTtsBeginNs / 1e6,
                ttsDetections.size()
        ));

        tts.processDetections(ttsDetections);
    }

    private static float clamp01(float v) {
        return Math.max(0f, Math.min(1f, v));
    }
}
