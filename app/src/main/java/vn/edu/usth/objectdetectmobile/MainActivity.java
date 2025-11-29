package vn.edu.usth.objectdetectmobile;

import android.Manifest;
import android.content.SharedPreferences;
import android.content.pm.PackageManager;
import android.hardware.camera2.CameraCharacteristics;
import android.hardware.camera2.CameraManager;
import android.os.Build;
import android.os.Bundle;
import android.os.SystemClock;
import android.util.Log;
import android.util.Size;
import android.view.View;
import android.widget.SeekBar;
import android.widget.TextView;
import android.widget.Toast;

import androidx.activity.ComponentActivity;
import androidx.annotation.NonNull;
import androidx.camera.camera2.interop.Camera2CameraInfo;
import androidx.camera.core.Camera;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageProxy;
import androidx.camera.core.Preview;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.camera.view.PreviewView;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

// NEW imports for the non-deprecated API
import androidx.camera.core.resolutionselector.AspectRatioStrategy;
import androidx.camera.core.resolutionselector.ResolutionSelector;
import androidx.camera.core.resolutionselector.ResolutionStrategy;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.CountDownLatch;

import ai.onnxruntime.OrtException;
import com.google.android.material.button.MaterialButton;
import com.google.android.material.switchmaterial.SwitchMaterial;

public class MainActivity extends ComponentActivity {
    private static final int REQ = 42;
    private static final String TAG = "MainActivity";

    private static final long DEPTH_INTERVAL_MS = 1500L;
    private static final long DEPTH_CACHE_MS = 3000L;
    private static final boolean ENABLE_INPUT_BLUR = true;
    private static final int BLUR_RADIUS = 1; // 1 => kernel 3x3
    private static final String PREFS_NAME = "depth_calibration_prefs";
    private static final String PREF_KEY_PREFIX = "depth_calibration_scale_";
    private static final float CALIB_MIN = 0.5f;
    private static final float CALIB_MAX = 2.5f;
    private static final int CALIB_PROGRESS_MAX = 200;
    private static final int ZOOM_PROGRESS_MAX = 1000;

    private PreviewView previewView;
    private OverlayView overlay;
    private ObjectDetector detector;
    private DepthEstimator depthEstimator;
    private ExecutorService exec;
    private long lastDepthMillis = 0L;
    private DepthEstimator.DepthMap lastDepthMap = null;
    private long lastDepthCacheTime = 0L;
    private SwitchMaterial realtimeSwitch;
    private SwitchMaterial blurSwitch;
    private SwitchMaterial stereoSwitch;
    private MaterialButton detectOnceButton;
    private MaterialButton settingsButton;
    private MaterialButton dualShotButton;
    private View controlPanel;
    private TextView depthModeText;
    private volatile boolean realtimeEnabled = true;
    private volatile boolean blurEnabled = ENABLE_INPUT_BLUR;
    private volatile boolean stereoFusionEnabled = false;
    private boolean stereoPipelineAvailable = false;
    private volatile boolean singleShotRequested = false;
    private volatile boolean singleShotRunning = false;
    private StereoDepthProcessor stereoProcessor;
    private boolean stereoSwitchInternalChange = false;
    private SeekBar calibrationSeek;
    private TextView calibrationValue;
    private volatile boolean sequentialStereoRunning = false;
    private java.util.List<CamInfo> backCameraInfos = new java.util.ArrayList<>();
    private SharedPreferences prefs;
    private String calibrationPrefKey;
    private float calibrationScale = 1f;
    private ProcessCameraProvider cameraProvider;
    private Camera currentCamera;
    private volatile int lensFacing = CameraSelector.LENS_FACING_BACK;
    private SeekBar zoomSeek;
    private TextView zoomValue;
    private MaterialButton flipCameraButton;
    private volatile float zoomMinRatio = 1f;
    private volatile float zoomMaxRatio = 1f;

    @Override protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        previewView = findViewById(R.id.previewView);
        overlay = findViewById(R.id.overlay);
        overlay.setLabels(loadLabels());
        realtimeSwitch = findViewById(R.id.switchRealtime);
        blurSwitch = findViewById(R.id.switchBlur);
        stereoSwitch = findViewById(R.id.switchStereo);
        detectOnceButton = findViewById(R.id.buttonDetectOnce);
        dualShotButton = findViewById(R.id.buttonDualShot);
        settingsButton = findViewById(R.id.buttonToggleSettings);
        flipCameraButton = findViewById(R.id.buttonFlipCamera);
        controlPanel = findViewById(R.id.controlPanel);
        depthModeText = findViewById(R.id.textDepthMode);
        calibrationSeek = findViewById(R.id.seekCalibration);
        calibrationValue = findViewById(R.id.textCalibrationValue);
        zoomSeek = findViewById(R.id.seekZoom);
        zoomValue = findViewById(R.id.textZoomValue);

        prefs = getSharedPreferences(PREFS_NAME, MODE_PRIVATE);
        calibrationPrefKey = buildCalibrationKey();
        calibrationScale = loadSavedCalibrationScale();
        DepthEstimator.setUserScale(calibrationScale);

        if (realtimeSwitch != null) {
            realtimeSwitch.setChecked(true);
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

        if (detectOnceButton != null) {
            detectOnceButton.setVisibility(View.GONE);
            detectOnceButton.setOnClickListener(v -> {
                if (singleShotRunning) {
                    return;
                }
                        singleShotRequested = true;
                        detectOnceButton.setEnabled(false);
                    });
        }

        if (dualShotButton != null) {
            dualShotButton.setOnClickListener(v -> {
                if (sequentialStereoRunning) return;
                handleSequentialDualShot();
            });
        }

        if (blurSwitch != null) {
            blurSwitch.setChecked(ENABLE_INPUT_BLUR);
            blurSwitch.setOnCheckedChangeListener((buttonView, isChecked) -> blurEnabled = isChecked);
        }

        if (stereoSwitch != null) {
            stereoSwitch.setEnabled(false);
            stereoSwitch.setText(R.string.stereo_toggle_disabled_hint);
            stereoSwitch.setOnCheckedChangeListener((buttonView, isChecked) -> {
                if (stereoSwitchInternalChange) {
                    return;
                }
                if (!stereoPipelineAvailable) {
                    if (isChecked) {
                        Toast.makeText(this, R.string.stereo_toggle_disabled_hint, Toast.LENGTH_SHORT).show();
                    }
                    stereoSwitchInternalChange = true;
                    buttonView.setChecked(false);
                    stereoSwitchInternalChange = false;
                    stereoFusionEnabled = false;
                    updateDepthModeLabel();
                    return;
                }
                stereoFusionEnabled = isChecked;
                updateDepthModeLabel();
            });
        }

        if (settingsButton != null) {
            settingsButton.setOnClickListener(v -> toggleSettingsPanel());
            applySettingsVisibility(false);
        } else {
            applySettingsVisibility(true);
        }

        if (flipCameraButton != null) {
            flipCameraButton.setOnClickListener(v -> switchCameraFacing());
        }

        setupCalibrationControls();
        setupZoomControls();
        updateDepthModeLabel();

        exec = Executors.newSingleThreadExecutor();

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
                != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.CAMERA}, REQ);
        } else {
            start();
        }
    }

    private void start() {
        try {
            detector = new ObjectDetector(this);
        } catch (Throwable e) {
            Log.e(TAG, "Detector init failed", e);
            Toast.makeText(this, "Detector load failed: " + e.getMessage(), Toast.LENGTH_LONG).show();
            return; // detector is required
        }

        try {
            depthEstimator = new DepthEstimator(this);
        } catch (Throwable e) {
            Log.w(TAG, "Depth estimator disabled", e);
            depthEstimator = null;
            lastDepthMap = null;
        }
        stereoProcessor = null;
        updateStereoSwitchAvailability(false);

        ProcessCameraProvider.getInstance(this).addListener(() -> {
            try {
                ProcessCameraProvider provider = ProcessCameraProvider.getInstance(this).get();
                cameraProvider = provider;
                provider.unbindAll();
                cacheBackCameraIds(provider);
                bindCameraUseCases();
            } catch (Throwable e) {
                Log.e(TAG, "Camera bind error", e);
                Toast.makeText(this, "Camera error: " + e.getMessage(), Toast.LENGTH_LONG).show();
            }
        }, ContextCompat.getMainExecutor(this));
    }

    private void bindCameraUseCases() {
        if (cameraProvider == null) return;
        try {
            cameraProvider.unbindAll();

            Preview preview =
                    new Preview.Builder()
                            .setResolutionSelector(
                                    new ResolutionSelector.Builder()
                                            .setAspectRatioStrategy(
                                                    AspectRatioStrategy.RATIO_4_3_FALLBACK_AUTO_STRATEGY
                                            )
                                            .build()
                            )
                            .build();
            preview.setSurfaceProvider(previewView.getSurfaceProvider());

            ImageAnalysis analysis =
                    new ImageAnalysis.Builder()
                            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                            .setResolutionSelector(
                                    new ResolutionSelector.Builder()
                                            .setAspectRatioStrategy(
                                                    AspectRatioStrategy.RATIO_4_3_FALLBACK_AUTO_STRATEGY
                                            )
                                            .setResolutionStrategy(
                                                    new ResolutionStrategy(
                                                            new Size(360, 360),
                                                            ResolutionStrategy.FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER
                                                    )
                                            )
                                            .build()
                            )
                            .build();

            analysis.setAnalyzer(exec, image -> {
                    boolean singleShotFrame = false;
                    try {
                        boolean shouldProcess = realtimeEnabled;
                if (!shouldProcess) {
                    if (singleShotRequested && !singleShotRunning) {
                        singleShotRequested = false;
                        singleShotRunning = true;
                        singleShotFrame = true;
                        shouldProcess = true;
                        if (stereoFusionEnabled && !stereoPipelineAvailable) {
                            singleShotFrame = false; // handled inside fallback
                            handleSequentialDualShot();
                            return;
                        }
                    }
                }

                        if (!shouldProcess) {
                            return;
                        }

                        int frameW = image.getWidth();
                        int frameH = image.getHeight();
                        int rotation = image.getImageInfo().getRotationDegrees();
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

                        int[] detectorInput = argb;
                        if (blurEnabled && BLUR_RADIUS > 0) {
                            detectorInput = boxBlur(argb, frameW, frameH, BLUR_RADIUS);
                        }

                        List<ObjectDetector.Detection> dets =
                                detector.detect(detectorInput, frameW, frameH);

                        DepthEstimator.DepthMap depthForFusion = null;
                        if (depthEstimator != null) {
                            long now = SystemClock.elapsedRealtime();
                            boolean shouldRunDepth = now - lastDepthMillis >= DEPTH_INTERVAL_MS;
                            if (shouldRunDepth) {
                                try {
                                    DepthEstimator.DepthMap depth = depthEstimator.estimate(argb, frameW, frameH);
                                    dets = depthEstimator.attachDepth(dets, depth);
                                    lastDepthMillis = SystemClock.elapsedRealtime();
                                    lastDepthMap = depth;
                                    lastDepthCacheTime = lastDepthMillis;
                                    depthForFusion = depth;
                                } catch (Throwable depthErr) {
                                    Log.w(TAG, "Depth inference disabled due to error", depthErr);
                                    try {
                                        depthEstimator.close();
                                    } catch (Exception ignore) {}
                                    depthEstimator = null;
                                    lastDepthMap = null;
                                }
                            } else if (lastDepthMap != null && now - lastDepthCacheTime <= DEPTH_CACHE_MS) {
                                dets = depthEstimator.attachDepth(dets, lastDepthMap);
                                depthForFusion = lastDepthMap;
                            }
                        }

                        if (stereoFusionEnabled && stereoProcessor != null
                                && depthForFusion != null && dets != null) {
                            dets = stereoProcessor.fuseDepth(depthForFusion, dets, frameW, frameH);
                        }

                        // Draw on overlay on UI thread
                        int finalW = frameW;
                        int finalH = frameH;
                        List<ObjectDetector.Detection> finalDets = dets;
                        runOnUiThread(() -> overlay.setDetections(finalDets, finalW, finalH));
                    } catch (OrtException t) {
                        Log.e(TAG, "detect failed", t);
                    } catch (Throwable t) {
                        Log.e(TAG, "analyzer crash", t);
                    } finally {
                        // ALWAYS close frame or pipeline can stall/crash
                        image.close();
                        if (singleShotFrame) {
                            singleShotRunning = false;
                            runOnUiThread(() -> {
                                if (detectOnceButton != null) {
                                    detectOnceButton.setEnabled(true);
                                }
                            });
                        }
                    }
                });

            CameraSelector selector = buildSelector(cameraProvider);

            Camera camera = cameraProvider.bindToLifecycle(this, selector, preview, analysis);
            currentCamera = camera;
            observeZoom(camera);
            try {
                if (lensFacing == CameraSelector.LENS_FACING_BACK) {
                    stereoProcessor = new StereoDepthProcessor(this,
                            Camera2CameraInfo.extractCameraCharacteristics(camera.getCameraInfo()));
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
        } catch (Throwable e) {
            Log.e(TAG, "Camera bind error", e);
            Toast.makeText(this, "Camera error: " + e.getMessage(), Toast.LENGTH_LONG).show();
        }
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
            }
            stereoSwitchInternalChange = false;
            updateDepthModeLabel();
        });
    }

    private String[] loadLabels() {
        List<String> list = new ArrayList<>();
        BufferedReader br = null;
        try {
            br = new BufferedReader(new InputStreamReader(getAssets().open("labels.txt")));
            String line;
            while ((line = br.readLine()) != null) list.add(line);
        } catch (Exception ignored) {
        } finally {
            try { if (br != null) br.close(); } catch (Exception ignored) {}
        }
        return list.toArray(new String[0]);
    }

    @Override public void onRequestPermissionsResult(int c, @NonNull String[] p, @NonNull int[] r) {
        super.onRequestPermissionsResult(c,p,r);
        if (c == REQ && r.length > 0 && r[0] == PackageManager.PERMISSION_GRANTED) start();
    }

    @Override protected void onDestroy() {
        super.onDestroy();
        if (exec != null) exec.shutdownNow();
        if (detector != null) {
            try {
                detector.close();
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        }
        if (depthEstimator != null) {
            try {
                depthEstimator.close();
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        }
        stereoProcessor = null;
        lastDepthMap = null;
    }

    private void toggleSettingsPanel() {
        boolean visible = controlPanel != null && controlPanel.getVisibility() != View.VISIBLE;
        applySettingsVisibility(visible);
    }

    private void applySettingsVisibility(boolean visible) {
        if (controlPanel != null) {
            controlPanel.setVisibility(visible ? View.VISIBLE : View.GONE);
        }
        if (settingsButton != null) {
            settingsButton.setIconResource(
                    visible ? android.R.drawable.ic_menu_close_clear_cancel
                            : android.R.drawable.ic_menu_manage
            );
            settingsButton.setContentDescription(
                    getString(visible ? R.string.settings_hide : R.string.settings_show)
            );
        }
    }

    private void setupCalibrationControls() {
        if (calibrationSeek == null) {
            DepthEstimator.setUserScale(calibrationScale);
            updateCalibrationLabel(calibrationScale);
            return;
        }
        calibrationSeek.setMax(CALIB_PROGRESS_MAX);
        calibrationSeek.setProgress(scaleToProgress(calibrationScale));
        updateCalibrationLabel(calibrationScale);
        calibrationSeek.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
            @Override
            public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser) {
                float scale = progressToScale(progress);
                calibrationScale = scale;
                DepthEstimator.setUserScale(scale);
                updateCalibrationLabel(scale);
                saveCalibration(scale);
            }

            @Override public void onStartTrackingTouch(SeekBar seekBar) { }

            @Override public void onStopTrackingTouch(SeekBar seekBar) { }
        });
    }

    private void updateCalibrationLabel(float scale) {
        if (calibrationValue == null) return;
        calibrationValue.setText(getString(R.string.depth_calibration_value, scale));
    }

    private void saveCalibration(float scale) {
        if (prefs == null || calibrationPrefKey == null) return;
        prefs.edit().putFloat(calibrationPrefKey, scale).apply();
    }

    private float loadSavedCalibrationScale() {
        if (prefs == null || calibrationPrefKey == null) return DepthEstimator.getUserScale();
        return prefs.getFloat(calibrationPrefKey, DepthEstimator.getUserScale());
    }

    private void setupZoomControls() {
        if (zoomSeek == null) return;
        zoomSeek.setMax(ZOOM_PROGRESS_MAX);
        zoomSeek.setProgress(0);
        if (zoomValue != null) {
            zoomValue.setText(getString(R.string.zoom_value_default));
        }
        zoomSeek.setOnSeekBarChangeListener(zoomListener);
    }

    private int scaleToProgress(float scale) {
        float clamped = Math.max(CALIB_MIN, Math.min(CALIB_MAX, scale));
        float pct = (clamped - CALIB_MIN) / (CALIB_MAX - CALIB_MIN);
        return Math.round(pct * CALIB_PROGRESS_MAX);
    }

    private float progressToScale(int progress) {
        int p = Math.max(0, Math.min(CALIB_PROGRESS_MAX, progress));
        float pct = p / (float) CALIB_PROGRESS_MAX;
        return CALIB_MIN + pct * (CALIB_MAX - CALIB_MIN);
    }

    private String buildCalibrationKey() {
        String model = Build.MANUFACTURER + "_" + Build.MODEL;
        float aperture = resolveBackCameraAperture();
        if (aperture > 0f) {
            return PREF_KEY_PREFIX + model + "_" + String.format(Locale.US, "%.2f", aperture);
        }
        return PREF_KEY_PREFIX + model;
    }

    private float resolveBackCameraAperture() {
        CameraManager mgr = (CameraManager) getSystemService(CAMERA_SERVICE);
        if (mgr == null) return -1f;
        try {
            for (String id : mgr.getCameraIdList()) {
                CameraCharacteristics cc = mgr.getCameraCharacteristics(id);
                Integer facing = cc.get(CameraCharacteristics.LENS_FACING);
                if (facing != null && facing == CameraCharacteristics.LENS_FACING_BACK) {
                    float[] apertures = cc.get(CameraCharacteristics.LENS_INFO_AVAILABLE_APERTURES);
                    if (apertures != null && apertures.length > 0) {
                        return apertures[0];
                    }
                }
            }
        } catch (Exception e) {
            Log.w(TAG, "Unable to read aperture for calibration key", e);
        }
        return -1f;
    }

    private void updateDepthModeLabel() {
        if (depthModeText == null) return;
        boolean stereoActive = stereoFusionEnabled && stereoPipelineAvailable && stereoProcessor != null;
        depthModeText.setText(getString(stereoActive ? R.string.depth_mode_stereo : R.string.depth_mode_mono));
    }

    private void cacheBackCameraIds(@NonNull ProcessCameraProvider provider) {
        java.util.List<CamInfo> list = new java.util.ArrayList<>();
        try {
            for (androidx.camera.core.CameraInfo info : provider.getAvailableCameraInfos()) {
                Integer facing = info.getLensFacing();
                if (facing != null && facing == CameraSelector.LENS_FACING_BACK) {
                    String id = Camera2CameraInfo.from(info).getCameraId();
                    CameraCharacteristics cc = Camera2CameraInfo.extractCameraCharacteristics(info);
                    float focal = 0f;
                    if (cc != null) {
                        float[] focals = cc.get(CameraCharacteristics.LENS_INFO_AVAILABLE_FOCAL_LENGTHS);
                        if (focals != null && focals.length > 0) focal = focals[0];
                    }
                    list.add(new CamInfo(id, focal));
                }
            }
            list.sort((a, b) -> Float.compare(a.focalLength, b.focalLength));
        } catch (Exception e) {
            Log.w(TAG, "cacheBackCameraIds failed", e);
        }
        backCameraInfos = list;
    }

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
                Toast.makeText(this, R.string.stereo_toggle_disabled_hint, Toast.LENGTH_SHORT).show();
            });
            return;
        }
        if (sequentialStereoRunning) return;
        singleShotRunning = true;
        sequentialStereoRunning = true;
        runOnUiThread(() -> Toast.makeText(this, R.string.sequential_dual_shot, Toast.LENGTH_SHORT).show());
        exec.execute(() -> {
            try {
                if (backCameraInfos == null || backCameraInfos.isEmpty()) {
                    cacheBackCameraIds(cameraProvider);
                }
                java.util.List<String> ids = chooseSequentialCamIds();
                if (ids == null || ids.isEmpty()) {
                    Log.w(TAG, "No secondary back camera found for sequential stereo");
                    return;
                }
                java.util.List<ObjectDetector.Detection> lastDets = null;
                DepthEstimator.DepthMap lastDepth = null;
                int lastW = 0, lastH = 0;
                for (String camId : ids) {
                    FrameCaptureResult res = captureSingleFrame(camId);
                    if (res != null && res.detections != null) {
                        lastDets = res.detections;
                        lastDepth = res.depthMap;
                        lastW = res.width;
                        lastH = res.height;
                    }
                }
                if (lastDets != null) {
                    if (stereoFusionEnabled && stereoProcessor != null && lastDepth != null) {
                        lastDets = stereoProcessor.fuseDepth(lastDepth, lastDets, lastW, lastH);
                    }
                    int fw = lastW, fh = lastH;
                    java.util.List<ObjectDetector.Detection> fd = lastDets;
                    runOnUiThread(() -> overlay.setDetections(fd, fw, fh));
                }
            } catch (Exception e) {
                Log.e(TAG, "Sequential stereo single shot failed", e);
            } finally {
                sequentialStereoRunning = false;
                singleShotRunning = false;
                runOnUiThread(() -> {
                    if (detectOnceButton != null) detectOnceButton.setEnabled(true);
                    bindCameraUseCases();
                });
            }
        });
    }

    private FrameCaptureResult captureSingleFrame(String cameraId) {
        CountDownLatch latch = new CountDownLatch(1);
        final FrameCaptureResult[] holder = new FrameCaptureResult[1];
        try {
            cameraProvider.unbindAll();
            Preview preview = new Preview.Builder()
                    .setResolutionSelector(
                            new ResolutionSelector.Builder()
                                    .setAspectRatioStrategy(AspectRatioStrategy.RATIO_4_3_FALLBACK_AUTO_STRATEGY)
                                    .build()
                    )
                    .build();
            preview.setSurfaceProvider(previewView.getSurfaceProvider());

            ImageAnalysis analysis = new ImageAnalysis.Builder()
                    .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                    .setResolutionSelector(
                            new ResolutionSelector.Builder()
                                    .setAspectRatioStrategy(AspectRatioStrategy.RATIO_4_3_FALLBACK_AUTO_STRATEGY)
                                    .setResolutionStrategy(
                                            new ResolutionStrategy(
                                                    new Size(360, 360),
                                                    ResolutionStrategy.FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER
                                            )
                                    )
                                    .build()
                    )
                    .build();

            analysis.setAnalyzer(exec, image -> {
                try {
                    int frameW = image.getWidth();
                    int frameH = image.getHeight();
                    int rotation = image.getImageInfo().getRotationDegrees();
                    int[] argb = Yuv.toArgb(image);
                    if (rotation != 0) {
                        argb = Yuv.rotate(argb, frameW, frameH, rotation);
                        if (rotation == 90 || rotation == 270) {
                            int tmp = frameW;
                            frameW = frameH;
                            frameH = tmp;
                        }
                    }
                    int[] detectorInput = argb;
                    if (blurEnabled && BLUR_RADIUS > 0) {
                        detectorInput = boxBlur(argb, frameW, frameH, BLUR_RADIUS);
                    }
                    java.util.List<ObjectDetector.Detection> dets = detector.detect(detectorInput, frameW, frameH);
                    DepthEstimator.DepthMap depth = null;
                    if (depthEstimator != null) {
                        try {
                            depth = depthEstimator.estimate(argb, frameW, frameH);
                            dets = depthEstimator.attachDepth(dets, depth);
                        } catch (Throwable depthErr) {
                            Log.w(TAG, "Depth inference disabled due to error", depthErr);
                            try { depthEstimator.close(); } catch (Exception ignore) {}
                            depthEstimator = null;
                        }
                    }
                    holder[0] = new FrameCaptureResult(dets, depth, frameW, frameH);
                } catch (Exception e) {
                    Log.e(TAG, "captureSingleFrame analyzer failed", e);
                } finally {
                    image.close();
                    latch.countDown();
                }
            });

            CameraSelector selector = new CameraSelector.Builder()
                    .addCameraFilter(cameraInfos -> {
                        java.util.List<androidx.camera.core.CameraInfo> filtered = new java.util.ArrayList<>();
                        for (androidx.camera.core.CameraInfo info : cameraInfos) {
                            try {
                                String id = Camera2CameraInfo.from(info).getCameraId();
                                if (cameraId.equals(id)) {
                                    filtered.add(info);
                                    break;
                                }
                            } catch (Exception ignored) { }
                        }
                        return filtered.isEmpty() ? cameraInfos : filtered;
                    })
                    .requireLensFacing(CameraSelector.LENS_FACING_BACK)
                    .build();

            cameraProvider.bindToLifecycle(this, selector, preview, analysis);
            latch.await(1500, java.util.concurrent.TimeUnit.MILLISECONDS);
            cameraProvider.unbind(preview, analysis);
        } catch (Exception e) {
            Log.e(TAG, "captureSingleFrame failed for id=" + cameraId, e);
        }
        return holder[0];
    }

    private static class FrameCaptureResult {
        final java.util.List<ObjectDetector.Detection> detections;
        final DepthEstimator.DepthMap depthMap;
        final int width;
        final int height;
        FrameCaptureResult(java.util.List<ObjectDetector.Detection> detections,
                           DepthEstimator.DepthMap depthMap,
                           int width, int height) {
            this.detections = detections;
            this.depthMap = depthMap;
            this.width = width;
            this.height = height;
        }
    }

    private static class CamInfo {
        final String id;
        final float focalLength;
        CamInfo(String id, float focalLength) {
            this.id = id;
            this.focalLength = focalLength;
        }
    }

    private void observeZoom(Camera camera) {
        if (zoomSeek == null || zoomValue == null) return;
        camera.getCameraInfo().getZoomState().observe(this, state -> {
            if (state == null) return;
            zoomMinRatio = state.getMinZoomRatio();
            zoomMaxRatio = state.getMaxZoomRatio();
            int progress = zoomRatioToProgress(state.getZoomRatio());
            zoomSeek.setOnSeekBarChangeListener(null);
            zoomSeek.setProgress(progress);
            zoomValue.setText(getString(R.string.zoom_value, state.getZoomRatio()));
            zoomSeek.setOnSeekBarChangeListener(zoomListener);
        });
        zoomSeek.setOnSeekBarChangeListener(zoomListener);
    }

    private final SeekBar.OnSeekBarChangeListener zoomListener = new SeekBar.OnSeekBarChangeListener() {
        @Override
        public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser) {
            if (!fromUser || currentCamera == null) return;
            float ratio = progressToZoomRatio(progress);
            currentCamera.getCameraControl().setZoomRatio(ratio);
        }

        @Override public void onStartTrackingTouch(SeekBar seekBar) { }
        @Override public void onStopTrackingTouch(SeekBar seekBar) { }
    };

    private int zoomRatioToProgress(float ratio) {
        float min = Math.max(1f, zoomMinRatio);
        float max = Math.max(min, zoomMaxRatio);
        float clamped = Math.max(min, Math.min(max, ratio));
        float pct = (clamped - min) / Math.max(1e-6f, (max - min));
        return Math.round(pct * ZOOM_PROGRESS_MAX);
    }

    private float progressToZoomRatio(int progress) {
        int p = Math.max(0, Math.min(ZOOM_PROGRESS_MAX, progress));
        float min = Math.max(1f, zoomMinRatio);
        float max = Math.max(min, zoomMaxRatio);
        float pct = p / (float) ZOOM_PROGRESS_MAX;
        return min + pct * (max - min);
    }

    private java.util.List<String> chooseSequentialCamIds() {
        if (backCameraInfos == null || backCameraInfos.isEmpty()) return java.util.Collections.emptyList();
        if (backCameraInfos.size() == 1) return java.util.Collections.singletonList(backCameraInfos.get(0).id);
        CamInfo wide = backCameraInfos.get(0);
        CamInfo tele = backCameraInfos.get(backCameraInfos.size() - 1);
        if (wide.id.equals(tele.id)) {
            return java.util.Collections.singletonList(wide.id);
        }
        java.util.List<String> out = new java.util.ArrayList<>(2);
        out.add(wide.id); // 1x
        out.add(tele.id); // 2x (longer focal)
        return out;
    }

    private void switchCameraFacing() {
        lensFacing = (lensFacing == CameraSelector.LENS_FACING_BACK)
                ? CameraSelector.LENS_FACING_FRONT
                : CameraSelector.LENS_FACING_BACK;
        updateStereoSwitchAvailability(false);
        bindCameraUseCases();
        updateDepthModeLabel();
    }

    private CameraSelector buildSelector(@NonNull ProcessCameraProvider provider) {
        if (lensFacing == CameraSelector.LENS_FACING_FRONT) {
            return new CameraSelector.Builder()
                    .requireLensFacing(CameraSelector.LENS_FACING_FRONT)
                    .build();
        }
        String logicalId = findLogicalMultiCameraId(provider);
        Log.i(TAG, "Selected logical multi-camera id=" + logicalId);
        CameraSelector.Builder builder = new CameraSelector.Builder()
                .requireLensFacing(CameraSelector.LENS_FACING_BACK);
        if (logicalId == null) {
            return builder.build();
        }
        return builder.addCameraFilter(cameraInfos -> {
            for (androidx.camera.core.CameraInfo info : cameraInfos) {
                try {
                    String id = Camera2CameraInfo.from(info).getCameraId();
                    if (logicalId.equals(id)) {
                        return java.util.Collections.singletonList(info);
                    }
                } catch (Exception ignored) { }
            }
            return cameraInfos;
        }).build();
    }

    private String findLogicalMultiCameraId(@NonNull ProcessCameraProvider provider) {
        try {
            for (androidx.camera.core.CameraInfo info : provider.getAvailableCameraInfos()) {
                Integer facing = info.getLensFacing();
                if (facing == null || facing != CameraSelector.LENS_FACING_BACK) continue;
                Camera2CameraInfo c2 = Camera2CameraInfo.from(info);
                CameraCharacteristics cc = Camera2CameraInfo.extractCameraCharacteristics(info);
                if (cc == null) continue;
                java.util.Set<String> ids = cc.getPhysicalCameraIds();
                if (ids != null && ids.size() >= 2) {
                    return c2.getCameraId();
                }
            }
        } catch (Exception e) {
            Log.w(TAG, "findLogicalMultiCameraId failed", e);
        }
        return null;
    }

    private static int[] boxBlur(int[] src, int w, int h, int radius) {
        int kernel = (radius * 2 + 1) * (radius * 2 + 1);
        int[] dst = new int[w * h];
        for (int y = 0; y < h; y++) {
            int yMin = Math.max(0, y - radius);
            int yMax = Math.min(h - 1, y + radius);
            for (int x = 0; x < w; x++) {
                int xMin = Math.max(0, x - radius);
                int xMax = Math.min(w - 1, x + radius);

                int count = 0;
                int sumR = 0, sumG = 0, sumB = 0;
                for (int yy = yMin; yy <= yMax; yy++) {
                    int base = yy * w;
                    for (int xx = xMin; xx <= xMax; xx++) {
                        int c = src[base + xx];
                        sumR += (c >> 16) & 0xFF;
                        sumG += (c >> 8) & 0xFF;
                        sumB += c & 0xFF;
                        count++;
                    }
                }
                if (count == 0) count = 1;
                int r = sumR / count;
                int g = sumG / count;
                int b = sumB / count;
                dst[y * w + x] = 0xFF000000 | (r << 16) | (g << 8) | b;
            }
        }
        return dst;
    }
}
