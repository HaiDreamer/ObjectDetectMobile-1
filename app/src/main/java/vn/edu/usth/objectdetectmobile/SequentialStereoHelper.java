// SequentialStereoHelper.java
package vn.edu.usth.objectdetectmobile;

import android.util.Size;

import androidx.activity.ComponentActivity;
import androidx.annotation.NonNull;
import androidx.annotation.OptIn;
import androidx.camera.camera2.interop.Camera2CameraInfo;
import androidx.camera.camera2.interop.ExperimentalCamera2Interop;
import androidx.camera.core.CameraInfo;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageProxy;
import androidx.camera.core.Preview;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.camera.view.PreviewView;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

// Non-deprecated resolution selector API
import androidx.camera.core.resolutionselector.AspectRatioStrategy;
import androidx.camera.core.resolutionselector.ResolutionSelector;
import androidx.camera.core.resolutionselector.ResolutionStrategy;

public class SequentialStereoHelper {

    private SequentialStereoHelper() {}

    // Result of the whole dual-shot run
    public static class Result {
        public final List<ObjectDetector.Detection> detections;
        public final DepthEstimator.DepthMap depthMap;
        public final int width;
        public final int height;

        public Result(List<ObjectDetector.Detection> detections,
                      DepthEstimator.DepthMap depthMap,
                      int width,
                      int height) {
            this.detections = detections;
            this.depthMap = depthMap;
            this.width = width;
            this.height = height;
        }
    }

    public interface Callback {
        /** Called on main thread when the final fused detections are ready */
        void onResult(Result result);

        /** Called on main thread if something goes wrong (optional: you can leave empty) */
        void onError(Throwable t);

        /** Called on main thread when everything is finished (success or error) */
        void onFinished();
    }

    /**
     * Run sequential stereo: capture from wide + tele (or single cam),
     * run detector + depth on each, then optionally fuse with StereoDepthProcessor.
     *
     * This runs on a background executor; all callbacks are posted to main thread.
     */
    @OptIn(markerClass = ExperimentalCamera2Interop.class)
    public static void runSequentialStereoShot(
            @NonNull ComponentActivity activity,
            @NonNull ProcessCameraProvider cameraProvider,
            @NonNull PreviewView previewView,
            @NonNull ExecutorService sharedExecutor,
            @NonNull List<CameraUtils.CamInfo> backCameraInfos,
            @NonNull ObjectDetector detector,
            DepthEstimator depthEstimator,          // may be null
            boolean blurEnabled,
            int blurRadius,
            boolean stereoFusionEnabled,
            StereoDepthProcessor stereoProcessor,   // may be null if fusion disabled
            @NonNull Callback callback
    ) {
        sharedExecutor.execute(() -> {
            try {
                List<String> ids = CameraUtils.chooseSequentialCamIds(backCameraInfos);
                if (ids == null || ids.isEmpty()) {
                    activity.runOnUiThread(() ->
                            callback.onError(new IllegalStateException("No back cameras")));
                    return;
                }

                List<ObjectDetector.Detection> lastDets = null;
                DepthEstimator.DepthMap lastDepth = null;
                int lastW = 0, lastH = 0;

                for (String camId : ids) {
                    FrameCaptureResult res = captureSingleFrame(
                            activity,
                            cameraProvider,
                            previewView,
                            detector,
                            depthEstimator,
                            blurEnabled,
                            blurRadius,
                            camId
                    );
                    if (res != null && res.detections != null) {
                        lastDets = res.detections;
                        lastDepth = res.depthMap;
                        lastW = res.width;
                        lastH = res.height;
                    }
                }

                if (lastDets == null) {
                    activity.runOnUiThread(() ->
                            callback.onError(new IllegalStateException("No detections")));
                    return;
                }

                // Optionally fuse
                if (stereoFusionEnabled && stereoProcessor != null && lastDepth != null) {
                    lastDets = stereoProcessor.fuseDepth(lastDepth, lastDets, lastW, lastH);
                }

                final List<ObjectDetector.Detection> finalDets = lastDets;
                final DepthEstimator.DepthMap finalDepth = lastDepth;
                final int fw = lastW;
                final int fh = lastH;

                activity.runOnUiThread(() ->
                        callback.onResult(new Result(finalDets, finalDepth, fw, fh)));

            } catch (Throwable t) {
                activity.runOnUiThread(() -> callback.onError(t));
            } finally {
                activity.runOnUiThread(callback::onFinished);
            }
        });
    }

    // -----------------------------------------------------------------------------------------
    // Single-frame capture (moved from MainActivity)
    // -----------------------------------------------------------------------------------------
    @OptIn(markerClass = ExperimentalCamera2Interop.class)
    private static FrameCaptureResult captureSingleFrame(
            @NonNull ComponentActivity activity,
            @NonNull ProcessCameraProvider cameraProvider,
            @NonNull PreviewView previewView,
            @NonNull ObjectDetector detector,
            DepthEstimator depthEstimator,  // may be null
            boolean blurEnabled,
            int blurRadius,
            @NonNull String cameraId
    ) {
        CountDownLatch latch = new CountDownLatch(1);
        final FrameCaptureResult[] holder = new FrameCaptureResult[1];

        ExecutorService captureAnalyzerExecutor = Executors.newSingleThreadExecutor();

        try {
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
                                    .setAspectRatioStrategy(
                                            AspectRatioStrategy.RATIO_4_3_FALLBACK_AUTO_STRATEGY
                                    )
                                    .setResolutionStrategy(
                                            new ResolutionStrategy(
                                                    new Size(360, 360),
                                                    ResolutionStrategy
                                                            .FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER
                                            )
                                    )
                                    .build()
                    )
                    .build();

            analysis.setAnalyzer(captureAnalyzerExecutor, image -> {
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

                    int[] detectorInput = (blurEnabled && blurRadius > 0)
                            ? ImageUtils.boxBlur(argb, frameW, frameH, blurRadius)
                            : argb;

                    List<ObjectDetector.Detection> dets =
                            detector.detect(detectorInput, frameW, frameH);

                    DepthEstimator.DepthMap depth = null;
                    if (depthEstimator != null) {
                        try {
                            depth = depthEstimator.estimate(argb, frameW, frameH);
                            dets = depthEstimator.attachDepth(dets, depth);
                        } catch (Throwable depthErr) {
                            // If depth fails, just skip depth; the app will disable it elsewhere
                            depth = null;
                        }
                    }

                    holder[0] = new FrameCaptureResult(dets, depth, frameW, frameH);
                } catch (Exception e) {
                    // swallow & log
                    e.printStackTrace();
                } finally {
                    image.close();
                    latch.countDown();
                }
            });

            CameraSelector selector = new CameraSelector.Builder()
                    .addCameraFilter(cameraInfos -> {
                        List<CameraInfo> filtered = new ArrayList<>();
                        for (CameraInfo info : cameraInfos) {
                            try {
                                String id = Camera2CameraInfo.from(info).getCameraId();
                                if (cameraId.equals(id)) {
                                    filtered.add(info);
                                    break;
                                }
                            } catch (Exception ignored) {
                            }
                        }
                        return filtered.isEmpty() ? cameraInfos : filtered;
                    })
                    .requireLensFacing(CameraSelector.LENS_FACING_BACK)
                    .build();

            activity.runOnUiThread(() -> {
                try {
                    cameraProvider.unbindAll();
                    cameraProvider.bindToLifecycle(activity, selector, preview, analysis);
                } catch (Exception e) {
                    e.printStackTrace();
                }
            });

            latch.await(1500, TimeUnit.MILLISECONDS);

            activity.runOnUiThread(() -> {
                try {
                    cameraProvider.unbind(preview, analysis);
                } catch (Exception e) {
                    e.printStackTrace();
                }
            });

        } catch (Exception e) {
            e.printStackTrace();
        } finally {
            captureAnalyzerExecutor.shutdownNow();
        }

        return holder[0];
    }

    private static class FrameCaptureResult {
        final List<ObjectDetector.Detection> detections;
        final DepthEstimator.DepthMap depthMap;
        final int width;
        final int height;

        FrameCaptureResult(List<ObjectDetector.Detection> detections,
                           DepthEstimator.DepthMap depthMap,
                           int width,
                           int height) {
            this.detections = detections;
            this.depthMap = depthMap;
            this.width = width;
            this.height = height;
        }
    }
}
