package vn.edu.usth.objectdetectmobile;

import android.os.SystemClock;
import android.util.Log;

import java.util.List;

public class DepthPipelineHelper {

    private static final String TAG = "DepthPipelineHelper";

    /**
     * Holds depth cache state across frames.
     */
    public static class DepthState {
        public long lastDepthMillis = 0L;
        public long lastDepthCacheTime = 0L;
        public DepthEstimator.DepthMap lastDepthMap = null;
    }

    /**
     * Result of running the depth pipeline: updated detections and
     * the depth map that should be used for fusion, if any.
     */
    public static class DepthResult {
        public final List<ObjectDetector.Detection> dets;
        public final DepthEstimator.DepthMap depthMap;

        public DepthResult(List<ObjectDetector.Detection> dets,
                           DepthEstimator.DepthMap depthMap) {
            this.dets = dets;
            this.depthMap = depthMap;
        }
    }

    /**
     * Run (or reuse) depth for a frame, attaching depth info to detections.
     *
     * - Uses intervalMs to throttle depth inference.
     * - Uses cacheMs to reuse recent depth maps.
     * - Updates the provided DepthState in-place.
     */
    public static DepthResult maybeRunDepth(
            DepthEstimator estimator,
            DepthState state,
            int[] argb,
            int frameW,
            int frameH,
            List<ObjectDetector.Detection> dets,
            long intervalMs,
            long cacheMs
    ) {
        if (state == null) {
            throw new IllegalArgumentException("DepthState must not be null");
        }

        if (estimator == null) {
            // No estimator => just reuse last cache for fusion (if any),
            // but don't modify detections.
            return new DepthResult(dets, state.lastDepthMap);
        }

        DepthEstimator.DepthMap depthForFusion = null;
        long now = SystemClock.elapsedRealtime();
        boolean shouldRunDepth = now - state.lastDepthMillis >= intervalMs;

        try {
            if (shouldRunDepth) {
                DepthEstimator.DepthMap depth =
                        estimator.estimate(argb, frameW, frameH);
                dets = estimator.attachDepth(dets, depth);

                state.lastDepthMillis = now;
                state.lastDepthMap = depth;
                state.lastDepthCacheTime = now;

                depthForFusion = depth;
            } else if (state.lastDepthMap != null &&
                    now - state.lastDepthCacheTime <= cacheMs) {
                dets = estimator.attachDepth(dets, state.lastDepthMap);
                depthForFusion = state.lastDepthMap;
            }
        } catch (Throwable depthErr) {
            Log.w(TAG, "Depth inference error", depthErr);
            // We leave estimator as-is. Caller can decide to disable it
            // if they want to, but we don't mutate it here.
            depthForFusion = null;
        }

        return new DepthResult(dets, depthForFusion);
    }
}
