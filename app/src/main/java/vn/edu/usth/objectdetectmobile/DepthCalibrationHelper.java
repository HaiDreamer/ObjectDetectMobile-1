package vn.edu.usth.objectdetectmobile;

import android.content.Context;
import android.content.SharedPreferences;
import android.hardware.camera2.CameraCharacteristics;
import android.hardware.camera2.CameraManager;
import android.os.Build;
import android.util.Log;
import android.widget.SeekBar;
import android.widget.TextView;

import java.util.Locale;

public final class DepthCalibrationHelper {

    private static final float CALIB_MIN = 0.5f;
    private static final float CALIB_MAX = 2.5f;
    private static final int CALIB_PROGRESS_MAX = 200;
    private static final String PREFS_NAME = "depth_calibration_prefs";
    private static final String PREF_KEY_PREFIX = "depth_calibration_scale_";
    private static final String TAG = "DepthCalibrationHelper";

    public static void bindCalibrationSeekBar(
            SeekBar seekBar,
            TextView label,
            float initialScale,
            SharedPreferences prefs,
            String prefKey
    ) {
        if (seekBar == null || label == null) {
            DepthEstimator.setUserScale(initialScale);
            label.setText(String.format(Locale.US, "%.2f", initialScale));
            return;
        }

        seekBar.setMax(getMaxProgress());
        seekBar.setProgress(scaleToProgress(initialScale));
        label.setText(String.format(Locale.US, "%.2f", initialScale));
        DepthEstimator.setUserScale(initialScale);

        seekBar.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
            @Override
            public void onProgressChanged(SeekBar sb, int progress, boolean fromUser) {
                float scale = progressToScale(progress);
                DepthEstimator.setUserScale(scale);
                label.setText(String.format(Locale.US, "%.2f", scale));
                saveCalibration(prefs, prefKey, scale);
            }
            @Override public void onStartTrackingTouch(SeekBar sb) { }
            @Override public void onStopTrackingTouch(SeekBar sb) { }
        });
    }

    // ---------- UI mapping: scale <-> SeekBar progress ----------

    public static int scaleToProgress(float scale) {
        float clamped = Math.max(CALIB_MIN, Math.min(CALIB_MAX, scale));
        float pct = (clamped - CALIB_MIN) / (CALIB_MAX - CALIB_MIN);
        return Math.round(pct * CALIB_PROGRESS_MAX);
    }

    public static float progressToScale(int progress) {
        int p = Math.max(0, Math.min(CALIB_PROGRESS_MAX, progress));
        float pct = p / (float) CALIB_PROGRESS_MAX;
        return CALIB_MIN + pct * (CALIB_MAX - CALIB_MIN);
    }

    public static int getMaxProgress() {
        return CALIB_PROGRESS_MAX;
    }

    // ---------- Prefs helpers ----------

    public static SharedPreferences getPrefs(Context ctx) {
        return ctx.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE);
    }

    public static String buildCalibrationKey(Context ctx) {
        String model = Build.MANUFACTURER + "_" + Build.MODEL;
        float aperture = resolveBackCameraAperture(ctx);
        if (aperture > 0f) {
            return PREF_KEY_PREFIX + model + "_" +
                    String.format(Locale.US, "%.2f", aperture);
        }
        return PREF_KEY_PREFIX + model;
    }

    public static float loadSavedCalibrationScale(SharedPreferences prefs,
                                                  String calibrationPrefKey,
                                                  float defaultValue) {
        if (prefs == null || calibrationPrefKey == null) return defaultValue;
        return prefs.getFloat(calibrationPrefKey, defaultValue);
    }

    public static void saveCalibration(SharedPreferences prefs,
                                       String calibrationPrefKey,
                                       float scale) {
        if (prefs == null || calibrationPrefKey == null) return;
        prefs.edit().putFloat(calibrationPrefKey, scale).apply();
    }

    // ---------- Camera aperture ----------

    private static float resolveBackCameraAperture(Context ctx) {
        CameraManager mgr = (CameraManager) ctx.getSystemService(Context.CAMERA_SERVICE);
        if (mgr == null) return -1f;
        try {
            for (String id : mgr.getCameraIdList()) {
                CameraCharacteristics cc = mgr.getCameraCharacteristics(id);
                Integer facing = cc.get(CameraCharacteristics.LENS_FACING);
                if (facing != null &&
                        facing == CameraCharacteristics.LENS_FACING_BACK) {
                    float[] apertures =
                            cc.get(CameraCharacteristics.LENS_INFO_AVAILABLE_APERTURES);
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
}
