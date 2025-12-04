package vn.edu.usth.objectdetectmobile;

import android.widget.SeekBar;
import android.widget.TextView;

import androidx.camera.core.Camera;

import java.util.Locale;

public final class ZoomHelper {

    public static final int ZOOM_PROGRESS_MAX = 1000;

    private ZoomHelper() {}

    public interface ZoomControl {
        float getMinZoomRatio();
        float getMaxZoomRatio();
        void setZoomRatio(float ratio);
    }

    public static void setupZoomSeekBar(
            SeekBar zoomSeek,
            TextView zoomValue,
            ZoomControl control
    ) {
        if (zoomSeek == null || zoomValue == null || control == null) return;

        zoomSeek.setEnabled(true);
        zoomSeek.setClickable(true);
        zoomSeek.setMax(ZOOM_PROGRESS_MAX);
        zoomSeek.setProgress(0);
        zoomValue.setText("1.0x");

        zoomSeek.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
            @Override
            public void onProgressChanged(SeekBar seekBar,
                                          int progress,
                                          boolean fromUser) {
                if (!fromUser) return;
                float ratio = progressToZoomRatio(progress,
                        control.getMinZoomRatio(),
                        control.getMaxZoomRatio());
                control.setZoomRatio(ratio);
                zoomValue.setText(String.format(Locale.US, "%.2fx", ratio));
            }

            @Override public void onStartTrackingTouch(SeekBar seekBar) { }
            @Override public void onStopTrackingTouch(SeekBar seekBar) { }
        });
    }

    public static int zoomRatioToProgress(float ratio, float min, float max) {
        min = Math.max(1f, min);
        max = Math.max(min, max);
        float clamped = Math.max(min, Math.min(max, ratio));
        float pct = (clamped - min) / Math.max(1e-6f, (max - min));
        return Math.round(pct * ZOOM_PROGRESS_MAX);
    }

    public static float progressToZoomRatio(int progress, float min, float max) {
        int p = Math.max(0, Math.min(ZOOM_PROGRESS_MAX, progress));
        min = Math.max(1f, min);
        max = Math.max(min, max);
        float pct = p / (float) ZOOM_PROGRESS_MAX;
        return min + pct * (max - min);
    }
}
