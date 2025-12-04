package vn.edu.usth.objectdetectmobile;

import android.annotation.SuppressLint;
import android.hardware.camera2.CameraCharacteristics;
import android.os.Build;
import android.util.Log;

import androidx.annotation.NonNull;
import androidx.annotation.OptIn;
import androidx.camera.camera2.interop.Camera2CameraInfo;
import androidx.camera.camera2.interop.ExperimentalCamera2Interop;
import androidx.camera.core.CameraInfo;
import androidx.camera.core.CameraSelector;
import androidx.camera.lifecycle.ProcessCameraProvider;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Set;

public final class CameraUtils {

    private static final String TAG = "CameraUtils";

    public static class CamInfo {
        public final String id;
        public final float focalLength;

        public CamInfo(String id, float focalLength) {
            this.id = id;
            this.focalLength = focalLength;
        }
    }

    private CameraUtils() {
        // no instances
    }

    @SuppressLint("RestrictedApi")
    @OptIn(markerClass = ExperimentalCamera2Interop.class)
    public static List<CamInfo> cacheBackCameraInfos(@NonNull ProcessCameraProvider provider) {
        List<CamInfo> list = new ArrayList<>();
        try {
            for (CameraInfo info : provider.getAvailableCameraInfos()) {
                Integer facing = info.getLensFacing();
                if (facing != null &&
                        facing == CameraSelector.LENS_FACING_BACK) {
                    String id = Camera2CameraInfo.from(info).getCameraId();
                    CameraCharacteristics cc =
                            Camera2CameraInfo.extractCameraCharacteristics(info);
                    float focal = 0f;
                    if (cc != null) {
                        float[] focals =
                                cc.get(CameraCharacteristics.LENS_INFO_AVAILABLE_FOCAL_LENGTHS);
                        if (focals != null && focals.length > 0) focal = focals[0];
                    }
                    list.add(new CamInfo(id, focal));
                }
            }
            list.sort((a, b) -> Float.compare(a.focalLength, b.focalLength));
        } catch (Exception e) {
            Log.w(TAG, "cacheBackCameraInfos failed", e);
        }
        return list;
    }

    public static List<String> chooseSequentialCamIds(List<CamInfo> backCameraInfos) {
        if (backCameraInfos == null || backCameraInfos.isEmpty())
            return Collections.emptyList();
        if (backCameraInfos.size() == 1)
            return Collections.singletonList(backCameraInfos.get(0).id);

        CamInfo wide = backCameraInfos.get(0);
        CamInfo tele = backCameraInfos.get(backCameraInfos.size() - 1);

        if (wide.id.equals(tele.id)) {
            return Collections.singletonList(wide.id);
        }
        List<String> out = new ArrayList<>(2);
        out.add(wide.id); // 1x
        out.add(tele.id); // 2x
        return out;
    }

    @SuppressLint("RestrictedApi")
    @OptIn(markerClass = ExperimentalCamera2Interop.class)
    public static CameraSelector buildBackSelector(
            @NonNull ProcessCameraProvider provider,
            int lensFacing
    ) {
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
            for (CameraInfo info : cameraInfos) {
                try {
                    String id = Camera2CameraInfo.from(info).getCameraId();
                    if (logicalId.equals(id)) {
                        return Collections.singletonList(info);
                    }
                } catch (Exception ignored) {
                }
            }
            return cameraInfos;
        }).build();
    }

    @SuppressLint("RestrictedApi")
    @OptIn(markerClass = ExperimentalCamera2Interop.class)
    private static String findLogicalMultiCameraId(@NonNull ProcessCameraProvider provider) {
        try {
            for (CameraInfo info : provider.getAvailableCameraInfos()) {
                Integer facing = info.getLensFacing();
                if (facing == null ||
                        facing != CameraSelector.LENS_FACING_BACK) continue;

                Camera2CameraInfo c2 = Camera2CameraInfo.from(info);
                CameraCharacteristics cc =
                        Camera2CameraInfo.extractCameraCharacteristics(info);
                if (cc == null) continue;

                Set<String> ids = null;
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
                    ids = cc.getPhysicalCameraIds();
                }
                if (ids != null && ids.size() >= 2) {
                    // logical multi-camera found
                    return c2.getCameraId();
                }
            }
        } catch (Exception e) {
            Log.w(TAG, "findLogicalMultiCameraId failed", e);
        }
        return null;
    }
}
