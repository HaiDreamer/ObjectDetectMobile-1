package vn.edu.usth.objectdetectmobile;

import android.app.DownloadManager;
import android.content.Context;
import android.net.Uri;
import android.os.Environment;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

import vn.edu.usth.objectdetectmobile.MainActivity.EnvMode;

public class ModelUtils {

    public static class ModelInfo {
        public String id;
        public String name;
        public String description;
        public String filename;
        public String url;
        public EnvMode relatedMode; // Có thể null nếu là model detection

        public ModelInfo(String id, String name, String description, String filename, String url, EnvMode relatedMode) {
            this.id = id;
            this.name = name;
            this.description = description;
            this.filename = filename;
            this.url = url;
            this.relatedMode = relatedMode;
        }
    }

    // Danh sách các model
    public static List<ModelInfo> getAvailableModels() {
        List<ModelInfo> list = new ArrayList<>();
        
        // Model 1: NearFocus Depth
        list.add(new ModelInfo(
                "depth_NearFocus",
                "NearFocus Depth Model",
                "Depth Anything V2 (Metric Hypersim) - FP16",
                "depth_anything_v2_metric_hypersim_vits_fp16.onnx",
                "https://haidreamer.github.io/models_mobile_app_gp_for_visually_impaired/depth_anything_v2_metric_hypersim_vits_fp16.onnx",
                EnvMode.NearFocus
        ));

        // Model 2: FarFocus Depth
        list.add(new ModelInfo(
                "depth_FarFocus",
                "FarFocus Depth Model",
                "Depth Anything V2 (Metric VKitti) - FP16",
                "depth_anything_v2_metric_vkitti_vits_fp16.onnx",
                "https://haidreamer.github.io/models_mobile_app_gp_for_visually_impaired/depth_anything_v2_metric_vkitti_vits_fp16.onnx",
                EnvMode.FarFocus
        ));

        // Ví dụ Model 3: Object Detection (Placeholder cho tương lai)
        list.add(new ModelInfo(
                "yolo_v8",
                "YOLOv8 Detection",
                "Object Detection Model - Int8",
                "yolov8n_int8.onnx",
                "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.onnx", // Link ví dụ
                null
        ));
        
        // Ví dụ Model 4: Segmentation (Placeholder)
        list.add(new ModelInfo(
                "seg_v8",
                "YOLOv8 Segmentation",
                "Segmentation Model - FP16",
                "yolov8n-seg.onnx",
                "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-seg.onnx", // Link ví dụ
                null
        ));

        return list;
    }

    public static boolean isModelDownloaded(Context context, ModelInfo model) {
        File file = new File(context.getExternalFilesDir(Environment.DIRECTORY_DOWNLOADS), model.filename);
        return file.exists() && file.length() > 0;
    }

    public static void deleteModel(Context context, ModelInfo model) {
        File file = new File(context.getExternalFilesDir(Environment.DIRECTORY_DOWNLOADS), model.filename);
        if (file.exists()) {
            file.delete();
        }
    }

    public static long downloadModel(Context context, ModelInfo model) {
        DownloadManager dm = (DownloadManager) context.getSystemService(Context.DOWNLOAD_SERVICE);
        if (dm == null) return -1;

        DownloadManager.Request req = new DownloadManager.Request(Uri.parse(model.url))
                .setTitle("Downloading " + model.name)
                .setDescription(model.description)
                .setNotificationVisibility(DownloadManager.Request.VISIBILITY_VISIBLE_NOTIFY_COMPLETED)
                .setDestinationInExternalFilesDir(context, Environment.DIRECTORY_DOWNLOADS, model.filename);

        return dm.enqueue(req);
    }
}