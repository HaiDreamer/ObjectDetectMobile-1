package vn.edu.usth.objectdetectmobile;

import android.content.SharedPreferences;
import android.os.Bundle;
import android.widget.ImageButton;
import android.widget.Toast;
import android.content.Intent;

import androidx.appcompat.app.AppCompatActivity;
import androidx.appcompat.widget.SwitchCompat;
import androidx.cardview.widget.CardView;
import androidx.core.content.ContextCompat;

import vn.edu.usth.objectdetectmobile.MainActivity.EnvMode;

public class Settings extends AppCompatActivity {

    private static final String DEPTH_DOWNLOAD_INDOOR_URL =
            "https://haidreamer.github.io/models_mobile_app_gp_for_visually_impaired/depth_anything_v2_metric_hypersim_vits_fp16.onnx";
    private static final String DEPTH_DOWNLOAD_OUTDOOR_URL =
            "https://haidreamer.github.io/models_mobile_app_gp_for_visually_impaired/depth_anything_v2_metric_vkitti_vits_fp16.onnx";
    private static final String PREF_DEPTH_MODEL_INDOOR_PATH = "pref_depth_model_indoor_path";
    private static final String PREF_DEPTH_MODEL_OUTDOOR_PATH = "pref_depth_model_outdoor_path";
    private static final String PREF_LAST_DEPTH_DOWNLOAD_ID = "pref_last_depth_download_id";
    private static final String PREF_LAST_DEPTH_DOWNLOAD_MODE = "pref_last_depth_download_mode";
    private static final String DEPTH_MODEL_PREFS = "depth_models";
    private static final String PREF_ENV_MODE = "pref_env_mode";

    private SharedPreferences depthModelPrefs;
    private SharedPreferences prefs;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.settings);

        depthModelPrefs = getSharedPreferences(DEPTH_MODEL_PREFS, MODE_PRIVATE);
        prefs = DepthCalibrationHelper.getPrefs(this);

        ImageButton buttonBack = findViewById(R.id.buttonBack);
        SwitchCompat switchBlur = findViewById(R.id.switchBlur);
        CardView depthCard = findViewById(R.id.DepthEstimation);
        CardView packageCard = findViewById(R.id.ModelPackage);
        CardView instructionCard = findViewById(R.id.Instruction);
        CardView blurCard = findViewById(R.id.BlurInput);

        buttonBack.setOnClickListener(v -> finish());

        if (packageCard != null) {
            packageCard.setOnClickListener(v -> {
                Intent intent = new Intent(Settings.this, ModelPackage.class);
                startActivity(intent);
            });
        }

        if (instructionCard != null) {
            instructionCard.setOnClickListener(v -> {
                Intent intent = new Intent(Settings.this, Instruction.class);
                startActivity(intent);
            });
        }

        if (depthCard != null) {
            depthCard.setOnClickListener(v -> {
                Intent intent = new Intent(Settings.this, DepthEstimation.class);
                startActivity(intent);
            });
        }

        switchBlur.setOnCheckedChangeListener((buttonView, isChecked) -> {
            Toast.makeText(this, isChecked ? "Blur Input ON" : "Blur Input OFF", Toast.LENGTH_SHORT).show();
        });
        switchBlur.setThumbTintList(ContextCompat.getColorStateList(this, R.color.switch_thumb1));
        switchBlur.setTrackTintList(ContextCompat.getColorStateList(this, R.color.switch_track1));

        if (blurCard != null) {
            blurCard.setOnClickListener(v -> {
                if (switchBlur != null) {
                    switchBlur.toggle();
                }
            });
        }
        /*
        if (depthCard != null) {
            depthCard.setOnClickListener(v -> showEnvChoiceDialog());
        }
        if (packageCard != null) {
            packageCard.setOnClickListener(v -> showModelActionsDialog());
        }
        if (instructionCard != null) {
            instructionCard.setOnClickListener(v ->
                    Toast.makeText(this, "Instructions coming soon", Toast.LENGTH_SHORT).show());
        }

        TextView titleSettings = findViewById(R.id.titleSettings);
        titleSettings.setText("Settings");
    }

    private void showEnvChoiceDialog() {
        String[] options = new String[]{"Indoor model", "Outdoor model"};
        new AlertDialog.Builder(this)
                .setTitle("Chọn chế độ độ sâu")
                .setItems(options, (dialog, which) -> {
                    EnvMode mode = (which == 1) ? EnvMode.OUTDOOR : EnvMode.INDOOR;
                    setEnvMode(mode);
                })
                .show();
    }

    private void showModelActionsDialog() {
        String[] actions = new String[]{
                "Tải model Indoor",
                "Tải model Outdoor",
                "Xóa model Indoor",
                "Xóa model Outdoor"
        };
        new AlertDialog.Builder(this)
                .setTitle("Model package")
                .setItems(actions, (dialog, which) -> {
                    switch (which) {
                        case 0:
                            startDepthModelDownload(DEPTH_DOWNLOAD_INDOOR_URL, EnvMode.INDOOR);
                            break;
                        case 1:
                            startDepthModelDownload(DEPTH_DOWNLOAD_OUTDOOR_URL, EnvMode.OUTDOOR);
                            break;
                        case 2:
                            deleteDepthModel(EnvMode.INDOOR);
                            break;
                        case 3:
                            deleteDepthModel(EnvMode.OUTDOOR);
                            break;
                        default:
                            break;
                    }
                })
                .show();
    }

    private void setEnvMode(EnvMode mode) {
        if (prefs != null) {
            prefs.edit().putString(PREF_ENV_MODE, mode.name()).apply();
        }
        boolean available = DepthEstimator.isModelAvailable(this, mode);
        Toast.makeText(
                this,
                "Đã chọn " + (mode == EnvMode.OUTDOOR ? "Outdoor" : "Indoor")
                        + (available ? "" : " (chưa có model, hãy tải ở Model Package)"),
                Toast.LENGTH_LONG
        ).show();
    }

    private void startDepthModelDownload(String url, EnvMode mode) {
        DownloadManager dm = (DownloadManager) getSystemService(DOWNLOAD_SERVICE);
        if (dm == null) {
            Toast.makeText(this, "DownloadManager not available", Toast.LENGTH_LONG).show();
            return;
        }

        File outFile = getDepthModelFileForMode(mode);
        if (outFile == null) {
            Toast.makeText(this, "No external files dir for downloads", Toast.LENGTH_LONG).show();
            return;
        }

        String fileName = outFile.getName();

        try {
            DownloadManager.Request req = new DownloadManager.Request(Uri.parse(url))
                    .setTitle("Downloading " + fileName)
                    .setDescription("Depth model for " + (mode == EnvMode.OUTDOOR ? "Outdoor" : "Indoor"))
                    .setNotificationVisibility(
                            DownloadManager.Request.VISIBILITY_VISIBLE_NOTIFY_COMPLETED)
                    .setDestinationInExternalFilesDir(
                            this,
                            Environment.DIRECTORY_DOWNLOADS,
                            fileName
                    );

            long id = dm.enqueue(req);

            String keyPath = (mode == EnvMode.OUTDOOR)
                    ? PREF_DEPTH_MODEL_OUTDOOR_PATH
                    : PREF_DEPTH_MODEL_INDOOR_PATH;

            if (depthModelPrefs != null) {
                depthModelPrefs.edit()
                        .putLong(PREF_LAST_DEPTH_DOWNLOAD_ID, id)
                        .putString(PREF_LAST_DEPTH_DOWNLOAD_MODE, mode.name())
                        .putString(keyPath, outFile.getAbsolutePath())
                        .apply();
            }

            Toast.makeText(this, "Downloading depth model...", Toast.LENGTH_SHORT).show();
        } catch (Throwable t) {
            Toast.makeText(this, "Download failed: " + t.getMessage(), Toast.LENGTH_LONG).show();
        }
    }

    private void deleteDepthModel(EnvMode mode) {
        File f = getDepthModelFileForMode(mode);
        boolean deleted = false;
        if (f != null && f.exists()) {
            deleted = f.delete();
        }
        String keyPath = (mode == EnvMode.OUTDOOR)
                ? PREF_DEPTH_MODEL_OUTDOOR_PATH
                : PREF_DEPTH_MODEL_INDOOR_PATH;
        if (depthModelPrefs != null) {
            depthModelPrefs.edit()
                    .remove(keyPath)
                    .apply();
        }
        Toast.makeText(
                this,
                (deleted ? "Đã xóa " : "Không tìm thấy ") +
                        (mode == EnvMode.OUTDOOR ? "Outdoor" : "Indoor") + " model",
                Toast.LENGTH_SHORT
        ).show();
    }

    private File getDepthModelFileForMode(EnvMode mode) {
        String fileName = (mode == EnvMode.OUTDOOR)
                ? "depth_anything_v2_metric_vkitti_vits_fp16.onnx"
                : "depth_anything_v2_metric_hypersim_vits_fp16.onnx";

        File dir = getExternalFilesDir(Environment.DIRECTORY_DOWNLOADS);
        if (dir == null) return null;
        return new File(dir, fileName);*/
    }
}
