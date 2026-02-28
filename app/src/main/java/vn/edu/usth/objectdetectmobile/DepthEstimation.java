package vn.edu.usth.objectdetectmobile;

import android.os.Bundle;
import android.content.SharedPreferences;
import android.widget.ImageButton;
import android.widget.TextView;
import android.widget.LinearLayout;
import androidx.appcompat.app.AppCompatActivity;
import androidx.appcompat.widget.SwitchCompat;
import androidx.core.content.ContextCompat;
import android.content.res.ColorStateList;
import android.graphics.Color;
import android.view.View;
import android.widget.Toast;

import com.google.android.material.button.MaterialButton;
import com.google.android.material.button.MaterialButtonToggleGroup;

public class DepthEstimation extends AppCompatActivity {

    private LinearLayout layoutMonocular, layoutStereo;
    private SwitchCompat switchIndoor, switchOutdoor;
    private TextView statusModeMono, statusModeStereo;
    
    private SharedPreferences prefs;
    private static final String PREF_ENV_MODE = "pref_env_mode";
    private static final String PREF_DEPTH_MODE = "pref_depth_mode"; // MONO or STEREO
    
    private boolean stereoAvailable = false;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.depth_estimation); // tên file XML bạn gửi
        
        prefs = DepthCalibrationHelper.getPrefs(this);
        // Nhận trạng thái hỗ trợ Stereo từ Settings
        stereoAvailable = getIntent().getBooleanExtra("STEREO_AVAILABLE", false);

        layoutMonocular = findViewById(R.id.layoutMonocular);
        layoutStereo = findViewById(R.id.layoutStereo);
        switchIndoor = findViewById(R.id.switchIndoor);
        switchOutdoor = findViewById(R.id.switchOutdoor);
        statusModeMono = findViewById(R.id.statusModeMono);
        statusModeStereo = findViewById(R.id.statusModeStereo);

        MaterialButtonToggleGroup toggleGroup = findViewById(R.id.buttonToggleModels);
        MaterialButton buttonMonocular = findViewById(R.id.buttonMonocular);
        MaterialButton buttonStereo = findViewById(R.id.buttonStereo);
        ImageButton buttonBack = findViewById(R.id.buttonBack);

        buttonBack.setOnClickListener(v -> finish());

        // Sự kiện chọn model
        toggleGroup.addOnButtonCheckedListener((group, checkedId, isChecked) -> {
            if (isChecked) {
                if (checkedId == R.id.buttonMonocular) {
                    // Monocular được chọn
                    buttonMonocular.setBackgroundTintList(ColorStateList.valueOf(Color.parseColor("#C2FFB5")));
                    buttonMonocular.setStrokeColor(ColorStateList.valueOf(Color.parseColor("#24C400")));

                    // Stereo reset về trắng + viền xám
                    buttonStereo.setBackgroundTintList(ColorStateList.valueOf(Color.WHITE));
                    buttonStereo.setStrokeColor(ColorStateList.valueOf(Color.parseColor("#EBEBEB")));

                    // Hiện layout Monocular, ẩn layout Stereo
                    layoutMonocular.setVisibility(View.VISIBLE);
                    layoutStereo.setVisibility(View.GONE);

                    // Cập nhật status Monocular
                    updateStatusMonocular();
                    
                    // Lưu Prefs
                    prefs.edit().putString(PREF_DEPTH_MODE, "MONO").apply();

                } else if (checkedId == R.id.buttonStereo) {
                    // Kiểm tra phần cứng
                    if (!stereoAvailable) {
                        Toast.makeText(this, "Thiết bị không hỗ trợ Stereo Camera", Toast.LENGTH_SHORT).show();
                        toggleGroup.check(R.id.buttonMonocular); // Revert về Mono
                        return;
                    }
                    
                    buttonStereo.setBackgroundTintList(ColorStateList.valueOf(Color.parseColor("#F7DFA4")));
                    buttonStereo.setStrokeColor(ColorStateList.valueOf(Color.parseColor("#EDA900")));

                    // Monocular reset về trắng + viền xám
                    buttonMonocular.setBackgroundTintList(ColorStateList.valueOf(Color.WHITE));
                    buttonMonocular.setStrokeColor(ColorStateList.valueOf(Color.parseColor("#EBEBEB")));

                    // Hiện layout Stereo, ẩn layout Monocular
                    layoutStereo.setVisibility(View.VISIBLE);
                    layoutMonocular.setVisibility(View.GONE);

                    // Reset Indoor/Outdoor khi chuyển sang Stereo
                    switchIndoor.setChecked(false);
                    switchOutdoor.setChecked(false);

                    // Cập nhật status Stereo
                    updateStatusStereo();
                    
                    // Lưu Prefs
                    prefs.edit().putString(PREF_DEPTH_MODE, "STEREO").apply();
                }
            }
        });

        // Sự kiện Indoor/Outdoor
        switchIndoor.setOnCheckedChangeListener((buttonView, isChecked) -> {
            if (isChecked) {
                // Nếu Indoor bật → tắt Outdoor
                switchOutdoor.setChecked(false);
            }
            // Lưu Prefs (Nếu bật -> INDOOR, nếu tắt mà Outdoor cũng tắt -> mặc định INDOOR)
            if (isChecked) prefs.edit().putString(PREF_ENV_MODE, "INDOOR").apply();
            updateStatusMonocular();
        });

        switchOutdoor.setOnCheckedChangeListener((buttonView, isChecked) -> {
            if (isChecked) {
                // Nếu Outdoor bật → tắt Indoor
                switchIndoor.setChecked(false);
            }
            // Lưu Prefs
            if (isChecked) prefs.edit().putString(PREF_ENV_MODE, "OUTDOOR").apply();
            updateStatusMonocular();
        });

        // Custom màu cho switch
        switchIndoor.setThumbTintList(ContextCompat.getColorStateList(this, R.color.switch_thumb2));
        switchIndoor.setTrackTintList(ContextCompat.getColorStateList(this, R.color.switch_track2));
        switchOutdoor.setThumbTintList(ContextCompat.getColorStateList(this, R.color.switch_thumb2));
        switchOutdoor.setTrackTintList(ContextCompat.getColorStateList(this, R.color.switch_track2));

        // --- SETUP UI STATE FROM PREFS ---
        // Gọi hàm này SAU KHI đã set listener để logic UI trong listener được kích hoạt
        setupInitialState(toggleGroup, buttonMonocular, buttonStereo);
    }
    
    private void setupInitialState(MaterialButtonToggleGroup group, MaterialButton btnMono, MaterialButton btnStereo) {
        // 1. Load Depth Mode (Mono/Stereo)
        String depthMode = prefs.getString(PREF_DEPTH_MODE, "MONO");
        if ("STEREO".equals(depthMode) && stereoAvailable) {
            group.check(R.id.buttonStereo);
        } else {
            group.check(R.id.buttonMonocular);
        }
        
        // 2. Load Env Mode (Indoor/Outdoor)
        String envMode = prefs.getString(PREF_ENV_MODE, "OUTDOOR");
        if ("OUTDOOR".equals(envMode)) {
            switchOutdoor.setChecked(true);
            switchIndoor.setChecked(false);
        } else {
            switchIndoor.setChecked(true);
            switchOutdoor.setChecked(false);
        }
    }

    // Cập nhật status cho Monocular
    private void updateStatusMonocular() {
        StringBuilder mode = new StringBuilder(" Monocular");
        if (switchIndoor.isChecked()) {
            mode.append(" . Indoor");
        } else if (switchOutdoor.isChecked()) {
            mode.append(" . Outdoor");
        }
        statusModeMono.setText(mode.toString());
    }

    // Cập nhật status cho Stereo
    private void updateStatusStereo() {
        statusModeStereo.setText(" Stereo");
    }
}
