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
        setContentView(R.layout.depth_estimation); // name file XML u send
        
        prefs = DepthCalibrationHelper.getPrefs(this);
        // receive status support Stereo from Settings
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

        // event choose model
        toggleGroup.addOnButtonCheckedListener((group, checkedId, isChecked) -> {
            if (isChecked) {
                if (checkedId == R.id.buttonMonocular) {
                    // Monocular has chosen
                    buttonMonocular.setBackgroundTintList(ColorStateList.valueOf(Color.parseColor("#C2FFB5")));
                    buttonMonocular.setStrokeColor(ColorStateList.valueOf(Color.parseColor("#24C400")));

                    // Stereo reset to white + gray border
                    buttonStereo.setBackgroundTintList(ColorStateList.valueOf(Color.WHITE));
                    buttonStereo.setStrokeColor(ColorStateList.valueOf(Color.parseColor("#EBEBEB")));

                    // show layout Monocular, hide layout Stereo
                    layoutMonocular.setVisibility(View.VISIBLE);
                    layoutStereo.setVisibility(View.GONE);

                    // update status Monocular
                    updateStatusMonocular();
                    
                    // Lưu Prefs
                    prefs.edit().putString(PREF_DEPTH_MODE, "MONO").apply();

                } else if (checkedId == R.id.buttonStereo) {
                    // check hardware
                    if (!stereoAvailable) {
                        Toast.makeText(this, "Thiết bị không hỗ trợ Stereo Camera", Toast.LENGTH_SHORT).show();
                        toggleGroup.check(R.id.buttonMonocular); // Revert to Mono
                        return;
                    }
                    
                    buttonStereo.setBackgroundTintList(ColorStateList.valueOf(Color.parseColor("#F7DFA4")));
                    buttonStereo.setStrokeColor(ColorStateList.valueOf(Color.parseColor("#EDA900")));

                    // Monocular reset to white and gray border
                    buttonMonocular.setBackgroundTintList(ColorStateList.valueOf(Color.WHITE));
                    buttonMonocular.setStrokeColor(ColorStateList.valueOf(Color.parseColor("#EBEBEB")));

                    // show layout Stereo, hide layout Monocular
                    layoutStereo.setVisibility(View.VISIBLE);
                    layoutMonocular.setVisibility(View.GONE);

                    // Reset Indoor/Outdoor when change to Stereo
                    switchIndoor.setChecked(false);
                    switchOutdoor.setChecked(false);

                    // update status Stereo
                    updateStatusStereo();
                    
                    // save Prefs
                    prefs.edit().putString(PREF_DEPTH_MODE, "STEREO").apply();
                }
            }
        });

        // Indoor/Outdoor events
        switchIndoor.setOnCheckedChangeListener((buttonView, isChecked) -> {
            if (isChecked) {
                // if Indoor open, turn off Outdoor
                switchOutdoor.setChecked(false);
            }
            // safe Prefs (if open -> INDOOR, if turn off Outdoor also turn off -> default INDOOR)
            if (isChecked) prefs.edit().putString(PREF_ENV_MODE, "INDOOR").apply();
            updateStatusMonocular();
        });

        switchOutdoor.setOnCheckedChangeListener((buttonView, isChecked) -> {
            if (isChecked) {
                // if Outdoor open, turn off Indoor
                switchIndoor.setChecked(false);
            }
            // safe Prefs
            if (isChecked) prefs.edit().putString(PREF_ENV_MODE, "OUTDOOR").apply();
            updateStatusMonocular();
        });

        // Custom color for switch
        switchIndoor.setThumbTintList(ContextCompat.getColorStateList(this, R.color.switch_thumb2));
        switchIndoor.setTrackTintList(ContextCompat.getColorStateList(this, R.color.switch_track2));
        switchOutdoor.setThumbTintList(ContextCompat.getColorStateList(this, R.color.switch_thumb2));
        switchOutdoor.setTrackTintList(ContextCompat.getColorStateList(this, R.color.switch_track2));

        // --- SETUP UI STATE FROM PREFS ---
        // Call this function after setting the listener so that the UI logic in the listener is activated
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
        String envMode = prefs.getString(PREF_ENV_MODE, "INDOOR");
        if ("OUTDOOR".equals(envMode)) {
            switchOutdoor.setChecked(true);
            switchIndoor.setChecked(false);
        } else {
            switchIndoor.setChecked(true);
            switchOutdoor.setChecked(false);
        }
    }

    // update status for Monocular
    private void updateStatusMonocular() {
        StringBuilder mode = new StringBuilder(" Monocular");
        if (switchIndoor.isChecked()) {
            mode.append(" . Indoor");
        } else if (switchOutdoor.isChecked()) {
            mode.append(" . Outdoor");
        }
        statusModeMono.setText(mode.toString());
    }

    // update status for Stereo
    private void updateStatusStereo() {
        statusModeStereo.setText(" Stereo");
    }
}
