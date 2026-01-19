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

    private static final String PREF_BLUR_ENABLED = "pref_blur_enabled";
    private static final boolean ENABLE_INPUT_BLUR = true; // Default

    private SharedPreferences prefs;
    private boolean stereoAvailable = false;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.settings);

        prefs = DepthCalibrationHelper.getPrefs(this);
        
        // Nhận trạng thái hỗ trợ Stereo từ MainActivity
        stereoAvailable = getIntent().getBooleanExtra("STEREO_AVAILABLE", false);

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
                intent.putExtra("STEREO_AVAILABLE", stereoAvailable);
                startActivity(intent);
            });
        }

        // Setup Blur Switch
        boolean isBlur = prefs.getBoolean(PREF_BLUR_ENABLED, ENABLE_INPUT_BLUR);
        switchBlur.setChecked(isBlur);

        switchBlur.setOnCheckedChangeListener((buttonView, isChecked) -> {
            prefs.edit().putBoolean(PREF_BLUR_ENABLED, isChecked).apply();
            // Toast.makeText(this, isChecked ? "Blur Input ON" : "Blur Input OFF", Toast.LENGTH_SHORT).show();
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
    }
}
