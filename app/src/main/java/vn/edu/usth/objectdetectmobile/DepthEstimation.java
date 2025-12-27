package vn.edu.usth.objectdetectmobile;

import android.os.Bundle;
import android.widget.ImageButton;
import android.widget.TextView;
import android.widget.LinearLayout;
import androidx.appcompat.app.AppCompatActivity;
import androidx.appcompat.widget.SwitchCompat;
import androidx.core.content.ContextCompat;
import android.content.res.ColorStateList;
import android.graphics.Color;
import android.view.View;

import com.google.android.material.button.MaterialButton;
import com.google.android.material.button.MaterialButtonToggleGroup;

public class DepthEstimation extends AppCompatActivity {

    private LinearLayout layoutMonocular, layoutStereo;
    private SwitchCompat switchIndoor, switchOutdoor;
    private TextView statusModeMono, statusModeStereo;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.depth_estimation); // tên file XML bạn gửi

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
                    // Monocular được chọn → nền xanh, viền xanh
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

                } else if (checkedId == R.id.buttonStereo) {
                    // Stereo được chọn → nền cam, viền cam
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
                }
            }
        });

        // Sự kiện Indoor/Outdoor
        switchIndoor.setOnCheckedChangeListener((buttonView, isChecked) -> {
            if (isChecked) {
                // Nếu Indoor bật → tắt Outdoor
                switchOutdoor.setChecked(false);
            }
            updateStatusMonocular();
        });

        switchOutdoor.setOnCheckedChangeListener((buttonView, isChecked) -> {
            if (isChecked) {
                // Nếu Outdoor bật → tắt Indoor
                switchIndoor.setChecked(false);
            }
            updateStatusMonocular();
        });

        // Custom màu cho switch
        switchIndoor.setThumbTintList(ContextCompat.getColorStateList(this, R.color.switch_thumb2));
        switchIndoor.setTrackTintList(ContextCompat.getColorStateList(this, R.color.switch_track2));
        switchOutdoor.setThumbTintList(ContextCompat.getColorStateList(this, R.color.switch_thumb2));
        switchOutdoor.setTrackTintList(ContextCompat.getColorStateList(this, R.color.switch_track2));
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
