package vn.edu.usth.objectdetectmobile;

import android.app.DownloadManager;
import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;
import android.content.IntentFilter;
import android.os.Build;
import android.os.Bundle;
import android.view.View;
import android.widget.ImageButton;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;

import java.util.ArrayList;
import java.util.List;

public class ModelPackage extends AppCompatActivity {
    private ImageButton buttonBack;
    private TextView textModel1, textModel2;
    private ImageView bin1, bin2;
    private TextView textModel3, textModel4;
    private ImageView down1, down2;
    
    private List<ModelUtils.ModelInfo> allModels;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.model_package);

        buttonBack = findViewById(R.id.buttonBack);

        textModel1 = findViewById(R.id.textModel1);
        bin1 = findViewById(R.id.bin1);

        textModel2 = findViewById(R.id.textModel2);
        bin2 = findViewById(R.id.bin2);

        textModel3 = findViewById(R.id.textModel3);
        down1 = findViewById(R.id.down1);

        textModel4 = findViewById(R.id.textModel4);
        down2 = findViewById(R.id.down2);

        buttonBack.setOnClickListener(v -> finish());

        allModels = ModelUtils.getAvailableModels();
        
        // Đăng ký receiver để cập nhật UI khi tải xong
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            registerReceiver(downloadReceiver, new IntentFilter(DownloadManager.ACTION_DOWNLOAD_COMPLETE), Context.RECEIVER_EXPORTED);
        } else {
            registerReceiver(downloadReceiver, new IntentFilter(DownloadManager.ACTION_DOWNLOAD_COMPLETE));
        }
    }

    @Override
    protected void onResume() {
        super.onResume();
        refreshUI();
    }
    
    @Override
    protected void onDestroy() {
        super.onDestroy();
        unregisterReceiver(downloadReceiver);
    }

    private void refreshUI() {
        List<ModelUtils.ModelInfo> downloaded = new ArrayList<>();
        List<ModelUtils.ModelInfo> notDownloaded = new ArrayList<>();

        for (ModelUtils.ModelInfo m : allModels) {
            if (ModelUtils.isModelDownloaded(this, m)) {
                downloaded.add(m);
            } else {
                notDownloaded.add(m);
            }
        }

        // --- Fill Downloaded Section (Slot 1 & 2) ---
        fillSlot(textModel1, bin1, downloaded, 0, true);
        fillSlot(textModel2, bin2, downloaded, 1, true);

        // --- Fill Available Section (Slot 3 & 4) ---
        fillSlot(textModel3, down1, notDownloaded, 0, false);
        fillSlot(textModel4, down2, notDownloaded, 1, false);
    }

    private void fillSlot(TextView tv, ImageView btn, List<ModelUtils.ModelInfo> list, int index, boolean isDelete) {
        if (index < list.size()) {
            ModelUtils.ModelInfo model = list.get(index);
            tv.setText(model.name);
            tv.setVisibility(View.VISIBLE);
            btn.setVisibility(View.VISIBLE);
            
            // Gỡ listener cũ để tránh duplicate
            btn.setOnClickListener(null);
            
            if (isDelete) {
                btn.setOnClickListener(v -> {
                    ModelUtils.deleteModel(this, model);
                    Toast.makeText(this, "Đã xóa " + model.name, Toast.LENGTH_SHORT).show();
                    refreshUI();
                });
            } else {
                btn.setOnClickListener(v -> {
                    ModelUtils.downloadModel(this, model);
                    Toast.makeText(this, "Đang tải " + model.name + "...", Toast.LENGTH_SHORT).show();
                });
            }
        } else {
            // Slot trống
            if (tv.getId() == R.id.textModel2) {
                tv.setText("YOLOv8 Segmentation");
                tv.setVisibility(View.VISIBLE);
                btn.setVisibility(View.VISIBLE);
                btn.setOnClickListener(v ->
                        Toast.makeText(this, "No model to delete", Toast.LENGTH_SHORT).show()
                );
            } else {
                tv.setVisibility(View.INVISIBLE);
                btn.setVisibility(View.INVISIBLE);
            }
            btn.setOnClickListener(null);
        }
    }
    
    private final BroadcastReceiver downloadReceiver = new BroadcastReceiver() {
        @Override
        public void onReceive(Context context, Intent intent) {
            // Khi có file tải xong, refresh lại list
            refreshUI();
        }
    };
}
