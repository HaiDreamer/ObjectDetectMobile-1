package vn.edu.usth.objectdetectmobile;

import android.os.Bundle;
import android.widget.ImageButton;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;
import androidx.cardview.widget.CardView;

public class ModelPackage extends AppCompatActivity {
    private ImageButton buttonBack;
    private TextView textModel1, textModel2;
    private ImageView bin1, bin2;
    private TextView textModel3, textModel4;
    private ImageView down1, down2;

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

        setupActions();
    }

    private void setupActions() {
        bin1.setOnClickListener(v ->
                Toast.makeText(this, "Xóa " + textModel1.getText(), Toast.LENGTH_SHORT).show()
        );

        bin2.setOnClickListener(v ->
                Toast.makeText(this, "Xóa " + textModel2.getText(), Toast.LENGTH_SHORT).show()
        );

        down1.setOnClickListener(v ->
                Toast.makeText(this, "Tải " + textModel3.getText(), Toast.LENGTH_SHORT).show()
        );

        down2.setOnClickListener(v ->
                Toast.makeText(this, "Tải " + textModel4.getText(), Toast.LENGTH_SHORT).show()
        );
    }
}


