package vn.edu.usth.objectdetectmobile;

import android.graphics.drawable.Drawable;
import android.os.Bundle;
import android.text.SpannableString;
import android.text.SpannableStringBuilder;
import android.text.Spanned;
import android.text.style.BulletSpan;
import android.text.style.ImageSpan;
import android.widget.ImageButton;
import android.widget.TextView;
import android.text.style.StyleSpan;
import android.graphics.Typeface;

import androidx.appcompat.app.AppCompatActivity;
import androidx.core.content.ContextCompat;

public class Instruction extends AppCompatActivity {
    private ImageButton buttonBack;
    private TextView textView;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.instruction);

        buttonBack = findViewById(R.id.buttonBack);
        buttonBack.setOnClickListener(v -> finish());

        textView = findViewById(R.id.Text);

        // Tạo builder để chứa nhiều đoạn text
        SpannableStringBuilder builder = new SpannableStringBuilder();

        // Step 1 - Tiêu đề in đậm, không có bullet
        SpannableString step1Title = new SpannableString("  Step 1. Download model\n");
        step1Title.setSpan(new StyleSpan(Typeface.BOLD), 2, step1Title.length(), Spanned.SPAN_EXCLUSIVE_EXCLUSIVE);
        Drawable icon1 = ContextCompat.getDrawable(this, R.drawable.ic_cloud2);
        if (icon1 != null) {
            icon1.setBounds(0, 0, icon1.getIntrinsicWidth(), icon1.getIntrinsicHeight());
            ImageSpan imgSpan = new ImageSpan(icon1, ImageSpan.ALIGN_BOTTOM);
            step1Title.setSpan(imgSpan, 0, 1, Spanned.SPAN_EXCLUSIVE_EXCLUSIVE);
        }
        builder.append(step1Title);

        // Mô tả Step 1 có bullet và thụt lề
        SpannableString step1Desc1 = new SpannableString("From Settings, go to Model Package.\n");
        step1Desc1.setSpan(new BulletSpan(40), 0, step1Desc1.length(), Spanned.SPAN_EXCLUSIVE_EXCLUSIVE);
        builder.append(step1Desc1);

        SpannableString step1Desc2 = new SpannableString("Manage your models that need to download.\n\n");
        step1Desc2.setSpan(new BulletSpan(40), 0, step1Desc2.length(), Spanned.SPAN_EXCLUSIVE_EXCLUSIVE);
        builder.append(step1Desc2);

        // Step 2 - Tiêu đề in đậm, không có bullet
        SpannableString step2Title = new SpannableString("  Step 2. Choose model\n");
        step2Title.setSpan(new StyleSpan(Typeface.BOLD), 2, step2Title.length(), Spanned.SPAN_EXCLUSIVE_EXCLUSIVE);
        Drawable icon2 = ContextCompat.getDrawable(this, R.drawable.ic_model2);
        if (icon2 != null) {
            icon2.setBounds(0, 0, icon2.getIntrinsicWidth(), icon2.getIntrinsicHeight());
            ImageSpan imgSpan = new ImageSpan(icon2, ImageSpan.ALIGN_BOTTOM);
            step2Title.setSpan(imgSpan, 0, 1, Spanned.SPAN_EXCLUSIVE_EXCLUSIVE);
        }
        builder.append(step2Title);

        // Mô tả Step 2 có bullet và thụt lề
        SpannableString step2Desc1 = new SpannableString("From Settings, go to Depth Estimation Models.\n");
        step2Desc1.setSpan(new BulletSpan(40), 0, step2Desc1.length(), Spanned.SPAN_EXCLUSIVE_EXCLUSIVE);
        builder.append(step2Desc1);

        SpannableString step2Desc2 = new SpannableString("Select the model and environment that you need.\n\n");
        step2Desc2.setSpan(new BulletSpan(40), 0, step2Desc2.length(), Spanned.SPAN_EXCLUSIVE_EXCLUSIVE);
        builder.append(step2Desc2);

        // Step 3 - Tiêu đề in đậm, không có bullet
        SpannableString step3Title = new SpannableString("  Step 3. View result\n");
        step3Title.setSpan(new StyleSpan(Typeface.BOLD), 2, step3Title.length(), Spanned.SPAN_EXCLUSIVE_EXCLUSIVE);
        Drawable icon3 = ContextCompat.getDrawable(this, R.drawable.ic_camera2);
        if (icon3 != null) {
            icon3.setBounds(0, 0, icon3.getIntrinsicWidth(), icon3.getIntrinsicHeight());
            ImageSpan imgSpan = new ImageSpan(icon3, ImageSpan.ALIGN_BOTTOM);
            step3Title.setSpan(imgSpan, 0, 1, Spanned.SPAN_EXCLUSIVE_EXCLUSIVE);
        }
        builder.append(step3Title);

        // Mô tả Step 3 có bullet và thụt lề
        SpannableString step3Desc1 = new SpannableString("Camera will detect object and notice to you.\n");
        step3Desc1.setSpan(new BulletSpan(40), 0, step3Desc1.length(), Spanned.SPAN_EXCLUSIVE_EXCLUSIVE);
        builder.append(step3Desc1);

        // Gán vào TextView
        textView.setText(builder);
    }
}
