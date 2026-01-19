package vn.edu.usth.objectdetectmobile;

import android.content.Context;
import android.graphics.*;
import android.util.AttributeSet;
import android.view.View;

import androidx.annotation.NonNull;

import java.util.*;
import java.util.Locale;

public class OverlayView extends View {
    private final Paint box = new Paint();
    private final Paint text = new Paint();
    private List<ObjectDetector.Detection> dets = new ArrayList<>();
    private String[] labels = new String[0];
    private int frameW = 1, frameH = 1;

    public OverlayView(Context c, AttributeSet a) {
        super(c, a);
        box.setStyle(Paint.Style.STROKE);
        box.setStrokeWidth(4f);
        box.setAntiAlias(true);
        text.setColor(Color.WHITE);
        text.setTextSize(36f);
        text.setAntiAlias(true);
    }

    public void setLabels(String[] labels) { this.labels = labels; }

    public void setDetections(List<ObjectDetector.Detection> dets, int frameW, int frameH) {
        this.dets = dets != null ? dets : new ArrayList<>();
        this.frameW = Math.max(1, frameW);
        this.frameH = Math.max(1, frameH);
        invalidate();
    }

    public void setDetections(List<ObjectDetector.Detection> dets) {
        setDetections(dets, frameW, frameH);
    }

    @Override protected void onDraw(@NonNull Canvas canvas) {
        super.onDraw(canvas);
        int vw = getWidth(), vh = getHeight();
        float scale = Math.min(vw / (float) frameW, vh / (float) frameH);
        float offsetX = (vw - frameW * scale) / 2f;
        float offsetY = (vh - frameH * scale) / 2f;
        for (ObjectDetector.Detection d : dets) {
            if (Float.isNaN(d.depth) || d.depth <= 0) {
                box.setColor(Color.WHITE); // Mặc định (không có depth)
            } else if (d.depth < 150) {
                box.setColor(Color.RED);   // Gần (< 1.5m)
            } else if (d.depth < 300) {
                box.setColor(Color.YELLOW); // Trung bình (1.5m - 3m)
            } else {
                box.setColor(Color.GREEN); // Xa (> 3m)
            }
            float left = offsetX + d.x1 * scale;
            float top = offsetY + d.y1 * scale;
            float right = offsetX + d.x2 * scale;
            float bottom = offsetY + d.y2 * scale;
            canvas.drawRect(left, top, right, bottom, box);
            String lab = (d.cls >= 0 && d.cls < labels.length) ? labels[d.cls] : ("cls " + d.cls);
            StringBuilder sb = new StringBuilder();
            sb.append(lab).append(String.format(Locale.US, " %.2f", d.score));
            if (!Float.isNaN(d.depth)) {
                sb.append(String.format(Locale.US, " %.0fcm", d.depth));
            }
            canvas.drawText(sb.toString(), left + 6, Math.max(0, top - 8), text);
        }
    }
}
