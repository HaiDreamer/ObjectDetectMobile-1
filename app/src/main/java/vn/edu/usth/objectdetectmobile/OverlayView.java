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
    private final Paint maskPaint = new Paint();
    private List<ObjectDetector.Detection> dets = new ArrayList<>();
    private String[] odLabels = new String[0];
    private String[] segLabels = new String[0];
    private int frameW = 1, frameH = 1;

    public OverlayView(Context c, AttributeSet a) {
        super(c, a);
        box.setStyle(Paint.Style.STROKE);
        box.setStrokeWidth(4f);
        box.setAntiAlias(true);
        text.setColor(Color.WHITE);
        text.setTextSize(36f);
        text.setAntiAlias(true);
        maskPaint.setStyle(Paint.Style.FILL);
        maskPaint.setFilterBitmap(true);
    }

    public void setLabels(String[] labels) { this.odLabels = labels != null ? labels : new String[0]; }

    public void setLabels(String[] odLabels, String[] segLabels) {
        this.odLabels = odLabels != null ? odLabels : new String[0];
        this.segLabels = segLabels != null ? segLabels : new String[0];
    }

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
            if (d.mask != null) {
                drawMask(canvas, d, offsetX, offsetY, scale);
            }
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
            String[] labels = labelsForSource(d.source);
            String lab = (d.cls >= 0 && d.cls < labels.length) ? labels[d.cls] : ("cls " + d.cls);
            StringBuilder sb = new StringBuilder();
            sb.append(lab).append(String.format(Locale.US, " %.2f", d.score));
            if (!Float.isNaN(d.depth)) {
                sb.append(String.format(Locale.US, " %.0fcm", d.depth));
            }
            canvas.drawText(sb.toString(), left + 6, Math.max(0, top - 8), text);
        }
    }

    private void drawMask(@NonNull Canvas canvas, @NonNull ObjectDetector.Detection d,
                          float offsetX, float offsetY, float scale) {
        ObjectDetector.Detection.Mask mask = d.mask;
        if (mask == null || mask.width <= 0 || mask.height <= 0) return;

        int color = colorForClass(d.cls);
        int rgb = color & 0x00FFFFFF;
        int w = mask.width;
        int h = mask.height;
        int size = w * h;
        int[] pixels = new int[size];
        byte[] alpha = mask.alpha;
        int edgeAlpha = 200;
        for (int y = 0; y < h; y++) {
            int row = y * w;
            for (int x = 0; x < w; x++) {
                int idx = row + x;
                if ((alpha[idx] & 0xFF) == 0) continue;
                boolean edge = (x == 0 || y == 0 || x == w - 1 || y == h - 1);
                if (!edge) {
                    if ((alpha[idx - 1] & 0xFF) == 0
                            || (alpha[idx + 1] & 0xFF) == 0
                            || (alpha[idx - w] & 0xFF) == 0
                            || (alpha[idx + w] & 0xFF) == 0) {
                        edge = true;
                    }
                }
                if (edge) {
                    pixels[idx] = (edgeAlpha << 24) | rgb;
                }
            }
        }

        Bitmap bmp = Bitmap.createBitmap(pixels, w, h, Bitmap.Config.ARGB_8888);
        float left = offsetX + mask.x * scale;
        float top = offsetY + mask.y * scale;
        float right = offsetX + (mask.x + mask.width) * scale;
        float bottom = offsetY + (mask.y + mask.height) * scale;
        RectF dst = new RectF(left, top, right, bottom);
        canvas.drawBitmap(bmp, null, dst, maskPaint);
    }

    private int colorForClass(int cls) {
        float hue = (cls * 37f) % 360f;
        return Color.HSVToColor(new float[]{hue, 0.7f, 1f});
    }

    private String[] labelsForSource(int source) {
        if (source == ObjectDetector.Detection.SOURCE_SEG && segLabels.length > 0) {
            return segLabels;
        }
        return odLabels;
    }
}
