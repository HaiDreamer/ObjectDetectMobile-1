package vn.edu.usth.objectdetectmobile;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.Rect;
import android.graphics.RectF;
import android.graphics.Typeface;
import android.util.AttributeSet;
import android.view.View;

import androidx.annotation.NonNull;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

public class OverlayView extends View {
    private final Paint boxPaint = new Paint();
    private final Paint textPaint = new Paint();
    private final Paint labelBgPaint = new Paint();
    private final Paint maskPaint = new Paint();
    private final Rect textBounds = new Rect();

    private List<ObjectDetector.Detection> dets = new ArrayList<>();
    private String[] odLabels = new String[0];
    private String[] segLabels = new String[0];
    private int frameW = 1, frameH = 1;

    private static final float BOX_STROKE = 3.5f;
    private static final float LABEL_TEXT_SIZE = 34f;
    private static final float LABEL_PAD_X = 10f;
    private static final float LABEL_PAD_Y = 6f;
    private static final float LABEL_GAP = 8f;

    private static final int[] CLASS_PALETTE = new int[]{
            Color.rgb(255, 230, 0),
            Color.rgb(0, 255, 120),
            Color.rgb(0, 190, 255),
            Color.rgb(255, 165, 0),
            Color.rgb(170, 255, 0),
            Color.rgb(255, 255, 130),
            Color.rgb(90, 255, 90),
            Color.rgb(0, 255, 80),
            Color.rgb(255, 230, 0),
            Color.rgb(180, 0, 255),
            Color.rgb(120, 255, 40),
            Color.rgb(80, 255, 60)
    };

    public OverlayView(Context c, AttributeSet a) {
        super(c, a);

        boxPaint.setStyle(Paint.Style.STROKE);
        boxPaint.setStrokeWidth(BOX_STROKE);
        boxPaint.setAntiAlias(true);

        textPaint.setColor(Color.BLACK);
        textPaint.setTextSize(LABEL_TEXT_SIZE);
        textPaint.setAntiAlias(true);
        textPaint.setTypeface(Typeface.DEFAULT_BOLD);

        labelBgPaint.setStyle(Paint.Style.FILL);
        labelBgPaint.setAntiAlias(true);

        maskPaint.setStyle(Paint.Style.FILL);
        maskPaint.setAntiAlias(true);
        maskPaint.setFilterBitmap(true);
    }

    public void setLabels(String[] labels) {
        this.odLabels = labels != null ? labels : new String[0];
    }

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

    @Override
    protected void onDraw(@NonNull Canvas canvas) {
        super.onDraw(canvas);
        int vw = getWidth();
        int vh = getHeight();
        float scale = Math.min(vw / (float) frameW, vh / (float) frameH);
        float offsetX = (vw - frameW * scale) / 2f;
        float offsetY = (vh - frameH * scale) / 2f;

        for (ObjectDetector.Detection d : dets) {
            String label = resolveLabel(d);
            int color = colorForDetection(d, label);
            boolean isSeg = d.source == ObjectDetector.Detection.SOURCE_SEG;

            if (d.mask != null) {
                drawMask(canvas, d, color, offsetX, offsetY, scale);
            }

            float left = offsetX + d.x1 * scale;
            float top = offsetY + d.y1 * scale;
            float right = offsetX + d.x2 * scale;
            float bottom = offsetY + d.y2 * scale;

            if (!isSeg) {
                boxPaint.setColor(color);
                canvas.drawRect(left, top, right, bottom, boxPaint);
            }

            String labelText = buildLabelText(label, d.score, d.depth);
            float labelLeft = left;
            float labelTop = top;
            if (isSeg && d.mask != null) {
                labelLeft = offsetX + d.mask.x * scale;
                labelTop = offsetY + d.mask.y * scale;
            }
            drawLabel(canvas, labelText, labelLeft, labelTop, color);
        }
    }

    private String resolveLabel(@NonNull ObjectDetector.Detection d) {
        String[] labels = labelsForSource(d.source);
        if (d.cls >= 0 && d.cls < labels.length) {
            return labels[d.cls];
        }
        return "cls " + d.cls;
    }

    private String buildLabelText(String label, float score, float depthCm) {
        StringBuilder sb = new StringBuilder();
        sb.append(normalizeLabel(label))
                .append(String.format(Locale.US, " %.2f", score));
        if (!Float.isNaN(depthCm) && depthCm > 0f) {
            sb.append(String.format(Locale.US, " %.2fm", depthCm / 100f));
        }
        return sb.toString();
    }

    private void drawLabel(@NonNull Canvas canvas, @NonNull String text,
                           float boxLeft, float boxTop, int color) {
        textPaint.getTextBounds(text, 0, text.length(), textBounds);
        Paint.FontMetrics fm = textPaint.getFontMetrics();

        float textW = textPaint.measureText(text);
        float textH = fm.descent - fm.ascent;

        float left = boxLeft;
        float top = boxTop - textH - LABEL_GAP - 2f * LABEL_PAD_Y;
        if (top < 0f) {
            top = boxTop + LABEL_GAP;
        }
        float right = left + textW + 2f * LABEL_PAD_X;
        float bottom = top + textH + 2f * LABEL_PAD_Y;

        labelBgPaint.setColor(withAlpha(color, 235));
        canvas.drawRoundRect(new RectF(left, top, right, bottom), 6f, 6f, labelBgPaint);

        float textX = left + LABEL_PAD_X;
        float textY = top + LABEL_PAD_Y - fm.ascent;
        canvas.drawText(text, textX, textY, textPaint);
    }

    private void drawMask(@NonNull Canvas canvas, @NonNull ObjectDetector.Detection d,
                          int color,
                          float offsetX, float offsetY, float scale) {
        ObjectDetector.Detection.Mask mask = d.mask;
        if (mask == null || mask.width <= 0 || mask.height <= 0) {
            return;
        }

        int rgb = color & 0x00FFFFFF;
        int w = mask.width;
        int h = mask.height;
        int[] pixels = new int[w * h];
        byte[] alpha = mask.alpha;

        for (int y = 0; y < h; y++) {
            int row = y * w;
            for (int x = 0; x < w; x++) {
                int idx = row + x;
                int raw = alpha[idx] & 0xFF;
                if (raw == 0) continue;

                boolean edge = (x == 0 || y == 0 || x == w - 1 || y == h - 1);
                if (!edge) {
                    if ((alpha[idx - 1] & 0xFF) == 0
                            || (alpha[idx + 1] & 0xFF) == 0
                            || (alpha[idx - w] & 0xFF) == 0
                            || (alpha[idx + w] & 0xFF) == 0) {
                        edge = true;
                    }
                }
                int fillAlpha = 45 + (raw * 130) / 255;
                int a = edge ? 225 : fillAlpha;
                pixels[idx] = (a << 24) | rgb;
            }
        }

        Bitmap bmp = Bitmap.createBitmap(pixels, w, h, Bitmap.Config.ARGB_8888);
        float left = offsetX + mask.x * scale;
        float top = offsetY + mask.y * scale;
        float right = offsetX + (mask.x + mask.width) * scale;
        float bottom = offsetY + (mask.y + mask.height) * scale;
        canvas.drawBitmap(bmp, null, new RectF(left, top, right, bottom), maskPaint);
    }

    private int colorForDetection(@NonNull ObjectDetector.Detection d, @NonNull String label) {
        String lower = label.toLowerCase(Locale.US);

        if (d.source == ObjectDetector.Detection.SOURCE_SEG) {
            if (lower.contains("stairs")) return Color.rgb(180, 0, 255);
            if (lower.contains("crosswalk")) return Color.rgb(255, 230, 0);
            if (lower.contains("sidewalk")) return Color.rgb(120, 255, 40);
            if (lower.contains("tree-lined") || lower.contains("tree line")) return Color.rgb(80, 255, 60);
        } else {
            if (lower.contains("person")) return Color.rgb(90, 255, 90);
            if (lower.contains("bicycle")) return Color.rgb(0, 190, 255);
            if (lower.contains("motocycle") || lower.contains("motorcycle")) return Color.rgb(170, 255, 0);
            if (lower.contains("car") || lower.contains("truck") || lower.contains("bus")) {
                return Color.rgb(255, 230, 0);
            }
            if (lower.contains("tree")) return Color.rgb(0, 255, 120);
            if (lower.contains("pole")) return Color.rgb(255, 165, 0);
            if (lower.contains("sign")) return Color.rgb(255, 255, 130);
        }

        int idx = Math.abs(d.cls) % CLASS_PALETTE.length;
        return CLASS_PALETTE[idx];
    }

    private String normalizeLabel(String label) {
        if (label == null || label.isEmpty()) return "object";
        return label.replace('_', ' ').trim().toLowerCase(Locale.US);
    }

    private static int withAlpha(int rgb, int alpha) {
        return (Math.max(0, Math.min(255, alpha)) << 24) | (rgb & 0x00FFFFFF);
    }

    private String[] labelsForSource(int source) {
        if (source == ObjectDetector.Detection.SOURCE_SEG && segLabels.length > 0) {
            return segLabels;
        }
        return odLabels;
    }
}
