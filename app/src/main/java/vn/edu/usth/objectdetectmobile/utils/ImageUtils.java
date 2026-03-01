package vn.edu.usth.objectdetectmobile.utils;

public final class ImageUtils {

    private ImageUtils() {}

    public static int[] resizeNearest(int[] src, int srcW, int srcH, int dstW, int dstH) {
        if (src == null || srcW <= 0 || srcH <= 0 || dstW <= 0 || dstH <= 0) {
            return src;
        }
        if (srcW == dstW && srcH == dstH) {
            return src;
        }

        int[] dst = new int[dstW * dstH];
        for (int y = 0; y < dstH; y++) {
            int sy = Math.min(srcH - 1, (int) ((long) y * srcH / dstH));
            int srcRow = sy * srcW;
            int dstRow = y * dstW;
            for (int x = 0; x < dstW; x++) {
                int sx = Math.min(srcW - 1, (int) ((long) x * srcW / dstW));
                dst[dstRow + x] = src[srcRow + sx];
            }
        }
        return dst;
    }

    public static int[] blurAtSize(int[] src, int srcW, int srcH, int blurSize, int radius) {
        if (src == null || srcW <= 0 || srcH <= 0 || blurSize <= 0 || radius <= 0) {
            return src;
        }

        int[] blurInput = (srcW == blurSize && srcH == blurSize)
                ? src
                : resizeNearest(src, srcW, srcH, blurSize, blurSize);

        int[] blurred = boxBlur(blurInput, blurSize, blurSize, radius);
        if (srcW == blurSize && srcH == blurSize) {
            return blurred;
        }
        return resizeNearest(blurred, blurSize, blurSize, srcW, srcH);
    }

    public static int[] boxBlur(int[] src, int w, int h, int radius) {
        int[] dst = new int[w * h];
        for (int y = 0; y < h; y++) {
            int yMin = Math.max(0, y - radius);
            int yMax = Math.min(h - 1, y + radius);
            for (int x = 0; x < w; x++) {
                int xMin = Math.max(0, x - radius);
                int xMax = Math.min(w - 1, x + radius);

                int count = 0;
                int sumR = 0, sumG = 0, sumB = 0;
                for (int yy = yMin; yy <= yMax; yy++) {
                    int base = yy * w;
                    for (int xx = xMin; xx <= xMax; xx++) {
                        int c = src[base + xx];
                        sumR += (c >> 16) & 0xFF;
                        sumG += (c >> 8) & 0xFF;
                        sumB += c & 0xFF;
                        count++;
                    }
                }
                if (count == 0) count = 1;
                int r = sumR / count;
                int g = sumG / count;
                int b = sumB / count;
                dst[y * w + x] = 0xFF000000 | (r << 16) | (g << 8) | b;
            }
        }
        return dst;
    }
}
