package vn.edu.usth.objectdetectmobile;

public final class ImageUtils {

    private ImageUtils() {}

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
