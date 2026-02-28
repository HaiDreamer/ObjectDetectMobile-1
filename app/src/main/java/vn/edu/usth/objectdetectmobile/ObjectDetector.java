package vn.edu.usth.objectdetectmobile;

import android.content.Context;
import android.util.Log;
import androidx.annotation.NonNull;
import ai.onnxruntime.*;

import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.nio.FloatBuffer;
import java.util.*;
import static java.lang.Math.*;

public class ObjectDetector implements AutoCloseable {
    private static final String TAG = "ObjectDetector";
    public static class Detection {
        public static final int SOURCE_OD = 0;
        public static final int SOURCE_SEG = 1;

        public static class Mask {
            public final byte[] alpha;
            public final int width;
            public final int height;
            public final int x;
            public final int y;

            public Mask(byte[] alpha, int width, int height, int x, int y) {
                this.alpha = alpha;
                this.width = width;
                this.height = height;
                this.x = x;
                this.y = y;
            }
        }

        public final float x1,y1,x2,y2,score,depth;
        public final int cls;       // class id
        public final int source;    // source model id
        public final float[] maskCoeffs;
        public final Mask mask;
        public Detection(float x1,float y1,float x2,float y2,float score,int cls){
            this(x1,y1,x2,y2,score,cls,Float.NaN, SOURCE_OD, null, null);
        }
        public Detection(float x1,float y1,float x2,float y2,float score,int cls,float depth){
            this(x1,y1,x2,y2,score,cls,depth, SOURCE_OD, null, null);
        }
        public Detection(float x1,float y1,float x2,float y2,float score,int cls,
                         float depth, int source, float[] maskCoeffs, Mask mask){
            this.x1=x1; this.y1=y1; this.x2=x2; this.y2=y2;
            this.score=score; this.cls=cls; this.depth=depth; this.source=source;
            this.maskCoeffs = maskCoeffs;
            this.mask = mask;
        }
        public Detection withDepth(float depthValue){
            return new Detection(x1,y1,x2,y2,score,cls,depthValue, source, maskCoeffs, mask);
        }
        public Detection withMask(Mask maskValue){
            return new Detection(x1,y1,x2,y2,score,cls,depth, source, null, maskValue);
        }
    }

    private final OrtEnvironment env;
    private final OrtSession session;
    private final int inputW = 640, inputH = 640;
    private final float confThresh = 0.25f, iouThresh = 0.45f;
    private final float maskThreshold = 0.5f;
    private final String inputName;
    private final int sourceId;

    public ObjectDetector(@NonNull Context ctx) throws OrtException {
        this(ctx, null, Detection.SOURCE_OD);
    }

    public ObjectDetector(@NonNull Context ctx, String assetName, int sourceId) throws OrtException {
        env = OrtEnvironment.getEnvironment();
        List<String> modelCandidates;
        if (assetName == null || assetName.isEmpty()) {
            modelCandidates = Util.buildModelCandidates(ctx,
                    "bestseg.onnx",
                    "best.onnx",
                    "yolov8m_compatible.onnx");
        } else {
            modelCandidates = Util.buildModelCandidates(ctx, assetName);
        }
        OrtSession.SessionOptions so = new OrtSession.SessionOptions();
        OrtSession created;
        String selectedAsset = null;
        OrtException lastOrtError = null;
        RuntimeException lastRuntimeError = null;

        created = null;
        for (String candidate : modelCandidates) {
            if (!Util.assetExists(ctx, candidate)) continue;
            String modelPath;
            try {
                modelPath = Util.cacheAsset(ctx, candidate);
                try {
                    created = env.createSession(modelPath, so);
                } catch (OrtException firstErr) {
                    // Cached file may be truncated; delete and retry once.
                    Util.deleteCachedAsset(ctx, candidate);
                    modelPath = Util.cacheAsset(ctx, candidate);
                    created = env.createSession(modelPath, so);
                }
                selectedAsset = candidate;
                break;
            } catch (OrtException e) {
                lastOrtError = e;
            } catch (RuntimeException e) {
                lastRuntimeError = e;
            }
        }

        if (created == null) {
            if (lastOrtError != null) throw lastOrtError;
            if (lastRuntimeError != null) throw lastRuntimeError;
            throw new RuntimeException("No model asset found");
        }

        this.sourceId = sourceIdFor(selectedAsset, sourceId);
        session = created;
        inputName = session.getInputInfo().keySet().iterator().next();
        Log.i(TAG, "Loaded detector model: " + selectedAsset);
    }

    public List<Detection> detect(int[] argb, int srcW, int srcH) throws OrtException {
        Letterbox lb = letterbox(argb, srcW, srcH);
        float[] chw = toCHW(lb.rgb, inputW, inputH);
        OnnxTensor input = OnnxTensor.createTensor(env, FloatBuffer.wrap(chw),
                new long[]{1,3,inputH,inputW});

        try (OrtSession.Result out = session.run(Collections.singletonMap(inputName, input))) {
            if (out.size() >= 2) {
                return parseSeg(out.get(0), out.get(1), lb, srcW, srcH);
            }
            return parseDet(out.get(0), lb.scale, lb.padX, lb.padY, srcW, srcH);
        }
    }

    // --- preprocessing ---
    private static class Letterbox { int[] rgb; float scale, padX, padY; }
    private Letterbox letterbox(int[] src, int w, int h) {
        float r = Math.min(inputW/(float)w, inputH/(float)h);
        int nw = (int)(w*r), nh = (int)(h*r);
        int dx = (inputW - nw)/2, dy = (inputH - nh)/2;

        int[] dst = new int[inputW*inputH]; // zero-padded
        for (int y=0; y<nh; y++) {
            int sy = Math.min((int)(y/r), h-1);
            for (int x=0; x<nw; x++) {
                int sx = Math.min((int)(x/r), w-1);
                dst[(y+dy)*inputW + (x+dx)] = src[sy*w + sx];
            }
        }
        Letterbox lb = new Letterbox();
        lb.rgb = dst; lb.scale = r; lb.padX = dx; lb.padY = dy;
        return lb;
    }

    private float[] toCHW(int[] rgb, int w, int h) {
        int size = w*h; float[] out = new float[3*size];
        int rI=0, gI=size, bI=2*size;
        for (int i=0;i<size;i++){
            int p = rgb[i];
            out[rI++] = ((p>>16)&0xFF)/255f;
            out[gI++] = ((p>>8)&0xFF)/255f;
            out[bI++] = (p&0xFF)/255f;
        }
        return out;
    }

    // --- parse YOLOv8 det output + NMS ---
    private List<Detection> parseDet(OnnxValue val, float scale, float padX, float padY, int imgW, int imgH) throws OrtException {
        OnnxTensor t = (OnnxTensor) val;
        long[] shape = t.getInfo().getShape(); // expect [1,84,N] or [1,N,84]
        float[] flat = toArray(t.getFloatBuffer());

        int dim1 = (int)shape[1], dim2 = (int)shape[2];
        boolean colsAreProps = dim1 < dim2; // [1,props,N]
        int props = colsAreProps ? dim1 : dim2;
        int clsCount = props - 4;
        int N = colsAreProps ? dim2 : dim1;

        List<Detection> dets = new ArrayList<>(N);
        if (colsAreProps) {
            int stride = N; // properties are stored in separate contiguous rows
            for (int i=0;i<N;i++){
                float x = flat[i];
                float y = flat[stride + i];
                float w = flat[2*stride + i];
                float h = flat[3*stride + i];

                int bestC = -1; float bestS = 0f;
                for (int c=0;c<clsCount;c++){
                    float s = flat[(4+c)*stride + i];
                    if (s>bestS){ bestS = s; bestC = c; }
                }
                if (bestS < confThresh) continue;

                float bx = x - w/2f, by = y - h/2f, ex = x + w/2f, ey = y + h/2f;
                float x1 = clamp((bx - padX)/scale, 0, imgW);
                float y1 = clamp((by - padY)/scale, 0, imgH);
                float x2 = clamp((ex - padX)/scale, 0, imgW);
                float y2 = clamp((ey - padY)/scale, 0, imgH);
                dets.add(new Detection(x1,y1,x2,y2,bestS,bestC,Float.NaN, sourceId, null, null));
            }
        } else {
            for (int i=0;i<N;i++){
                int base = i*props;
                float x = flat[base+0], y = flat[base+1],
                        w = flat[base+2], h = flat[base+3];

                int bestC = -1; float bestS = 0f;
                for (int c=0;c<clsCount;c++){
                    float s = flat[base+4+c];
                    if (s>bestS){ bestS = s; bestC = c; }
                }
                if (bestS < confThresh) continue;

                float bx = x - w/2f, by = y - h/2f, ex = x + w/2f, ey = y + h/2f;
                float x1 = clamp((bx - padX)/scale, 0, imgW);
                float y1 = clamp((by - padY)/scale, 0, imgH);
                float x2 = clamp((ex - padX)/scale, 0, imgW);
                float y2 = clamp((ey - padY)/scale, 0, imgH);
                dets.add(new Detection(x1,y1,x2,y2,bestS,bestC,Float.NaN, sourceId, null, null));
            }
        }
        return nms(dets, iouThresh);
    }

    private List<Detection> parseSeg(OnnxValue detVal, OnnxValue protoVal,
                                     Letterbox lb, int imgW, int imgH) throws OrtException {
        OnnxTensor detT = (OnnxTensor) detVal;
        float[] flat = toArray(detT.getFloatBuffer());
        long[] detShape = detT.getInfo().getShape(); // expect [1,props,N] or [1,N,props]

        ProtoInfo proto = ProtoInfo.from((OnnxTensor) protoVal);
        int nm = proto.nm;

        int dim1 = (int) detShape[1];
        int dim2 = (int) detShape[2];
        boolean colsAreProps = dim1 < dim2; // [1,props,N]
        int props = colsAreProps ? dim1 : dim2;
        int clsCount = props - 4 - nm;
        int N = colsAreProps ? dim2 : dim1;

        if (clsCount <= 0) {
            return Collections.emptyList();
        }

        List<Detection> dets = new ArrayList<>(N);
        if (colsAreProps) {
            int stride = N; // properties are stored in separate contiguous rows
            int coeffBase = 4 + clsCount;
            for (int i = 0; i < N; i++) {
                float x = flat[i];
                float y = flat[stride + i];
                float w = flat[2 * stride + i];
                float h = flat[3 * stride + i];

                int bestC = -1;
                float bestS = 0f;
                for (int c = 0; c < clsCount; c++) {
                    float s = flat[(4 + c) * stride + i];
                    if (s > bestS) {
                        bestS = s;
                        bestC = c;
                    }
                }
                if (bestS < confThresh) continue;

                float[] coeffs = new float[nm];
                for (int m = 0; m < nm; m++) {
                    coeffs[m] = flat[(coeffBase + m) * stride + i];
                }

                float bx = x - w / 2f, by = y - h / 2f, ex = x + w / 2f, ey = y + h / 2f;
                float x1 = clamp((bx - lb.padX) / lb.scale, 0, imgW);
                float y1 = clamp((by - lb.padY) / lb.scale, 0, imgH);
                float x2 = clamp((ex - lb.padX) / lb.scale, 0, imgW);
                float y2 = clamp((ey - lb.padY) / lb.scale, 0, imgH);
                dets.add(new Detection(x1, y1, x2, y2, bestS, bestC,
                        Float.NaN, sourceId, coeffs, null));
            }
        } else {
            for (int i = 0; i < N; i++) {
                int base = i * props;
                float x = flat[base];
                float y = flat[base + 1];
                float w = flat[base + 2];
                float h = flat[base + 3];

                int bestC = -1;
                float bestS = 0f;
                for (int c = 0; c < clsCount; c++) {
                    float s = flat[base + 4 + c];
                    if (s > bestS) {
                        bestS = s;
                        bestC = c;
                    }
                }
                if (bestS < confThresh) continue;

                float[] coeffs = new float[nm];
                int coeffBase = base + 4 + clsCount;
                for (int m = 0; m < nm; m++) {
                    coeffs[m] = flat[coeffBase + m];
                }

                float bx = x - w / 2f, by = y - h / 2f, ex = x + w / 2f, ey = y + h / 2f;
                float x1 = clamp((bx - lb.padX) / lb.scale, 0, imgW);
                float y1 = clamp((by - lb.padY) / lb.scale, 0, imgH);
                float x2 = clamp((ex - lb.padX) / lb.scale, 0, imgW);
                float y2 = clamp((ey - lb.padY) / lb.scale, 0, imgH);
                dets.add(new Detection(x1, y1, x2, y2, bestS, bestC,
                        Float.NaN, sourceId, coeffs, null));
            }
        }

        List<Detection> kept = nms(dets, iouThresh);
        if (kept.isEmpty()) return kept;
        List<Detection> out = new ArrayList<>(kept.size());
        for (Detection d : kept) {
            Detection.Mask mask = buildMask(d, proto, lb, imgW, imgH);
            out.add(mask != null ? d.withMask(mask) : d);
        }
        return out;
    }

    private static float clamp(float v, int lo, int hi){ return Math.max(lo, Math.min(hi, v)); }

    private static float iou(Detection A, Detection B){
        float ix1 = max(A.x1,B.x1), iy1 = max(A.y1,B.y1);
        float ix2 = min(A.x2,B.x2), iy2 = min(A.y2,B.y2);
        float iw = max(0f, ix2-ix1), ih = max(0f, iy2-iy1);
        float inter = iw*ih;
        float a = (A.x2-A.x1)*(A.y2-A.y1);
        float b = (B.x2-B.x1)*(B.y2-B.y1);
        return inter/(a+b-inter+1e-6f);
    }

    private static List<Detection> nms(List<Detection> in, float iouTh){
        ArrayList<Detection> dets = new ArrayList<>(in);
        dets.sort((d1,d2)-> Float.compare(d2.score, d1.score));
        List<Detection> keep = new ArrayList<>();
        while(!dets.isEmpty()){
            Detection a = dets.remove(0);
            keep.add(a);
            dets.removeIf(b -> b.cls==a.cls && iou(a,b) > iouTh);
        }
        return keep;
    }

    private static class ProtoInfo {
        final float[] data;
        final int nm;
        final int mh;
        final int mw;
        final boolean nchw;

        private ProtoInfo(float[] data, int nm, int mh, int mw, boolean nchw) {
            this.data = data;
            this.nm = nm;
            this.mh = mh;
            this.mw = mw;
            this.nchw = nchw;
        }

        float valueAt(int m, int y, int x) {
            if (nchw) {
                return data[(m * mh + y) * mw + x];
            }
            return data[(y * mw + x) * nm + m];
        }

        static ProtoInfo from(OnnxTensor t) throws OrtException {
            long[] shape = t.getInfo().getShape(); // [1,nm,mh,mw] or [1,mh,mw,nm]
            if (shape.length != 4) {
                throw new OrtException("Unexpected proto shape");
            }
            int d1 = (int) shape[1];
            int d2 = (int) shape[2];
            int d3 = (int) shape[3];
            float[] data = toArray(t.getFloatBuffer());

            if (d1 <= d2 && d1 <= d3) {
                return new ProtoInfo(data, d1, d2, d3, true);
            }
            if (d3 <= d1 && d3 <= d2) {
                return new ProtoInfo(data, d3, d1, d2, false);
            }
            return new ProtoInfo(data, d1, d2, d3, true);
        }
    }

    private Detection.Mask buildMask(Detection det, ProtoInfo proto,
                                     Letterbox lb, int imgW, int imgH) {
        if (det.maskCoeffs == null) return null;

        int boxX1 = clampInt((int) Math.floor(det.x1), 0, imgW - 1);
        int boxY1 = clampInt((int) Math.floor(det.y1), 0, imgH - 1);
        int boxX2 = clampInt((int) Math.ceil(det.x2), 0, imgW - 1);
        int boxY2 = clampInt((int) Math.ceil(det.y2), 0, imgH - 1);
        int boxW = Math.max(1, boxX2 - boxX1);
        int boxH = Math.max(1, boxY2 - boxY1);

        float x1Lb = det.x1 * lb.scale + lb.padX;
        float y1Lb = det.y1 * lb.scale + lb.padY;
        float x2Lb = det.x2 * lb.scale + lb.padX;
        float y2Lb = det.y2 * lb.scale + lb.padY;

        float sx = proto.mw / (float) inputW;
        float sy = proto.mh / (float) inputH;
        int mx1 = clampInt((int) Math.floor(x1Lb * sx), 0, proto.mw - 1);
        int my1 = clampInt((int) Math.floor(y1Lb * sy), 0, proto.mh - 1);
        int mx2 = clampInt((int) Math.ceil(x2Lb * sx), 0, proto.mw - 1);
        int my2 = clampInt((int) Math.ceil(y2Lb * sy), 0, proto.mh - 1);

        int roiW = Math.max(1, mx2 - mx1 + 1);
        int roiH = Math.max(1, my2 - my1 + 1);
        float[] roi = new float[roiW * roiH];

        for (int y = 0; y < roiH; y++) {
            int yy = my1 + y;
            int row = y * roiW;
            for (int x = 0; x < roiW; x++) {
                int xx = mx1 + x;
                float sum = 0f;
                for (int m = 0; m < proto.nm; m++) {
                    sum += det.maskCoeffs[m] * proto.valueAt(m, yy, xx);
                }
                roi[row + x] = sigmoid(sum);
            }
        }

        byte[] alpha = new byte[boxW * boxH];
        for (int y = 0; y < boxH; y++) {
            int syi = Math.min((int) ((y + 0.5f) * roiH / boxH), roiH - 1);
            int srcRow = syi * roiW;
            int dstRow = y * boxW;
            for (int x = 0; x < boxW; x++) {
                int sxi = Math.min((int) ((x + 0.5f) * roiW / boxW), roiW - 1);
                float v = roi[srcRow + sxi];
                if (v >= maskThreshold) {
                    int a = Math.min(255, Math.max(0, Math.round(v * 255f)));
                    alpha[dstRow + x] = (byte) a;
                }
            }
        }
        return new Detection.Mask(alpha, boxW, boxH, boxX1, boxY1);
    }

    private static float sigmoid(float x) {
        return (float) (1.0 / (1.0 + Math.exp(-x)));
    }

    private static int clampInt(int v, int lo, int hi) {
        return Math.max(lo, Math.min(hi, v));
    }

    private static float[] toArray(FloatBuffer buf) {
        float[] out = new float[buf.remaining()];
        buf.get(out);
        return out;
    }

    private static int sourceIdFor(String assetName, int fallback) {
        if (assetName == null) return fallback;
        String lower = assetName.toLowerCase(Locale.US);
        if (lower.contains("seg")) return Detection.SOURCE_SEG;
        return fallback;
    }

    @Override public void close() throws Exception {
        session.close();
    }

    // Utility to read asset fully
    static class Util {
        static byte[] readAllBytes(android.content.res.AssetManager am, String name){
            try(java.io.InputStream is = am.open(name);
                java.io.ByteArrayOutputStream bos = new java.io.ByteArrayOutputStream()){
                byte[] buf = new byte[1<<16]; int r;
                while ((r=is.read(buf))!=-1) bos.write(buf,0,r);
                return bos.toByteArray();
            } catch (Exception e){ throw new RuntimeException(e); }
        }

        static String cacheAsset(Context ctx, String assetName){
            File dir = new File(ctx.getFilesDir(), "models");
            if (!dir.exists()) dir.mkdirs();
            File out = new File(dir, assetName);
            if (out.exists() && out.length() > 0) return out.getAbsolutePath();
            try (InputStream is = ctx.getAssets().open(assetName);
                 FileOutputStream fos = new FileOutputStream(out)) {
                byte[] buf = new byte[1<<16]; int r;
                while ((r = is.read(buf)) != -1) fos.write(buf, 0, r);
                fos.flush();
                return out.getAbsolutePath();
            } catch (Exception e){
                throw new RuntimeException(e);
            }
        }

        static void deleteCachedAsset(Context ctx, String assetName) {
            try {
                File dir = new File(ctx.getFilesDir(), "models");
                File out = new File(dir, assetName);
                if (out.exists()) {
                    //noinspection ResultOfMethodCallIgnored
                    out.delete();
                }
            } catch (Exception ignored) {
            }
        }

        static String chooseFirstExistingAsset(Context ctx, String... names) {
            for (String name : names) {
                String preferred = preferInt8Asset(ctx, name);
                if (assetExists(ctx, preferred)) return preferred;
                if (assetExists(ctx, name)) return name;
            }
            throw new RuntimeException("No model asset found");
        }

        static List<String> buildModelCandidates(Context ctx, String... names) {
            LinkedHashSet<String> out = new LinkedHashSet<>();
            for (String name : names) {
                if (name == null || name.isEmpty()) continue;
                String preferred = preferInt8Asset(ctx, name);
                out.add(preferred);
                if (!preferred.equals(name)) {
                    out.add(name);
                }
            }
            return new ArrayList<>(out);
        }

        static boolean assetExists(Context ctx, String assetName) {
            try (InputStream ignored = ctx.getAssets().open(assetName)) {
                return true;
            } catch (Exception e) {
                return false;
            }
        }

        static String preferInt8Asset(Context ctx, String assetName) {
            if (assetName == null) return null;
            String lower = assetName.toLowerCase(Locale.US);
            if (lower.endsWith("_int8.onnx")) return assetName;
            if (lower.endsWith(".onnx")) {
                String int8Name = assetName.substring(0, assetName.length() - 5) + "_int8.onnx";
                if (assetExists(ctx, int8Name)) return int8Name;
            }
            return assetName;
        }
    }
}
