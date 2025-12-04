package vn.edu.usth.objectdetectmobile;

import android.content.Context;
import android.util.Log;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;

public class LabelHelper {

    private static final String TAG = "LabelHelper";

    private LabelHelper() {
        // no instances
    }

    public static String[] loadLabels(Context context, String assetFileName) {
        List<String> list = new ArrayList<>();
        BufferedReader br = null;
        try {
            br = new BufferedReader(
                    new InputStreamReader(context.getAssets().open(assetFileName))
            );
            String line;
            while ((line = br.readLine()) != null) {
                list.add(line);
            }
        } catch (Exception e) {
            Log.w(TAG, "Failed to load labels from assets: " + assetFileName, e);
        } finally {
            try {
                if (br != null) br.close();
            } catch (Exception ignored) {
            }
        }
        return list.toArray(new String[0]);
    }
}
