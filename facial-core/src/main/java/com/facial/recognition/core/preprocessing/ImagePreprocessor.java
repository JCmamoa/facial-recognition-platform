package com.facial.recognition.core.preprocessing;

import com.facial.recognition.core.model.DetectedFace;
import com.facial.recognition.core.model.FaceLandmarks;
import com.facial.recognition.core.model.Point2D;
import lombok.extern.slf4j.Slf4j;
import org.bytedeco.javacpp.indexer.FloatRawIndexer;
import org.bytedeco.opencv.opencv_imgproc.CLAHE;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.MatVector;
import org.bytedeco.opencv.opencv_core.Point2f;
import org.bytedeco.opencv.opencv_core.Rect;
import org.bytedeco.opencv.opencv_core.Scalar;
import org.bytedeco.opencv.opencv_core.Size;
import org.bytedeco.opencv.global.opencv_core;
import org.bytedeco.opencv.global.opencv_imgproc;

/**
 * Advanced image preprocessing pipeline:
 * - Color space conversions (BGR to Grayscale, RGB, YCrCb, HSV)
 * - Contrast Limited Adaptive Histogram Equalization (CLAHE)
 * - Brightness, contrast and gamma normalization
 * - Facial landmark alignment via affine rotation matrix
 */
@Slf4j
public class ImagePreprocessor {

    /**
     * Converts BGR image to single-channel Grayscale.
     */
    public Mat bgrToGray(Mat bgr) {
        if (bgr == null || bgr.empty()) return new Mat();
        Mat gray = new Mat();
        opencv_imgproc.cvtColor(bgr, gray, opencv_imgproc.COLOR_BGR2GRAY);
        return gray;
    }

    /**
     * Converts BGR image to RGB format.
     */
    public Mat bgrToRgb(Mat bgr) {
        if (bgr == null || bgr.empty()) return new Mat();
        Mat rgb = new Mat();
        opencv_imgproc.cvtColor(bgr, rgb, opencv_imgproc.COLOR_BGR2RGB);
        return rgb;
    }

    /**
     * Normalizes brightness and contrast using linear transform: g(x) = alpha * f(x) + beta.
     *
     * @param src Source image
     * @param alpha Contrast gain multiplier (1.0 = unchanged, >1.0 = higher contrast)
     * @param beta Brightness bias shift (-255 to +255, 0 = unchanged)
     */
    public Mat normalizeBrightnessAndContrast(Mat src, double alpha, double beta) {
        if (src == null || src.empty()) return new Mat();
        Mat normalized = new Mat();
        src.convertTo(normalized, -1, alpha, beta);
        return normalized;
    }

    /**
     * Applies Contrast Limited Adaptive Histogram Equalization (CLAHE).
     * For color images, converts to YCrCb or Lab and equalizes the luminance (Y/L) channel.
     */
    public Mat applyClahe(Mat src, double clipLimit, int tileGridSize) {
        if (src == null || src.empty()) return new Mat();

        CLAHE clahe = opencv_imgproc.createCLAHE(clipLimit, new Size(tileGridSize, tileGridSize));

        if (src.channels() == 1) {
            Mat dst = new Mat();
            clahe.apply(src, dst);
            return dst;
        }

        // Color image: process luminance channel in YCrCb
        Mat yCrCb = new Mat();
        opencv_imgproc.cvtColor(src, yCrCb, opencv_imgproc.COLOR_BGR2YCrCb);

        MatVector channels = new MatVector();
        opencv_core.split(yCrCb, channels);

        Mat yChannel = channels.get(0);
        Mat equalizedY = new Mat();
        clahe.apply(yChannel, equalizedY);
        channels.put(0, equalizedY);

        Mat mergedYCrCb = new Mat();
        opencv_core.merge(channels, mergedYCrCb);

        Mat resultBgr = new Mat();
        opencv_imgproc.cvtColor(mergedYCrCb, resultBgr, opencv_imgproc.COLOR_YCrCb2BGR);

        return resultBgr;
    }

    /**
     * Applies non-linear Gamma correction to balance extreme lighting (backlight, shadows).
     */
    public Mat applyGammaCorrection(Mat src, double gamma) {
        if (src == null || src.empty() || gamma <= 0.0) return src;

        // Build lookup table
        byte[] lut = new byte[256];
        double invGamma = 1.0 / gamma;
        for (int i = 0; i < 256; i++) {
            lut[i] = (byte) Math.min(255, Math.max(0, (int) (Math.pow(i / 255.0, invGamma) * 255.0)));
        }

        Mat lutMat = new Mat(1, 256, opencv_core.CV_8U);
        lutMat.data().put(lut);

        Mat dst = new Mat();
        opencv_core.LUT(src, lutMat, dst);
        return dst;
    }

    /**
     * Geometrically aligns a face using the 5 detected landmarks (especially eyes).
     * Computes rotation angle to make the inter-ocular line horizontal, scales and warps
     * the image to a standardized canonical size (e.g., 112x112 standard for deep models).
     */
    public Mat alignFace(Mat src, FaceLandmarks landmarks, int targetWidth, int targetHeight) {
        if (src == null || src.empty() || landmarks == null ||
            landmarks.getRightEye() == null || landmarks.getLeftEye() == null) {
            return src;
        }

        Point2D rightEye = landmarks.getRightEye();
        Point2D leftEye = landmarks.getLeftEye();

        // Calculate angle between eyes
        double dX = leftEye.getX() - rightEye.getX();
        double dY = leftEye.getY() - rightEye.getY();
        double angle = Math.toDegrees(Math.atan2(dY, dX));

        // Center between eyes
        Point2D eyesCenter = landmarks.getEyeCenter();
        Point2f center = new Point2f(eyesCenter.getX(), eyesCenter.getY());

        // Eye distance
        double eyeDistance = Math.sqrt(dX * dX + dY * dY);
        // Desired distance in canonical 112x112 target is ~35-40% of target width
        double desiredEyeDistance = targetWidth * 0.38;
        double scale = desiredEyeDistance / Math.max(eyeDistance, 1.0);

        // Rotation matrix
        Mat rotMat = opencv_imgproc.getRotationMatrix2D(center, angle, scale);

        // Adjust translation to place eye center at desired canonical position
        double desiredEyeCenterX = targetWidth * 0.5;
        double desiredEyeCenterY = targetHeight * 0.38;

        org.bytedeco.javacpp.indexer.DoubleRawIndexer indexer = rotMat.createIndexer();
        double tx = (desiredEyeCenterX - center.x());
        double ty = (desiredEyeCenterY - center.y());
        indexer.put(0, 2, indexer.get(0, 2) + tx);
        indexer.put(1, 2, indexer.get(1, 2) + ty);
        indexer.release();

        Mat alignedFace = new Mat();
        opencv_imgproc.warpAffine(src, alignedFace, rotMat, new Size(targetWidth, targetHeight),
                opencv_imgproc.INTER_LINEAR, opencv_core.BORDER_REFLECT_101, new Scalar(0, 0, 0, 0));

        return alignedFace;
    }

    /**
     * Crops face from source image with margin factor (e.g. 0.15 = 15% extra padding).
     */
    public Mat cropFace(Mat src, DetectedFace face, double marginFactor) {
        if (src == null || src.empty() || face == null) {
            return null;
        }

        int marginX = (int) (face.getWidth() * marginFactor);
        int marginY = (int) (face.getHeight() * marginFactor);

        int x1 = Math.max(0, face.getX() - marginX);
        int y1 = Math.max(0, face.getY() - marginY);
        int x2 = Math.min(src.cols(), face.getX() + face.getWidth() + marginX);
        int y2 = Math.min(src.rows(), face.getY() + face.getHeight() + marginY);

        int w = x2 - x1;
        int h = y2 - y1;

        if (w <= 0 || h <= 0) {
            return null;
        }

        Rect cropRoi = new Rect(x1, y1, w, h);
        return new Mat(src, cropRoi).clone();
    }
}
