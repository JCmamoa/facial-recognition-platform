package com.facial.recognition.core.capture;

import org.bytedeco.javacv.Java2DFrameConverter;
import org.bytedeco.javacv.OpenCVFrameConverter;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Rect;
import org.bytedeco.opencv.opencv_core.Scalar;
import org.bytedeco.opencv.opencv_core.Point;
import org.bytedeco.opencv.global.opencv_imgproc;
import org.bytedeco.opencv.global.opencv_imgcodecs;

import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.InputStream;
import javax.imageio.ImageIO;

/**
 * High-performance conversion and visualization utilities between OpenCV Mat,
 * JavaCV Frame, BufferedImage, and byte arrays.
 */
public final class OpenCVUtils {

    private OpenCVUtils() {}

    /**
     * Converts OpenCV Mat to BufferedImage.
     */
    public static BufferedImage matToBufferedImage(Mat mat) {
        if (mat == null || mat.empty()) {
            return null;
        }
        try (OpenCVFrameConverter.ToMat toMat = new OpenCVFrameConverter.ToMat();
             Java2DFrameConverter toBIm = new Java2DFrameConverter()) {
            return toBIm.convert(toMat.convert(mat));
        } catch (Exception e) {
            // Fallback via JPEG encoding in memory
            try {
                byte[] bytes = matToByteArray(mat, ".jpg");
                try (InputStream in = new ByteArrayInputStream(bytes)) {
                    return ImageIO.read(in);
                }
            } catch (Exception ex) {
                throw new RuntimeException("Failed to convert Mat to BufferedImage", ex);
            }
        }
    }

    /**
     * Converts BufferedImage to OpenCV Mat (BGR format).
     */
    public static Mat bufferedImageToMat(BufferedImage image) {
        if (image == null) {
            return null;
        }
        try (Java2DFrameConverter fromBIm = new Java2DFrameConverter();
             OpenCVFrameConverter.ToMat toMat = new OpenCVFrameConverter.ToMat()) {
            return toMat.convert(fromBIm.convert(image)).clone();
        } catch (Exception e) {
            try {
                ByteArrayOutputStream baos = new ByteArrayOutputStream();
                ImageIO.write(image, "jpg", baos);
                byte[] bytes = baos.toByteArray();
                return byteArrayToMat(bytes);
            } catch (Exception ex) {
                throw new RuntimeException("Failed to convert BufferedImage to Mat", ex);
            }
        }
    }

    /**
     * Encodes Mat to byte array with specified format (.jpg, .png).
     */
    public static byte[] matToByteArray(Mat mat, String format) {
        if (mat == null || mat.empty()) {
            return new byte[0];
        }
        org.bytedeco.javacpp.BytePointer outputBuffer = new org.bytedeco.javacpp.BytePointer();
        try {
            boolean success = opencv_imgcodecs.imencode(format, mat, outputBuffer);
            if (!success) {
                return new byte[0];
            }
            byte[] byteArray = new byte[(int) outputBuffer.limit()];
            outputBuffer.get(byteArray);
            return byteArray;
        } finally {
            outputBuffer.deallocate();
        }
    }

    /**
     * Decodes byte array (JPEG/PNG) to OpenCV Mat.
     */
    public static Mat byteArrayToMat(byte[] bytes) {
        if (bytes == null || bytes.length == 0) {
            return new Mat();
        }
        return opencv_imgcodecs.imdecode(new Mat(bytes), opencv_imgcodecs.IMREAD_COLOR);
    }
}
