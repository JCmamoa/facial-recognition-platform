package com.facial.recognition.core.preprocessing;

import com.facial.recognition.core.model.FaceLandmarks;
import com.facial.recognition.core.model.Point2D;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Scalar;
import org.bytedeco.opencv.global.opencv_core;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;

class ImagePreprocessorTest {

    private final ImagePreprocessor preprocessor = new ImagePreprocessor();

    @Test
    @DisplayName("Convert BGR to Gray produces single-channel Mat")
    void testBgrToGray() {
        Mat bgr = new Mat(100, 100, opencv_core.CV_8UC3, new Scalar(100, 150, 200, 0));
        Mat gray = preprocessor.bgrToGray(bgr);

        assertThat(gray.channels()).isEqualTo(1);
        assertThat(gray.cols()).isEqualTo(100);
        assertThat(gray.rows()).isEqualTo(100);
    }

    @Test
    @DisplayName("Convert BGR to RGB preserves 3 channels")
    void testBgrToRgb() {
        Mat bgr = new Mat(50, 50, opencv_core.CV_8UC3, new Scalar(10, 20, 30, 0));
        Mat rgb = preprocessor.bgrToRgb(bgr);

        assertThat(rgb.channels()).isEqualTo(3);
        assertThat(rgb.cols()).isEqualTo(50);
    }

    @Test
    @DisplayName("CLAHE processing produces valid normalized image")
    void testApplyClahe() {
        Mat bgr = new Mat(100, 100, opencv_core.CV_8UC3, new Scalar(80, 90, 110, 0));
        Mat clahe = preprocessor.applyClahe(bgr, 2.0, 8);

        assertThat(clahe.empty()).isFalse();
        assertThat(clahe.channels()).isEqualTo(3);
    }

    @Test
    @DisplayName("Face alignment aligns and scales face into target dimensions")
    void testAlignFace() {
        Mat bgr = new Mat(200, 200, opencv_core.CV_8UC3, new Scalar(120, 120, 120, 0));
        FaceLandmarks landmarks = FaceLandmarks.builder()
                .rightEye(new Point2D(70, 80))
                .leftEye(new Point2D(130, 80))
                .noseTip(new Point2D(100, 110))
                .rightMouth(new Point2D(80, 140))
                .leftMouth(new Point2D(120, 140))
                .build();

        Mat aligned = preprocessor.alignFace(bgr, landmarks, 112, 112);

        assertThat(aligned.empty()).isFalse();
        assertThat(aligned.cols()).isEqualTo(112);
        assertThat(aligned.rows()).isEqualTo(112);
    }
}
