package com.facial.recognition.core.detection;

import com.facial.recognition.core.exception.DetectionException;
import com.facial.recognition.core.model.DetectedFace;
import org.bytedeco.opencv.opencv_core.Mat;

import java.util.List;

/**
 * Interface for deep learning and cascade-based face detection algorithms.
 */
public interface FaceDetector extends AutoCloseable {

    /**
     * Detects all faces present in the input image using the default confidence threshold.
     */
    List<DetectedFace> detectFaces(Mat image) throws DetectionException;

    /**
     * Detects all faces present in the input image with a custom confidence threshold.
     */
    List<DetectedFace> detectFaces(Mat image, float scoreThreshold) throws DetectionException;

    /**
     * Returns the name/architecture of this detector (e.g., "YuNet-DNN", "HaarCascade").
     */
    String getName();

    @Override
    default void close() {
        // Default no-op
    }
}
