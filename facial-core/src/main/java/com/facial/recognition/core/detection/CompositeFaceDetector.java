package com.facial.recognition.core.detection;

import com.facial.recognition.core.exception.DetectionException;
import com.facial.recognition.core.model.DetectedFace;
import lombok.extern.slf4j.Slf4j;
import org.bytedeco.opencv.opencv_core.Mat;

import java.util.List;

/**
 * Resilient composite face detector that prioritizes high-accuracy DNN detection (YuNet)
 * and automatically falls back to classical Haar cascades in case of hardware or model failures.
 */
@Slf4j
public class CompositeFaceDetector implements FaceDetector {

    private final FaceDetector primaryDetector;
    private final FaceDetector fallbackDetector;

    public CompositeFaceDetector() {
        FaceDetector primary;
        FaceDetector fallback;
        try {
            primary = new YuNetFaceDetector();
        } catch (Exception e) {
            log.warn("YuNet DNN detector could not be loaded as primary, falling back to Haar: {}", e.getMessage());
            primary = null;
        }

        try {
            fallback = new HaarCascadeFaceDetector();
        } catch (Exception e) {
            log.warn("HaarCascade detector could not be loaded: {}", e.getMessage());
            fallback = null;
        }

        this.primaryDetector = primary;
        this.fallbackDetector = fallback;
    }

    public CompositeFaceDetector(FaceDetector primary, FaceDetector fallback) {
        this.primaryDetector = primary;
        this.fallbackDetector = fallback;
    }

    @Override
    public List<DetectedFace> detectFaces(Mat image) throws DetectionException {
        return detectFaces(image, 0.6f);
    }

    @Override
    public List<DetectedFace> detectFaces(Mat image, float scoreThreshold) throws DetectionException {
        if (primaryDetector != null) {
            try {
                return primaryDetector.detectFaces(image, scoreThreshold);
            } catch (Exception e) {
                log.warn("Primary detector '{}' failed. Invoking fallback detector...", primaryDetector.getName(), e);
            }
        }
        if (fallbackDetector != null) {
            return fallbackDetector.detectFaces(image, scoreThreshold);
        }
        throw new DetectionException("No functional face detector is currently available.");
    }

    @Override
    public String getName() {
        return "Composite(" + (primaryDetector != null ? primaryDetector.getName() : "None") + " -> " +
                (fallbackDetector != null ? fallbackDetector.getName() : "None") + ")";
    }

    @Override
    public void close() {
        if (primaryDetector != null) primaryDetector.close();
        if (fallbackDetector != null) fallbackDetector.close();
    }
}
