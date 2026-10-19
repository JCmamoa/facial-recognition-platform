package com.facial.recognition.core.detection;

import com.facial.recognition.core.exception.DetectionException;
import com.facial.recognition.core.exception.ModelInitializationException;
import com.facial.recognition.core.model.DetectedFace;
import com.facial.recognition.core.model.FaceLandmarks;
import com.facial.recognition.core.model.Point2D;
import lombok.extern.slf4j.Slf4j;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Rect;
import org.bytedeco.opencv.opencv_core.RectVector;
import org.bytedeco.opencv.opencv_core.Size;
import org.bytedeco.opencv.opencv_objdetect.CascadeClassifier;
import org.bytedeco.opencv.global.opencv_imgproc;

import java.io.File;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.List;

/**
 * Classical Viola-Jones Haar Cascade face detector using OpenCV's CascadeClassifier.
 * Useful as a lightweight fallback for edge devices or when neural networks are unavailable.
 */
@Slf4j
public class HaarCascadeFaceDetector implements FaceDetector {

    private final String cascadePath;
    private CascadeClassifier classifier;

    public HaarCascadeFaceDetector() {
        this(resolveDefaultCascadePath());
    }

    public HaarCascadeFaceDetector(String cascadePath) {
        this.cascadePath = cascadePath;
        initClassifier();
    }

    private static String resolveDefaultCascadePath() {
        String customPath = System.getProperty("facial.haarcascade.path", "/home/carlos-44/models/haarcascade_frontalface_alt2.xml");
        File file = new File(customPath);
        if (file.exists()) {
            return file.getAbsolutePath();
        }
        try (InputStream in = HaarCascadeFaceDetector.class.getResourceAsStream("/models/haarcascade_frontalface_alt2.xml")) {
            if (in != null) {
                Path tempFile = Files.createTempFile("haarcascade_", ".xml");
                tempFile.toFile().deleteOnExit();
                Files.copy(in, tempFile, StandardCopyOption.REPLACE_EXISTING);
                return tempFile.toAbsolutePath().toString();
            }
        } catch (Exception e) {
            log.warn("Failed to extract default Haar cascade from classpath: {}", e.getMessage());
        }
        return customPath;
    }

    private synchronized void initClassifier() {
        try {
            log.info("Loading Haar CascadeClassifier from: {}", cascadePath);
            File file = new File(cascadePath);
            if (!file.exists()) {
                throw new ModelInitializationException("Cascade file not found at " + cascadePath);
            }
            classifier = new CascadeClassifier(cascadePath);
            if (classifier.empty()) {
                throw new ModelInitializationException("Failed to load cascade classifier from " + cascadePath);
            }
            log.info("Haar CascadeClassifier loaded successfully.");
        } catch (Exception e) {
            throw new ModelInitializationException("Error loading CascadeClassifier: " + e.getMessage(), e);
        }
    }

    @Override
    public List<DetectedFace> detectFaces(Mat image) throws DetectionException {
        return detectFaces(image, 0.5f);
    }

    @Override
    public synchronized List<DetectedFace> detectFaces(Mat image, float scoreThreshold) throws DetectionException {
        if (image == null || image.empty()) {
            return List.of();
        }

        try {
            Mat gray = new Mat();
            if (image.channels() > 1) {
                opencv_imgproc.cvtColor(image, gray, opencv_imgproc.COLOR_BGR2GRAY);
            } else {
                gray = image;
            }

            opencv_imgproc.equalizeHist(gray, gray);

            RectVector faces = new RectVector();
            classifier.detectMultiScale(gray, faces, 1.1, 4, 0, new Size(30, 30), new Size());

            List<DetectedFace> detectedList = new ArrayList<>();
            long size = faces.size();
            for (long i = 0; i < size; i++) {
                Rect r = faces.get(i);
                int x = r.x();
                int y = r.y();
                int w = r.width();
                int h = r.height();

                // Approximate 5 landmarks geometrically for classical cascades
                Point2D rightEye = new Point2D(x + w * 0.32f, y + h * 0.38f);
                Point2D leftEye = new Point2D(x + w * 0.68f, y + h * 0.38f);
                Point2D noseTip = new Point2D(x + w * 0.50f, y + h * 0.55f);
                Point2D rightMouth = new Point2D(x + w * 0.35f, y + h * 0.75f);
                Point2D leftMouth = new Point2D(x + w * 0.65f, y + h * 0.75f);

                FaceLandmarks landmarks = FaceLandmarks.builder()
                        .rightEye(rightEye)
                        .leftEye(leftEye)
                        .noseTip(noseTip)
                        .rightMouth(rightMouth)
                        .leftMouth(leftMouth)
                        .build();

                DetectedFace detected = DetectedFace.builder()
                        .x(x)
                        .y(y)
                        .width(w)
                        .height(h)
                        .confidence(0.85f)
                        .landmarks(landmarks)
                        .livenessPassed(true)
                        .build();

                detectedList.add(detected);
            }

            return detectedList;
        } catch (Exception e) {
            log.error("Haar cascade face detection error", e);
            throw new DetectionException("Haar cascade face detection failed: " + e.getMessage(), e);
        }
    }

    @Override
    public String getName() {
        return "HaarCascade";
    }

    @Override
    public synchronized void close() {
        if (classifier != null) {
            classifier.close();
            classifier = null;
        }
    }
}
