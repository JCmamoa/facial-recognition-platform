package com.facial.recognition.core.detection;

import com.facial.recognition.core.exception.DetectionException;
import com.facial.recognition.core.exception.ModelInitializationException;
import com.facial.recognition.core.model.DetectedFace;
import com.facial.recognition.core.model.FaceLandmarks;
import com.facial.recognition.core.model.Point2D;
import lombok.extern.slf4j.Slf4j;
import org.bytedeco.javacpp.indexer.FloatRawIndexer;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Size;
import org.bytedeco.opencv.opencv_objdetect.FaceDetectorYN;

import java.io.File;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.List;

/**
 * High-accuracy Deep Neural Network face detector based on OpenCV's YuNet ONNX model.
 * Produces accurate bounding boxes and 5 facial landmarks across extreme angles, scales,
 * and varying lighting conditions.
 */
@Slf4j
public class YuNetFaceDetector implements FaceDetector {

    private final String modelPath;
    private final float defaultScoreThreshold;
    private final float nmsThreshold;
    private final int topK;

    private FaceDetectorYN detector;
    private int lastWidth = -1;
    private int lastHeight = -1;

    public YuNetFaceDetector() {
        this(resolveDefaultModelPath(), 0.6f, 0.3f, 5000);
    }

    public YuNetFaceDetector(String modelPath) {
        this(modelPath, 0.6f, 0.3f, 5000);
    }

    public YuNetFaceDetector(String modelPath, float scoreThreshold, float nmsThreshold, int topK) {
        this.modelPath = modelPath;
        this.defaultScoreThreshold = scoreThreshold;
        this.nmsThreshold = nmsThreshold;
        this.topK = topK;
        initDetector();
    }

    private static String resolveDefaultModelPath() {
        String customPath = System.getProperty("facial.yunet.model.path", "/home/carlos-44/models/face_detection_yunet.onnx");
        File file = new File(customPath);
        if (file.exists()) {
            return file.getAbsolutePath();
        }
        // Try extracting from classpath resources
        try (InputStream in = YuNetFaceDetector.class.getResourceAsStream("/models/face_detection_yunet.onnx")) {
            if (in != null) {
                Path tempFile = Files.createTempFile("yunet_", ".onnx");
                tempFile.toFile().deleteOnExit();
                Files.copy(in, tempFile, StandardCopyOption.REPLACE_EXISTING);
                return tempFile.toAbsolutePath().toString();
            }
        } catch (Exception e) {
            log.warn("Failed to extract default YuNet model from classpath: {}", e.getMessage());
        }
        return customPath;
    }

    private synchronized void initDetector() {
        try {
            log.info("Initializing YuNet FaceDetectorYN from model: {}", modelPath);
            File file = new File(modelPath);
            if (!file.exists()) {
                throw new ModelInitializationException("YuNet model file not found at " + modelPath);
            }
            // Initialize with placeholder size (320x320), will be dynamically resized per input image
            detector = FaceDetectorYN.create(
                    modelPath,
                    "",
                    new Size(320, 320),
                    defaultScoreThreshold,
                    nmsThreshold,
                    topK,
                    0, // backend: DNN_BACKEND_DEFAULT
                    0  // target: DNN_TARGET_CPU
            );
            log.info("YuNet FaceDetectorYN successfully loaded and initialized.");
        } catch (Exception e) {
            throw new ModelInitializationException("Could not initialize YuNet FaceDetectorYN from " + modelPath, e);
        }
    }

    @Override
    public List<DetectedFace> detectFaces(Mat image) throws DetectionException {
        return detectFaces(image, defaultScoreThreshold);
    }

    @Override
    public synchronized List<DetectedFace> detectFaces(Mat image, float scoreThreshold) throws DetectionException {
        if (image == null || image.empty()) {
            return List.of();
        }

        try {
            int width = image.cols();
            int height = image.rows();

            if (width != lastWidth || height != lastHeight) {
                detector.setInputSize(new Size(width, height));
                lastWidth = width;
                lastHeight = height;
            }
            detector.setScoreThreshold(scoreThreshold);

            Mat faces = new Mat();
            detector.detect(image, faces);

            List<DetectedFace> detectedList = new ArrayList<>();
            if (faces.empty() || faces.rows() == 0) {
                return detectedList;
            }

            FloatRawIndexer indexer = faces.createIndexer();
            try {
                int numFaces = faces.rows();
                for (int i = 0; i < numFaces; i++) {
                    float x = indexer.get(i, 0);
                    float y = indexer.get(i, 1);
                    float w = indexer.get(i, 2);
                    float h = indexer.get(i, 3);

                    // Landmarks
                    Point2D rightEye = new Point2D(indexer.get(i, 4), indexer.get(i, 5));
                    Point2D leftEye = new Point2D(indexer.get(i, 6), indexer.get(i, 7));
                    Point2D noseTip = new Point2D(indexer.get(i, 8), indexer.get(i, 9));
                    Point2D rightMouth = new Point2D(indexer.get(i, 10), indexer.get(i, 11));
                    Point2D leftMouth = new Point2D(indexer.get(i, 12), indexer.get(i, 13));
                    float conf = indexer.get(i, 14);

                    FaceLandmarks landmarks = FaceLandmarks.builder()
                            .rightEye(rightEye)
                            .leftEye(leftEye)
                            .noseTip(noseTip)
                            .rightMouth(rightMouth)
                            .leftMouth(leftMouth)
                            .build();

                    DetectedFace face = DetectedFace.builder()
                            .x(Math.max(0, (int) x))
                            .y(Math.max(0, (int) y))
                            .width(Math.max(1, (int) w))
                            .height(Math.max(1, (int) h))
                            .confidence(conf)
                            .landmarks(landmarks)
                            .livenessPassed(true)
                            .build();

                    detectedList.add(face);
                }
            } finally {
                indexer.release();
            }

            return detectedList;
        } catch (Exception e) {
            log.error("Error detecting faces via YuNet", e);
            throw new DetectionException("YuNet face detection failed: " + e.getMessage(), e);
        }
    }

    @Override
    public String getName() {
        return "YuNet-DNN";
    }

    @Override
    public synchronized void close() {
        if (detector != null) {
            detector.close();
            detector = null;
        }
    }
}
