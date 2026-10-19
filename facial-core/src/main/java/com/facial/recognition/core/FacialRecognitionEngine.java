package com.facial.recognition.core;

import com.facial.recognition.core.biometrics.BiometricThresholds;
import com.facial.recognition.core.biometrics.InMemoryVectorDatabase;
import com.facial.recognition.core.biometrics.VectorDatabase;
import com.facial.recognition.core.capture.OpenCVUtils;
import com.facial.recognition.core.detection.CompositeFaceDetector;
import com.facial.recognition.core.detection.FaceDetector;
import com.facial.recognition.core.embedding.FaceEmbeddingExtractor;
import com.facial.recognition.core.embedding.SFaceEmbeddingExtractor;
import com.facial.recognition.core.exception.FacialRecognitionException;
import com.facial.recognition.core.liveness.LivenessDetector;
import com.facial.recognition.core.model.BiometricProfile;
import com.facial.recognition.core.model.DetectedFace;
import com.facial.recognition.core.model.FaceLandmarks;
import com.facial.recognition.core.model.Point2D;
import com.facial.recognition.core.model.RecognitionResult;
import com.facial.recognition.core.preprocessing.ImagePreprocessor;
import lombok.Getter;
import lombok.Setter;
import lombok.extern.slf4j.Slf4j;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Point;
import org.bytedeco.opencv.opencv_core.Rect;
import org.bytedeco.opencv.opencv_core.Scalar;
import org.bytedeco.opencv.global.opencv_imgproc;

import java.util.ArrayList;
import java.util.List;

/**
 * Main orchestration engine for the facial recognition and biometric detection platform.
 * Unifies frame acquisition, pre-processing, DNN face detection, landmark alignment,
 * 128-dimensional embedding extraction, vector search, and on-frame visual annotation.
 */
@Slf4j
public class FacialRecognitionEngine implements AutoCloseable {

    @Getter
    private final FaceDetector faceDetector;

    @Getter
    private final FaceEmbeddingExtractor embeddingExtractor;

    @Getter
    private final ImagePreprocessor imagePreprocessor;

    @Getter
    private final LivenessDetector livenessDetector;

    @Getter
    private final VectorDatabase vectorDatabase;

    @Getter @Setter
    private double similarityThreshold = BiometricThresholds.DEFAULT_COSINE_SIMILARITY_THRESHOLD;

    @Getter @Setter
    private boolean livenessCheckEnabled = true;

    public FacialRecognitionEngine() {
        this(new CompositeFaceDetector(),
             new SFaceEmbeddingExtractor(),
             new ImagePreprocessor(),
             new LivenessDetector(),
             new InMemoryVectorDatabase());
    }

    public FacialRecognitionEngine(FaceDetector detector,
                                   FaceEmbeddingExtractor extractor,
                                   ImagePreprocessor preprocessor,
                                   LivenessDetector livenessDetector,
                                   VectorDatabase vectorDatabase) {
        this.faceDetector = detector;
        this.embeddingExtractor = extractor;
        this.imagePreprocessor = preprocessor;
        this.livenessDetector = livenessDetector;
        this.vectorDatabase = vectorDatabase;
        log.info("FacialRecognitionEngine initialized successfully with detector: {}", detector.getName());
    }

    /**
     * Detects faces in the given image.
     */
    public List<DetectedFace> detectFaces(Mat image) {
        return faceDetector.detectFaces(image);
    }

    /**
     * Detects faces and extracts their 128-d embeddings and liveness scores.
     */
    public List<DetectedFace> detectAndExtractEmbeddings(Mat image) {
        List<DetectedFace> faces = faceDetector.detectFaces(image);
        for (DetectedFace face : faces) {
            if (livenessCheckEnabled) {
                livenessDetector.verifyLiveness(image, face);
            }
            float[] emb = embeddingExtractor.extractEmbedding(image, face);
            face.setEmbedding(emb);
        }
        return faces;
    }

    /**
     * Enrolls a new biometric subject profile using an image containing the subject's face.
     */
    public BiometricProfile enrollSubject(String id, String name, String documentId,
                                          String role, String department, Mat faceImage) {
        log.info("Enrolling subject '{}' (id: {})", name, id);
        List<DetectedFace> faces = detectFaces(faceImage);
        if (faces.isEmpty()) {
            throw new FacialRecognitionException("No face detected in enrollment image for subject " + name);
        }
        // Choose largest face in enrollment image
        DetectedFace primaryFace = faces.stream()
                .max((a, b) -> Double.compare(a.getArea(), b.getArea()))
                .orElse(faces.get(0));

        float[] embedding = embeddingExtractor.extractEmbedding(faceImage, primaryFace);

        BiometricProfile profile = vectorDatabase.findById(id).orElseGet(() ->
                BiometricProfile.builder()
                        .id(id)
                        .name(name)
                        .documentId(documentId)
                        .role(role)
                        .department(department)
                        .embeddings(new ArrayList<>())
                        .build()
        );

        profile.addEmbedding(embedding);
        vectorDatabase.enroll(profile);
        return profile;
    }

    /**
     * Processes a video frame in real time: detects faces, verifies liveness,
     * extracts embeddings, identifies against vector database, and annotates the frame if requested.
     */
    public List<RecognitionResult> processFrame(Mat frame, boolean drawOverlay) {
        if (frame == null || frame.empty()) {
            return List.of();
        }

        long startTime = System.currentTimeMillis();
        List<DetectedFace> faces = detectFaces(frame);
        List<RecognitionResult> results = new ArrayList<>();

        for (DetectedFace face : faces) {
            if (livenessCheckEnabled) {
                livenessDetector.verifyLiveness(frame, face);
            }

            float[] embedding = embeddingExtractor.extractEmbedding(frame, face);
            face.setEmbedding(embedding);

            RecognitionResult result;
            if (livenessCheckEnabled && !face.isLivenessPassed()) {
                result = RecognitionResult.builder()
                        .subjectId("SPOOF_ALERT")
                        .subjectName("Alerta: Spoofing")
                        .matched(false)
                        .similarity(0.0)
                        .distance(Double.MAX_VALUE)
                        .detectionConfidence(face.getConfidence())
                        .detectedFace(face)
                        .inferenceTimeMs(System.currentTimeMillis() - startTime)
                        .build();
            } else {
                result = vectorDatabase.identify(embedding, similarityThreshold);
                result.setDetectedFace(face);
                result.setDetectionConfidence(face.getConfidence());
            }

            results.add(result);

            if (drawOverlay) {
                drawFaceOverlay(frame, face, result);
            }
        }

        return results;
    }

    /**
     * Renders bounding boxes, landmark points, and biometric tags onto the frame.
     */
    public void drawFaceOverlay(Mat frame, DetectedFace face, RecognitionResult result) {
        // Green for verified match, Red for unknown / spoof alert
        Scalar boxColor = result.isMatched() ?
                new Scalar(0, 255, 0, 0) : // Green
                new Scalar(0, 0, 255, 0);  // Red

        // Draw Bounding Box
        opencv_imgproc.rectangle(frame,
                new Point(face.getX(), face.getY()),
                new Point(face.getX() + face.getWidth(), face.getY() + face.getHeight()),
                boxColor, 2, opencv_imgproc.LINE_AA, 0);

        // Draw 5 Facial Landmarks (Cyan dots)
        FaceLandmarks lm = face.getLandmarks();
        if (lm != null) {
            Scalar landmarkColor = new Scalar(255, 255, 0, 0); // Cyan
            drawPoint(frame, lm.getRightEye(), landmarkColor);
            drawPoint(frame, lm.getLeftEye(), landmarkColor);
            drawPoint(frame, lm.getNoseTip(), landmarkColor);
            drawPoint(frame, lm.getRightMouth(), landmarkColor);
            drawPoint(frame, lm.getLeftMouth(), landmarkColor);
        }

        // Draw Info Banner above face
        String label = result.isMatched() ?
                String.format("%s (%.1f%%)", result.getSubjectName(), result.getSimilarity() * 100) :
                "Desconhecido";

        int textY = Math.max(20, face.getY() - 10);
        // Background banner box for legibility
        opencv_imgproc.rectangle(frame,
                new Point(face.getX(), textY - 18),
                new Point(face.getX() + Math.max(160, face.getWidth()), textY + 4),
                boxColor, -1, opencv_imgproc.LINE_AA, 0);

        opencv_imgproc.putText(frame, label,
                new Point(face.getX() + 4, textY - 2),
                opencv_imgproc.FONT_HERSHEY_SIMPLEX, 0.50,
                new Scalar(0, 0, 0, 0), 1, opencv_imgproc.LINE_AA, false);
    }

    private void drawPoint(Mat frame, Point2D pt, Scalar color) {
        if (pt != null) {
            opencv_imgproc.circle(frame,
                    new Point((int) pt.getX(), (int) pt.getY()),
                    3, color, -1, opencv_imgproc.LINE_AA, 0);
        }
    }

    @Override
    public void close() {
        if (faceDetector != null) faceDetector.close();
        if (embeddingExtractor != null) embeddingExtractor.close();
    }
}
