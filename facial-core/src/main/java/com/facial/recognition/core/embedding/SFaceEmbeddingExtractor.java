package com.facial.recognition.core.embedding;

import com.facial.recognition.core.capture.OpenCVUtils;
import com.facial.recognition.core.exception.FacialRecognitionException;
import com.facial.recognition.core.exception.ModelInitializationException;
import com.facial.recognition.core.model.DetectedFace;
import com.facial.recognition.core.model.FaceLandmarks;
import lombok.extern.slf4j.Slf4j;
import org.bytedeco.javacpp.indexer.FloatRawIndexer;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Rect;
import org.bytedeco.opencv.opencv_core.Size;
import org.bytedeco.opencv.opencv_objdetect.FaceRecognizerSF;
import org.bytedeco.opencv.global.opencv_core;
import org.bytedeco.opencv.global.opencv_imgproc;

import java.io.File;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;

/**
 * Deep Neural Network facial feature extractor implementing the SFace (SphereFace/ArcFace inspired)
 * deep metric learning architecture via OpenCV's FaceRecognizerSF.
 * Generates compact, discriminative 128-dimensional biometric embeddings.
 */
@Slf4j
public class SFaceEmbeddingExtractor implements FaceEmbeddingExtractor {

    private static final int EMBEDDING_DIMENSION = 128;

    private final String modelPath;
    private FaceRecognizerSF recognizer;

    public SFaceEmbeddingExtractor() {
        this(resolveDefaultModelPath());
    }

    public SFaceEmbeddingExtractor(String modelPath) {
        this.modelPath = modelPath;
        initRecognizer();
    }

    private static String resolveDefaultModelPath() {
        String customPath = System.getProperty("facial.sface.model.path");
        if (customPath == null || customPath.isBlank()) {
            customPath = System.getenv("FACIAL_SFACE_MODEL_PATH");
        }
        if (customPath != null && !customPath.isBlank()) {
            File file = new File(customPath);
            if (file.isFile()) {
                return file.getAbsolutePath();
            }
        }
        try (InputStream in = SFaceEmbeddingExtractor.class.getResourceAsStream("/models/face_recognition_sface.onnx")) {
            if (in != null) {
                Path tempFile = Files.createTempFile("sface_", ".onnx");
                tempFile.toFile().deleteOnExit();
                Files.copy(in, tempFile, StandardCopyOption.REPLACE_EXISTING);
                return tempFile.toAbsolutePath().toString();
            }
        } catch (Exception e) {
            log.warn("Failed to extract default SFace model from classpath: {}", e.getMessage());
        }
        return customPath == null || customPath.isBlank()
                ? "models/face_recognition_sface.onnx"
                : customPath;
    }

    private synchronized void initRecognizer() {
        try {
            log.info("Loading SFace FaceRecognizerSF from: {}", modelPath);
            File file = new File(modelPath);
            if (!file.exists()) {
                throw new ModelInitializationException("SFace model file not found at " + modelPath);
            }
            recognizer = FaceRecognizerSF.create(
                    modelPath,
                    "",
                    0, // DNN_BACKEND_DEFAULT
                    0  // DNN_TARGET_CPU
            );
            log.info("SFace FaceRecognizerSF initialized successfully.");
        } catch (Exception e) {
            throw new ModelInitializationException("Could not initialize SFace FaceRecognizerSF: " + e.getMessage(), e);
        }
    }

    @Override
    public synchronized float[] extractEmbedding(Mat originalImage, DetectedFace detectedFace) throws FacialRecognitionException {
        if (originalImage == null || originalImage.empty() || detectedFace == null) {
            return new float[EMBEDDING_DIMENSION];
        }

        try {
            Mat alignedFace = alignAndCropFace(originalImage, detectedFace);
            if (alignedFace == null || alignedFace.empty()) {
                return new float[EMBEDDING_DIMENSION];
            }

            // Save aligned image bytes to the detected face for UI preview
            byte[] jpegBytes = OpenCVUtils.matToByteArray(alignedFace, ".jpg");
            detectedFace.setAlignedFaceImageBytes(jpegBytes);

            return extractEmbedding(alignedFace);
        } catch (Exception e) {
            log.error("Failed to extract embedding for detected face", e);
            throw new FacialRecognitionException("Embedding extraction failed: " + e.getMessage(), e);
        }
    }

    @Override
    public synchronized float[] extractEmbedding(Mat alignedFace) throws FacialRecognitionException {
        if (alignedFace == null || alignedFace.empty()) {
            return new float[EMBEDDING_DIMENSION];
        }

        try {
            Mat feature = new Mat();
            recognizer.feature(alignedFace, feature);

            if (feature.empty() || feature.cols() != EMBEDDING_DIMENSION) {
                log.warn("SFace feature extraction produced empty or invalid dimensions (cols={})", feature.cols());
                return new float[EMBEDDING_DIMENSION];
            }

            float[] rawEmbedding = new float[EMBEDDING_DIMENSION];
            FloatRawIndexer indexer = feature.createIndexer();
            try {
                for (int i = 0; i < EMBEDDING_DIMENSION; i++) {
                    rawEmbedding[i] = indexer.get(0, i);
                }
            } finally {
                indexer.release();
            }

            // L2-normalize vector to lie on the unit hypersphere
            return EmbeddingNormalizer.l2Normalize(rawEmbedding);
        } catch (Exception e) {
            log.error("Error during SFace feature inference", e);
            throw new FacialRecognitionException("SFace feature extraction failed: " + e.getMessage(), e);
        }
    }

    @Override
    public synchronized Mat alignAndCropFace(Mat originalImage, DetectedFace detectedFace) throws FacialRecognitionException {
        if (originalImage == null || originalImage.empty() || detectedFace == null) {
            return null;
        }

        try {
            FaceLandmarks lm = detectedFace.getLandmarks();
            if (lm != null && lm.getRightEye() != null && lm.getLeftEye() != null &&
                lm.getNoseTip() != null && lm.getRightMouth() != null && lm.getLeftMouth() != null) {

                // Build 1x15 Mat expected by FaceRecognizerSF.alignCrop
                Mat faceBoxAndLandmarks = new Mat(1, 15, opencv_core.CV_32F);
                FloatRawIndexer indexer = faceBoxAndLandmarks.createIndexer();
                try {
                    indexer.put(0, 0, (float) detectedFace.getX());
                    indexer.put(0, 1, (float) detectedFace.getY());
                    indexer.put(0, 2, (float) detectedFace.getWidth());
                    indexer.put(0, 3, (float) detectedFace.getHeight());
                    indexer.put(0, 4, lm.getRightEye().getX());
                    indexer.put(0, 5, lm.getRightEye().getY());
                    indexer.put(0, 6, lm.getLeftEye().getX());
                    indexer.put(0, 7, lm.getLeftEye().getY());
                    indexer.put(0, 8, lm.getNoseTip().getX());
                    indexer.put(0, 9, lm.getNoseTip().getY());
                    indexer.put(0, 10, lm.getRightMouth().getX());
                    indexer.put(0, 11, lm.getRightMouth().getY());
                    indexer.put(0, 12, lm.getLeftMouth().getX());
                    indexer.put(0, 13, lm.getLeftMouth().getY());
                    indexer.put(0, 14, detectedFace.getConfidence());
                } finally {
                    indexer.release();
                }

                Mat aligned = new Mat();
                recognizer.alignCrop(originalImage, faceBoxAndLandmarks, aligned);
                return aligned;
            } else {
                // Fallback cropping to 112x112
                int x = Math.max(0, detectedFace.getX());
                int y = Math.max(0, detectedFace.getY());
                int w = Math.min(originalImage.cols() - x, detectedFace.getWidth());
                int h = Math.min(originalImage.rows() - y, detectedFace.getHeight());

                if (w <= 0 || h <= 0) return null;

                Rect roi = new Rect(x, y, w, h);
                Mat cropped = new Mat(originalImage, roi);
                Mat resized = new Mat();
                opencv_imgproc.resize(cropped, resized, new Size(112, 112));
                return resized;
            }
        } catch (Exception e) {
            log.error("Failed to align and crop face", e);
            throw new FacialRecognitionException("Face alignment failed: " + e.getMessage(), e);
        }
    }

    @Override
    public int getEmbeddingDimension() {
        return EMBEDDING_DIMENSION;
    }

    @Override
    public synchronized void close() {
        if (recognizer != null) {
            recognizer.close();
            recognizer = null;
        }
    }
}
