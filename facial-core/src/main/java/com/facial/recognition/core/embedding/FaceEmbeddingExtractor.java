package com.facial.recognition.core.embedding;

import com.facial.recognition.core.exception.FacialRecognitionException;
import com.facial.recognition.core.model.DetectedFace;
import org.bytedeco.opencv.opencv_core.Mat;

/**
 * Interface for extracting high-dimensional biometric facial signatures (embeddings)
 * from detected and aligned face images.
 */
public interface FaceEmbeddingExtractor extends AutoCloseable {

    /**
     * Extracts a normalized high-dimensional facial embedding vector from the original image
     * and its detected face metadata (bounding box and 5 landmarks).
     */
    float[] extractEmbedding(Mat originalImage, DetectedFace detectedFace) throws FacialRecognitionException;

    /**
     * Extracts a normalized facial embedding vector from an already aligned and cropped face image.
     */
    float[] extractEmbedding(Mat alignedFace) throws FacialRecognitionException;

    /**
     * Aligns and crops the facial region according to landmark geometry into canonical size (112x112).
     */
    Mat alignAndCropFace(Mat originalImage, DetectedFace detectedFace) throws FacialRecognitionException;

    /**
     * Returns the dimensionality of the generated feature vector (e.g., 128 for SFace, 512 for FaceNet/ArcFace).
     */
    int getEmbeddingDimension();

    @Override
    default void close() {
        // Default no-op
    }
}
