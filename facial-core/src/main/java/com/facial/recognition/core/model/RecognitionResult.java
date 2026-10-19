package com.facial.recognition.core.model;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.io.Serializable;

/**
 * Result of a 1:1 biometric verification or 1:N biometric identification.
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class RecognitionResult implements Serializable {
    private static final long serialVersionUID = 1L;

    private String subjectId;
    private String subjectName;
    private String department;
    private String role;

    // Matched flag based on similarity vs threshold
    private boolean matched;

    // Cosine similarity metric [0.0, 1.0]
    private double similarity;

    // Euclidean distance (L2) metric
    private double distance;

    // Detection confidence
    private float detectionConfidence;

    // Associated detected face
    private DetectedFace detectedFace;

    // Inference and processing time in milliseconds
    private long inferenceTimeMs;

    @Builder.Default
    private long timestamp = System.currentTimeMillis();

    public static RecognitionResult unknown(DetectedFace face, long inferenceTimeMs) {
        return RecognitionResult.builder()
                .subjectId("UNKNOWN")
                .subjectName("Desconhecido")
                .matched(false)
                .similarity(0.0)
                .distance(Double.MAX_VALUE)
                .detectionConfidence(face != null ? face.getConfidence() : 0f)
                .detectedFace(face)
                .inferenceTimeMs(inferenceTimeMs)
                .build();
    }
}
