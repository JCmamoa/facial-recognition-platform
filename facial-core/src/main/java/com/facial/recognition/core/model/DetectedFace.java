package com.facial.recognition.core.model;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.io.Serializable;

/**
 * Model representing a detected face with its bounding box, confidence score,
 * landmark coordinates, extracted facial signature (embedding), and liveness flags.
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class DetectedFace implements Serializable {
    private static final long serialVersionUID = 1L;

    // Bounding Box
    private int x;
    private int y;
    private int width;
    private int height;

    // Detection confidence (0.0 to 1.0)
    private float confidence;

    // 5-point facial landmarks
    private FaceLandmarks landmarks;

    // 128-dimensional biometric embedding vector
    private float[] embedding;

    // Cropped and aligned face image bytes (JPEG / PNG format)
    private byte[] alignedFaceImageBytes;

    // Anti-spoofing and liveness score
    @Builder.Default
    private float livenessScore = 1.0f;

    @Builder.Default
    private boolean livenessPassed = true;

    // Track ID for multi-frame tracking
    private Integer trackingId;

    public double getCenterX() {
        return x + (width / 2.0);
    }

    public double getCenterY() {
        return y + (height / 2.0);
    }

    public double getArea() {
        return width * height;
    }
}
