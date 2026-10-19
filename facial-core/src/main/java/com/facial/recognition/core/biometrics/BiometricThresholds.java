package com.facial.recognition.core.biometrics;

/**
 * Standard biometric verification thresholds for SFace, ArcFace, and deep metric models.
 */
public final class BiometricThresholds {

    private BiometricThresholds() {}

    /**
     * Default Cosine Similarity threshold for SFace model.
     * OpenCV SFace recommended benchmark threshold is ~0.363.
     * In enterprise biometric access control, 0.40 - 0.50 provides high false acceptance rejection.
     */
    public static final double DEFAULT_COSINE_SIMILARITY_THRESHOLD = 0.40;

    /**
     * Default Euclidean Distance (L2) threshold for SFace model.
     * OpenCV SFace recommended benchmark threshold is ~1.128.
     * Scores lower than this threshold are considered genuine matches.
     */
    public static final double DEFAULT_EUCLIDEAN_DISTANCE_THRESHOLD = 1.128;

    /**
     * Strict Cosine Similarity threshold for high-security applications (banking, physical security).
     */
    public static final double STRICT_COSINE_SIMILARITY_THRESHOLD = 0.55;
}
