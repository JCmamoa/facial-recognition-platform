package com.facial.recognition.core.embedding;

/**
 * Utility for normalizing and manipulating high-dimensional biometric vectors.
 */
public final class EmbeddingNormalizer {

    private EmbeddingNormalizer() {}

    /**
     * Normalizes a float vector to unit length (L2 norm): v / ||v||_2.
     * Ensures cosine similarity can be computed via simple dot product.
     */
    public static float[] l2Normalize(float[] vector) {
        if (vector == null || vector.length == 0) {
            return new float[0];
        }
        double sumSq = 0.0;
        for (float val : vector) {
            sumSq += val * val;
        }
        double norm = Math.sqrt(sumSq);
        if (norm < 1e-10) {
            return vector.clone();
        }
        float[] normalized = new float[vector.length];
        for (int i = 0; i < vector.length; i++) {
            normalized[i] = (float) (vector[i] / norm);
        }
        return normalized;
    }

    /**
     * Computes the L2 norm (magnitude) of a vector.
     */
    public static double magnitude(float[] vector) {
        if (vector == null || vector.length == 0) return 0.0;
        double sumSq = 0.0;
        for (float val : vector) {
            sumSq += val * val;
        }
        return Math.sqrt(sumSq);
    }
}
