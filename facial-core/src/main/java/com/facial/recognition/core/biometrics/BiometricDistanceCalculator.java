package com.facial.recognition.core.biometrics;

/**
 * Metric calculations for comparing high-dimensional biometric facial vectors.
 */
public final class BiometricDistanceCalculator {

    private BiometricDistanceCalculator() {}

    /**
     * Calculates the Cosine Similarity between two embedding vectors:
     * CosSim(A, B) = (A . B) / (||A|| * ||B||)
     * Result is in range [-1.0, 1.0], where 1.0 indicates identical directions.
     */
    public static double cosineSimilarity(float[] v1, float[] v2) {
        if (v1 == null || v2 == null || v1.length == 0 || v1.length != v2.length) {
            return 0.0;
        }

        double dotProduct = 0.0;
        double normA = 0.0;
        double normB = 0.0;

        for (int i = 0; i < v1.length; i++) {
            dotProduct += v1[i] * v2[i];
            normA += v1[i] * v1[i];
            normB += v2[i] * v2[i];
        }

        if (normA <= 1e-12 || normB <= 1e-12) {
            return 0.0;
        }

        double sim = dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
        // Clamp to [-1.0, 1.0] to prevent floating point inaccuracies
        return Math.max(-1.0, Math.min(1.0, sim));
    }

    /**
     * Calculates Cosine Distance: 1.0 - CosineSimilarity.
     */
    public static double cosineDistance(float[] v1, float[] v2) {
        return 1.0 - cosineSimilarity(v1, v2);
    }

    /**
     * Calculates the Euclidean (L2) Distance between two embedding vectors:
     * L2(A, B) = sqrt( sum( (A_i - B_i)^2 ) )
     */
    public static double euclideanDistance(float[] v1, float[] v2) {
        if (v1 == null || v2 == null || v1.length == 0 || v1.length != v2.length) {
            return Double.MAX_VALUE;
        }

        double sumSq = 0.0;
        for (int i = 0; i < v1.length; i++) {
            double diff = v1[i] - v2[i];
            sumSq += diff * diff;
        }

        return Math.sqrt(sumSq);
    }

    /**
     * Calculates Manhattan (L1) Distance between two embedding vectors:
     * L1(A, B) = sum( |A_i - B_i| )
     */
    public static double manhattanDistance(float[] v1, float[] v2) {
        if (v1 == null || v2 == null || v1.length == 0 || v1.length != v2.length) {
            return Double.MAX_VALUE;
        }

        double sum = 0.0;
        for (int i = 0; i < v1.length; i++) {
            sum += Math.abs(v1[i] - v2[i]);
        }

        return sum;
    }
}
