package com.facial.recognition.core.biometrics;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.within;

class BiometricDistanceCalculatorTest {

    @Test
    @DisplayName("Identical vectors should yield cosine similarity 1.0 and euclidean distance 0.0")
    void testIdenticalVectors() {
        float[] v1 = new float[]{0.6f, 0.8f, 0.0f, 0.0f};
        float[] v2 = new float[]{0.6f, 0.8f, 0.0f, 0.0f};

        double cosSim = BiometricDistanceCalculator.cosineSimilarity(v1, v2);
        double euclDist = BiometricDistanceCalculator.euclideanDistance(v1, v2);

        assertThat(cosSim).isCloseTo(1.0, within(1e-5));
        assertThat(euclDist).isCloseTo(0.0, within(1e-5));
    }

    @Test
    @DisplayName("Orthogonal vectors should yield cosine similarity 0.0")
    void testOrthogonalVectors() {
        float[] v1 = new float[]{1.0f, 0.0f, 0.0f};
        float[] v2 = new float[]{0.0f, 1.0f, 0.0f};

        double cosSim = BiometricDistanceCalculator.cosineSimilarity(v1, v2);
        double cosDist = BiometricDistanceCalculator.cosineDistance(v1, v2);

        assertThat(cosSim).isCloseTo(0.0, within(1e-5));
        assertThat(cosDist).isCloseTo(1.0, within(1e-5));
    }

    @Test
    @DisplayName("Opposite vectors should yield cosine similarity -1.0")
    void testOppositeVectors() {
        float[] v1 = new float[]{0.5f, 0.5f};
        float[] v2 = new float[]{-0.5f, -0.5f};

        double cosSim = BiometricDistanceCalculator.cosineSimilarity(v1, v2);
        assertThat(cosSim).isCloseTo(-1.0, within(1e-5));
    }

    @Test
    @DisplayName("Manhattan distance calculation matches sum of absolute differences")
    void testManhattanDistance() {
        float[] v1 = new float[]{1.0f, 2.0f, 3.0f};
        float[] v2 = new float[]{4.0f, 0.0f, 1.0f};

        // |1-4| + |2-0| + |3-1| = 3 + 2 + 2 = 7.0
        double dist = BiometricDistanceCalculator.manhattanDistance(v1, v2);
        assertThat(dist).isCloseTo(7.0, within(1e-5));
    }
}
