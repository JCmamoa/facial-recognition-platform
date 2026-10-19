package com.facial.recognition.dl4j;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;

class BiometricNeuralClassifierTest {

    @Test
    @DisplayName("BiometricNeuralClassifier trains and predicts identity classes")
    void testTrainAndPredict() {
        int dim = 16;
        int numClasses = 3;
        BiometricNeuralClassifier classifier = new BiometricNeuralClassifier(dim, numClasses);

        // Generate synthetic embedding dataset for 3 subjects
        List<float[]> embeddings = new ArrayList<>();
        List<Integer> labels = new ArrayList<>();

        for (int i = 0; i < 30; i++) {
            int cls = i % numClasses;
            float[] vec = new float[dim];
            // Signal distinct dimension for each class
            vec[cls] = 1.0f + (float) (Math.random() * 0.1);
            vec[(cls + 1) % dim] = 0.2f * (float) Math.random();
            embeddings.add(vec);
            labels.add(cls);
        }

        // Train for 15 epochs
        classifier.train(embeddings, labels, 15);

        // Test prediction with sample vector for class 1
        float[] probe = new float[dim];
        probe[1] = 1.0f;

        BiometricNeuralClassifier.ClassificationOutput result = classifier.predict(probe);

        assertThat(result).isNotNull();
        assertThat(result.getProbabilities()).hasSize(numClasses);
        assertThat(result.getPredictedClass()).isEqualTo(1);
        assertThat(result.getConfidence()).isGreaterThan(0.50);
    }
}
