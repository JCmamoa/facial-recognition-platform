package com.facial.recognition.dl4j;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.within;

class Nd4jVectorUtilsTest {

    @Test
    @DisplayName("Convert between float array and ND4J vector")
    void testArrayConversion() {
        float[] original = new float[]{1.5f, 2.5f, -3.0f, 4.0f};
        INDArray nd4j = Nd4jVectorUtils.toNd4jVector(original);

        assertThat(nd4j.shape()).containsExactly(1L, 4L);

        float[] converted = Nd4jVectorUtils.toFloatArray(nd4j);
        assertThat(converted).containsExactly(original);
    }

    @Test
    @DisplayName("ND4J L2 normalization scales vector to norm of 1.0")
    void testL2Normalization() {
        float[] v = new float[]{3.0f, 4.0f};
        INDArray nd4j = Nd4jVectorUtils.toNd4jVector(v);
        INDArray normalized = Nd4jVectorUtils.l2Normalize(nd4j);

        double norm = normalized.norm2Number().doubleValue();
        assertThat(norm).isCloseTo(1.0, within(1e-5));

        float[] res = Nd4jVectorUtils.toFloatArray(normalized);
        assertThat(res[0]).isCloseTo(0.6f, within(1e-4f));
        assertThat(res[1]).isCloseTo(0.8f, within(1e-4f));
    }

    @Test
    @DisplayName("ND4J cosine similarity computes exact dot product of normalized vectors")
    void testCosineSimilarity() {
        INDArray v1 = Nd4jVectorUtils.toNd4jVector(new float[]{1.0f, 0.0f});
        INDArray v2 = Nd4jVectorUtils.toNd4jVector(new float[]{0.0f, 1.0f});

        double sim = Nd4jVectorUtils.cosineSimilarity(v1, v2);
        assertThat(sim).isCloseTo(0.0, within(1e-5));
    }

    @Test
    @DisplayName("Batch cosine similarity computes matrix product of probe against gallery")
    void testBatchCosineSimilarity() {
        INDArray probe = Nd4jVectorUtils.toNd4jVector(new float[]{1.0f, 0.0f, 0.0f});

        List<float[]> gallery = List.of(
                new float[]{1.0f, 0.0f, 0.0f}, // Identical -> 1.0
                new float[]{0.0f, 1.0f, 0.0f}, // Orthogonal -> 0.0
                new float[]{0.7071f, 0.7071f, 0.0f} // 45 deg -> ~0.7071
        );
        INDArray galleryMat = Nd4jVectorUtils.toNd4jMatrix(gallery);

        INDArray batchSim = Nd4jVectorUtils.batchCosineSimilarity(probe, galleryMat);
        assertThat(batchSim.shape()).containsExactly(1L, 3L);

        float[] sims = Nd4jVectorUtils.toFloatArray(batchSim);
        assertThat(sims[0]).isCloseTo(1.0f, within(1e-4f));
        assertThat(sims[1]).isCloseTo(0.0f, within(1e-4f));
        assertThat(sims[2]).isCloseTo(0.7071f, within(1e-3f));
    }
}
