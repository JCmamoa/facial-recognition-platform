package com.facial.recognition.dl4j;

import lombok.extern.slf4j.Slf4j;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.util.List;

/**
 * High-performance vector operations and tensor manipulations using ND4J (Native BLAS accelerated).
 */
@Slf4j
public final class Nd4jVectorUtils {

    private Nd4jVectorUtils() {}

    /**
     * Converts a float array to an ND4J 1D vector (1, N).
     */
    public static INDArray toNd4jVector(float[] vector) {
        if (vector == null || vector.length == 0) {
            return Nd4j.empty();
        }
        return Nd4j.create(vector, new long[]{1, vector.length});
    }

    /**
     * Converts a batch of float arrays to an ND4J 2D matrix (batchSize, dimension).
     */
    public static INDArray toNd4jMatrix(List<float[]> vectors) {
        if (vectors == null || vectors.isEmpty()) {
            return Nd4j.empty();
        }
        int rows = vectors.size();
        int cols = vectors.get(0).length;
        float[] flatData = new float[rows * cols];
        for (int i = 0; i < rows; i++) {
            float[] v = vectors.get(i);
            System.arraycopy(v, 0, flatData, i * cols, cols);
        }
        return Nd4j.create(flatData, new long[]{rows, cols});
    }

    /**
     * Converts an ND4J vector back to a Java float array.
     */
    public static float[] toFloatArray(INDArray array) {
        if (array == null || array.isEmpty()) {
            return new float[0];
        }
        return array.data().asFloat();
    }

    /**
     * Normalizes an ND4J vector or matrix rows to unit length (L2 norm).
     */
    public static INDArray l2Normalize(INDArray vector) {
        if (vector == null || vector.isEmpty()) {
            return vector;
        }
        INDArray norm = vector.norm2(1);
        return vector.divColumnVector(norm);
    }

    /**
     * Computes Cosine Similarity between two ND4J vectors using native BLAS dot product.
     */
    public static double cosineSimilarity(INDArray v1, INDArray v2) {
        if (v1 == null || v2 == null || v1.isEmpty() || v2.isEmpty()) {
            return 0.0;
        }
        return Transforms.cosineSim(v1, v2);
    }

    /**
     * Computes Euclidean Distance between two ND4J vectors.
     */
    public static double euclideanDistance(INDArray v1, INDArray v2) {
        if (v1 == null || v2 == null || v1.isEmpty() || v2.isEmpty()) {
            return Double.MAX_VALUE;
        }
        return Transforms.euclideanDistance(v1, v2);
    }

    /**
     * Computes the batch pairwise cosine similarity matrix between probe vector (1, D)
     * and gallery matrix (N, D).
     * Output is a (1, N) tensor containing similarity of probe against every registered subject.
     */
    public static INDArray batchCosineSimilarity(INDArray probeNormalized, INDArray galleryNormalized) {
        if (probeNormalized == null || galleryNormalized == null || probeNormalized.isEmpty() || galleryNormalized.isEmpty()) {
            return Nd4j.empty();
        }
        // Since both probe and gallery vectors are L2-normalized: CosSim(probe, gallery_i) = probe * gallery_i^T
        return probeNormalized.mmul(galleryNormalized.transpose());
    }
}
