package com.facial.recognition.core.model;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.io.Serializable;
import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Biometric Profile representing a registered subject in the vector database.
 * Supports multiple embeddings (facial signatures) per subject to account for
 * illumination changes, facial hair, glasses, or varying angles.
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class BiometricProfile implements Serializable {
    private static final long serialVersionUID = 1L;

    private String id;
    private String name;
    private String documentId;
    private String role;
    private String department;

    @Builder.Default
    private List<float[]> embeddings = new ArrayList<>();

    @Builder.Default
    private LocalDateTime createdAt = LocalDateTime.now();

    @Builder.Default
    private LocalDateTime updatedAt = LocalDateTime.now();

    @Builder.Default
    private boolean active = true;

    @Builder.Default
    private Map<String, String> metadata = new HashMap<>();

    public void addEmbedding(float[] embedding) {
        if (embedding != null && embedding.length > 0) {
            this.embeddings.add(embedding);
            this.updatedAt = LocalDateTime.now();
        }
    }

    /**
     * Computes the mean (centroid) embedding vector across all registered samples.
     */
    public float[] computeCentroidEmbedding() {
        if (embeddings == null || embeddings.isEmpty()) {
            return new float[0];
        }
        int dim = embeddings.get(0).length;
        float[] centroid = new float[dim];
        for (float[] emb : embeddings) {
            for (int i = 0; i < dim; i++) {
                centroid[i] += emb[i];
            }
        }
        int count = embeddings.size();
        double normSq = 0.0;
        for (int i = 0; i < dim; i++) {
            centroid[i] /= count;
            normSq += centroid[i] * centroid[i];
        }
        // Normalize centroid to unit sphere (L2 norm)
        double norm = Math.sqrt(normSq);
        if (norm > 1e-8) {
            for (int i = 0; i < dim; i++) {
                centroid[i] = (float) (centroid[i] / norm);
            }
        }
        return centroid;
    }
}
