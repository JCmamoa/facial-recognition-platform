package com.facial.recognition.persistence.entity;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import jakarta.persistence.*;
import lombok.*;

import java.io.Serializable;
import java.time.LocalDateTime;

/**
 * Entity for persisting 128-dimensional biometric facial signature vectors.
 */
@Entity
@Table(name = "face_signatures")
@Getter
@Setter
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class FaceSignatureEntity implements Serializable {

    private static final ObjectMapper MAPPER = new ObjectMapper();

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "user_id")
    private UserProfileEntity userProfile;

    @Column(name = "dimension", nullable = false)
    @Builder.Default
    private int dimension = 128;

    @Lob
    @Column(name = "vector_json", nullable = false, columnDefinition = "CLOB")
    private String vectorJson;

    @Column(name = "captured_at", nullable = false)
    @Builder.Default
    private LocalDateTime capturedAt = LocalDateTime.now();

    public float[] toFloatArray() {
        if (vectorJson == null || vectorJson.isBlank()) {
            return new float[0];
        }
        try {
            return MAPPER.readValue(vectorJson, float[].class);
        } catch (JsonProcessingException e) {
            throw new IllegalStateException("Failed to deserialize biometric vector", e);
        }
    }

    public static FaceSignatureEntity fromFloatArray(float[] vector) {
        try {
            String json = MAPPER.writeValueAsString(vector);
            return FaceSignatureEntity.builder()
                    .dimension(vector != null ? vector.length : 0)
                    .vectorJson(json)
                    .capturedAt(LocalDateTime.now())
                    .build();
        } catch (JsonProcessingException e) {
            throw new IllegalArgumentException("Failed to serialize biometric vector", e);
        }
    }
}
