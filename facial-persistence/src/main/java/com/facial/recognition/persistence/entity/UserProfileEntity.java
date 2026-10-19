package com.facial.recognition.persistence.entity;

import com.facial.recognition.core.model.BiometricProfile;
import jakarta.persistence.*;
import lombok.*;

import java.io.Serializable;
import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.List;

/**
 * Entity representing an enrolled user with their registered biometric signatures.
 */
@Entity
@Table(name = "user_profiles")
@Getter
@Setter
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class UserProfileEntity implements Serializable {

    @Id
    @Column(name = "id", length = 64, nullable = false, unique = true)
    private String id;

    @Column(name = "name", nullable = false, length = 128)
    private String name;

    @Column(name = "document_id", length = 64)
    private String documentId;

    @Column(name = "role", length = 64)
    private String role;

    @Column(name = "department", length = 64)
    private String department;

    @Column(name = "active", nullable = false)
    @Builder.Default
    private boolean active = true;

    @OneToMany(mappedBy = "userProfile", cascade = CascadeType.ALL, orphanRemoval = true, fetch = FetchType.EAGER)
    @Builder.Default
    private List<FaceSignatureEntity> signatures = new ArrayList<>();

    @Column(name = "created_at", nullable = false)
    @Builder.Default
    private LocalDateTime createdAt = LocalDateTime.now();

    @Column(name = "updated_at", nullable = false)
    @Builder.Default
    private LocalDateTime updatedAt = LocalDateTime.now();

    public void addSignature(FaceSignatureEntity signature) {
        signatures.add(signature);
        signature.setUserProfile(this);
        this.updatedAt = LocalDateTime.now();
    }

    public BiometricProfile toBiometricProfile() {
        List<float[]> embeddings = new ArrayList<>();
        if (signatures != null) {
            for (FaceSignatureEntity sig : signatures) {
                embeddings.add(sig.toFloatArray());
            }
        }
        return BiometricProfile.builder()
                .id(this.id)
                .name(this.name)
                .documentId(this.documentId)
                .role(this.role)
                .department(this.department)
                .active(this.active)
                .embeddings(embeddings)
                .createdAt(this.createdAt)
                .updatedAt(this.updatedAt)
                .build();
    }

    public static UserProfileEntity fromBiometricProfile(BiometricProfile profile) {
        UserProfileEntity entity = UserProfileEntity.builder()
                .id(profile.getId())
                .name(profile.getName())
                .documentId(profile.getDocumentId())
                .role(profile.getRole())
                .department(profile.getDepartment())
                .active(profile.isActive())
                .createdAt(profile.getCreatedAt() != null ? profile.getCreatedAt() : LocalDateTime.now())
                .updatedAt(profile.getUpdatedAt() != null ? profile.getUpdatedAt() : LocalDateTime.now())
                .signatures(new ArrayList<>())
                .build();

        if (profile.getEmbeddings() != null) {
            for (float[] emb : profile.getEmbeddings()) {
                entity.addSignature(FaceSignatureEntity.fromFloatArray(emb));
            }
        }
        return entity;
    }
}
