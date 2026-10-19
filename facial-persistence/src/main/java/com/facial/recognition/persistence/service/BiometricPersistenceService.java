package com.facial.recognition.persistence.service;

import com.facial.recognition.core.biometrics.VectorDatabase;
import com.facial.recognition.core.model.BiometricProfile;
import com.facial.recognition.persistence.entity.FaceSignatureEntity;
import com.facial.recognition.persistence.entity.UserProfileEntity;
import com.facial.recognition.persistence.repository.UserProfileRepository;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.ArrayList;
import java.util.List;
import java.util.Optional;

/**
 * Service orchestrating biometric persistence, synchronization with in-memory VectorDatabase,
 * and profile management.
 */
@Slf4j
@Service
@RequiredArgsConstructor
public class BiometricPersistenceService {

    private final UserProfileRepository userProfileRepository;

    /**
     * Synchronizes all active profiles from the database into the in-memory vector database.
     */
    @Transactional(readOnly = true)
    public int syncToVectorDatabase(VectorDatabase vectorDatabase) {
        if (vectorDatabase == null) return 0;

        List<UserProfileEntity> activeProfiles = userProfileRepository.findByActiveTrue();
        int loaded = 0;
        for (UserProfileEntity entity : activeProfiles) {
            BiometricProfile profile = entity.toBiometricProfile();
            if (!profile.getEmbeddings().isEmpty()) {
                vectorDatabase.enroll(profile);
                loaded++;
            }
        }
        log.info("Synchronized {} active biometric profiles to in-memory VectorDatabase.", loaded);
        return loaded;
    }

    /**
     * Saves or updates a profile in the persistent database and enrolls into VectorDatabase.
     */
    @Transactional
    public BiometricProfile saveProfile(BiometricProfile profile, VectorDatabase vectorDatabase) {
        if (profile == null || profile.getId() == null) {
            throw new IllegalArgumentException("Invalid biometric profile");
        }

        UserProfileEntity entity = userProfileRepository.findById(profile.getId())
                .orElseGet(() -> UserProfileEntity.builder()
                        .id(profile.getId())
                        .name(profile.getName())
                        .documentId(profile.getDocumentId())
                        .role(profile.getRole())
                        .department(profile.getDepartment())
                        .active(profile.isActive())
                        .signatures(new ArrayList<>())
                        .build());

        entity.setName(profile.getName());
        entity.setDocumentId(profile.getDocumentId());
        entity.setRole(profile.getRole());
        entity.setDepartment(profile.getDepartment());
        entity.setActive(profile.isActive());

        // Sync signatures
        entity.getSignatures().clear();
        if (profile.getEmbeddings() != null) {
            for (float[] emb : profile.getEmbeddings()) {
                entity.addSignature(FaceSignatureEntity.fromFloatArray(emb));
            }
        }

        UserProfileEntity saved = userProfileRepository.save(entity);
        BiometricProfile savedProfile = saved.toBiometricProfile();

        if (vectorDatabase != null && savedProfile.isActive()) {
            vectorDatabase.enroll(savedProfile);
        }

        log.info("Persisted biometric profile for user: {} (id: {})", saved.getName(), saved.getId());
        return savedProfile;
    }

    @Transactional(readOnly = true)
    public Optional<BiometricProfile> findById(String id) {
        return userProfileRepository.findById(id).map(UserProfileEntity::toBiometricProfile);
    }

    @Transactional(readOnly = true)
    public List<BiometricProfile> findAll() {
        return userProfileRepository.findAll().stream()
                .map(UserProfileEntity::toBiometricProfile)
                .toList();
    }

    @Transactional
    public boolean deleteProfile(String id, VectorDatabase vectorDatabase) {
        if (userProfileRepository.existsById(id)) {
            userProfileRepository.deleteById(id);
            if (vectorDatabase != null) {
                vectorDatabase.remove(id);
            }
            log.info("Deleted profile id: {}", id);
            return true;
        }
        return false;
    }
}
