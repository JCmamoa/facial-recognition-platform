package com.facial.recognition.persistence.service;

import com.facial.recognition.core.biometrics.InMemoryVectorDatabase;
import com.facial.recognition.core.biometrics.VectorDatabase;
import com.facial.recognition.core.model.BiometricProfile;
import com.facial.recognition.persistence.entity.UserProfileEntity;
import com.facial.recognition.persistence.repository.UserProfileRepository;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import java.util.ArrayList;
import java.util.List;
import java.util.Optional;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.*;

@ExtendWith(MockitoExtension.class)
class BiometricPersistenceServiceTest {

    @Mock
    private UserProfileRepository userProfileRepository;

    @InjectMocks
    private BiometricPersistenceService persistenceService;

    private VectorDatabase vectorDatabase;

    @BeforeEach
    void setUp() {
        vectorDatabase = new InMemoryVectorDatabase();
    }

    @Test
    @DisplayName("saveProfile saves entity and enrolls vector into VectorDatabase")
    void testSaveProfile() {
        float[] sampleEmbedding = new float[]{0.1f, 0.2f, 0.3f};
        BiometricProfile profile = BiometricProfile.builder()
                .id("USR-999")
                .name("Roberto Santos")
                .role("Engineer")
                .department("IT")
                .active(true)
                .embeddings(new ArrayList<>(List.of(sampleEmbedding)))
                .build();

        UserProfileEntity mockEntity = UserProfileEntity.fromBiometricProfile(profile);
        when(userProfileRepository.findById("USR-999")).thenReturn(Optional.empty());
        when(userProfileRepository.save(any(UserProfileEntity.class))).thenReturn(mockEntity);

        BiometricProfile saved = persistenceService.saveProfile(profile, vectorDatabase);

        assertThat(saved).isNotNull();
        assertThat(saved.getId()).isEqualTo("USR-999");
        assertThat(vectorDatabase.findById("USR-999")).isPresent();
        verify(userProfileRepository, times(1)).save(any(UserProfileEntity.class));
    }

    @Test
    @DisplayName("syncToVectorDatabase populates VectorDatabase with active database profiles")
    void testSyncToVectorDatabase() {
        float[] sampleEmbedding = new float[]{0.5f, 0.5f, 0.0f};
        BiometricProfile profile = BiometricProfile.builder()
                .id("USR-101")
                .name("Ana Paula")
                .active(true)
                .embeddings(new ArrayList<>(List.of(sampleEmbedding)))
                .build();

        UserProfileEntity entity = UserProfileEntity.fromBiometricProfile(profile);
        when(userProfileRepository.findByActiveTrue()).thenReturn(List.of(entity));

        int loadedCount = persistenceService.syncToVectorDatabase(vectorDatabase);

        assertThat(loadedCount).isEqualTo(1);
        assertThat(vectorDatabase.findById("USR-101")).isPresent();
    }
}
