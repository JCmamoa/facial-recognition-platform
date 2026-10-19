package com.facial.recognition.core.biometrics;

import com.facial.recognition.core.model.BiometricProfile;
import com.facial.recognition.core.model.RecognitionResult;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;

class InMemoryVectorDatabaseTest {

    private InMemoryVectorDatabase vectorDb;

    @BeforeEach
    void setUp() {
        vectorDb = new InMemoryVectorDatabase();

        // Enroll subject 1: Carlos (Unit vector along dim 0)
        float[] carlosEmb = new float[]{1.0f, 0.0f, 0.0f, 0.0f};
        BiometricProfile carlos = BiometricProfile.builder()
                .id("USR-001")
                .name("Carlos Silva")
                .role("Security Lead")
                .department("Engineering")
                .embeddings(new ArrayList<>(List.of(carlosEmb)))
                .build();
        vectorDb.enroll(carlos);

        // Enroll subject 2: Mariana (Unit vector along dim 1)
        float[] marianaEmb = new float[]{0.0f, 1.0f, 0.0f, 0.0f};
        BiometricProfile mariana = BiometricProfile.builder()
                .id("USR-002")
                .name("Mariana Souza")
                .role("AI Architect")
                .department("R&D")
                .embeddings(new ArrayList<>(List.of(marianaEmb)))
                .build();
        vectorDb.enroll(mariana);
    }

    @Test
    @DisplayName("1:1 Verification should succeed for matching subject")
    void testVerifySuccess() {
        float[] probe = new float[]{0.95f, 0.05f, 0.0f, 0.0f}; // Very close to Carlos
        RecognitionResult result = vectorDb.verify("USR-001", probe, 0.70);

        assertThat(result.isMatched()).isTrue();
        assertThat(result.getSubjectName()).isEqualTo("Carlos Silva");
        assertThat(result.getSimilarity()).isGreaterThan(0.90);
    }

    @Test
    @DisplayName("1:1 Verification should fail if similarity is below threshold")
    void testVerifyFailure() {
        float[] probe = new float[]{0.1f, 0.95f, 0.0f, 0.0f}; // Probe is Mariana, not Carlos
        RecognitionResult result = vectorDb.verify("USR-001", probe, 0.70);

        assertThat(result.isMatched()).isFalse();
    }

    @Test
    @DisplayName("1:N Identification should match closest enrolled subject")
    void testIdentifySuccess() {
        float[] probe = new float[]{0.05f, 0.98f, 0.0f, 0.0f};
        RecognitionResult result = vectorDb.identify(probe, 0.50);

        assertThat(result.isMatched()).isTrue();
        assertThat(result.getSubjectId()).isEqualTo("USR-002");
        assertThat(result.getSubjectName()).isEqualTo("Mariana Souza");
    }

    @Test
    @DisplayName("1:N Identification returns UNKNOWN when no subject exceeds threshold")
    void testIdentifyUnknown() {
        float[] probe = new float[]{0.0f, 0.0f, 1.0f, 0.0f}; // Orthogonal to both
        RecognitionResult result = vectorDb.identify(probe, 0.50);

        assertThat(result.isMatched()).isFalse();
        assertThat(result.getSubjectId()).isEqualTo("UNKNOWN");
    }

    @Test
    @DisplayName("Top-K search ranks candidates by descending similarity")
    void testSearchTopK() {
        float[] probe = new float[]{0.8f, 0.6f, 0.0f, 0.0f}; // Angle closer to Carlos than Mariana
        List<RecognitionResult> topK = vectorDb.searchTopK(probe, 2, 0.10);

        assertThat(topK).hasSize(2);
        assertThat(topK.get(0).getSubjectId()).isEqualTo("USR-001");
        assertThat(topK.get(1).getSubjectId()).isEqualTo("USR-002");
        assertThat(topK.get(0).getSimilarity()).isGreaterThan(topK.get(1).getSimilarity());
    }
}
