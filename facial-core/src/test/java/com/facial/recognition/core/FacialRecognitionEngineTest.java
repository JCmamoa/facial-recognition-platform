package com.facial.recognition.core;

import com.facial.recognition.core.biometrics.InMemoryVectorDatabase;
import com.facial.recognition.core.capture.SimulatedFrameSource;
import com.facial.recognition.core.detection.CompositeFaceDetector;
import com.facial.recognition.core.embedding.SFaceEmbeddingExtractor;
import com.facial.recognition.core.liveness.LivenessDetector;
import com.facial.recognition.core.model.BiometricProfile;
import com.facial.recognition.core.model.RecognitionResult;
import com.facial.recognition.core.preprocessing.ImagePreprocessor;
import org.bytedeco.opencv.opencv_core.Mat;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;

class FacialRecognitionEngineTest {

    private FacialRecognitionEngine engine;
    private SimulatedFrameSource simulatedSource;

    @BeforeEach
    void setUp() {
        engine = new FacialRecognitionEngine(
                new CompositeFaceDetector(),
                new SFaceEmbeddingExtractor(),
                new ImagePreprocessor(),
                new LivenessDetector(),
                new InMemoryVectorDatabase()
        );
        simulatedSource = new SimulatedFrameSource(640, 480, 30.0);
        simulatedSource.start();
    }

    @AfterEach
    void tearDown() {
        if (simulatedSource != null) {
            simulatedSource.stop();
        }
        if (engine != null) {
            engine.close();
        }
    }

    @Test
    @DisplayName("Simulated frames can be captured and processed by the engine")
    void testProcessSimulatedFrame() {
        Mat frame = simulatedSource.captureFrame();
        assertThat(frame).isNotNull();
        assertThat(frame.empty()).isFalse();

        List<RecognitionResult> results = engine.processFrame(frame, true);
        assertThat(results).isNotNull();
        // The frame has been processed and annotated without exceptions
    }

    @Test
    @DisplayName("Manual profile enrollment and retrieval works in engine")
    void testEnrollAndVerifyProfile() {
        float[] sampleEmbedding = new float[128];
        sampleEmbedding[0] = 1.0f; // Unit vector

        BiometricProfile profile = BiometricProfile.builder()
                .id("USR-100")
                .name("Alexandre Costa")
                .documentId("123.456.789-00")
                .role("Security Analyst")
                .department("Operations")
                .embeddings(new ArrayList<>(List.of(sampleEmbedding)))
                .build();

        engine.getVectorDatabase().enroll(profile);

        RecognitionResult verifyResult = engine.getVectorDatabase().verify("USR-100", sampleEmbedding, 0.50);
        assertThat(verifyResult.isMatched()).isTrue();
        assertThat(verifyResult.getSubjectName()).isEqualTo("Alexandre Costa");
    }
}
