package com.facial.recognition.api.config;

import com.facial.recognition.core.FacialRecognitionEngine;
import com.facial.recognition.core.biometrics.InMemoryVectorDatabase;
import com.facial.recognition.core.biometrics.VectorDatabase;
import com.facial.recognition.core.capture.FrameSource;
import com.facial.recognition.core.capture.OpenCVWebcamFrameSource;
import com.facial.recognition.core.capture.RtspStreamFrameSource;
import com.facial.recognition.core.capture.SimulatedFrameSource;
import com.facial.recognition.core.detection.CompositeFaceDetector;
import com.facial.recognition.core.embedding.SFaceEmbeddingExtractor;
import com.facial.recognition.core.liveness.LivenessDetector;
import com.facial.recognition.core.model.BiometricProfile;
import com.facial.recognition.core.preprocessing.ImagePreprocessor;
import com.facial.recognition.persistence.service.BiometricPersistenceService;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.boot.CommandLineRunner;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

import java.util.ArrayList;
import java.util.List;

/**
 * Spring Boot configuration for the facial recognition and biometric detection platform.
 */
@Slf4j
@Configuration
public class FacialPlatformConfig {

    @Value("${facial.camera.type:simulated}")
    private String cameraType;

    @Value("${facial.camera.device-index:0}")
    private int cameraDeviceIndex;

    @Value("${facial.camera.rtsp-url:}")
    private String rtspUrl;

    @Bean
    public VectorDatabase vectorDatabase() {
        return new InMemoryVectorDatabase();
    }

    @Bean
    public FacialRecognitionEngine facialRecognitionEngine(VectorDatabase vectorDatabase) {
        log.info("Creating FacialRecognitionEngine bean with SFace 128-d extractor and Composite detector...");
        return new FacialRecognitionEngine(
                new CompositeFaceDetector(),
                new SFaceEmbeddingExtractor(),
                new ImagePreprocessor(),
                new LivenessDetector(),
                vectorDatabase
        );
    }

    @Bean
    public FrameSource frameSource() {
        log.info("Configuring FrameSource with camera type: {}", cameraType);
        FrameSource source;
        if ("webcam".equalsIgnoreCase(cameraType)) {
            source = new OpenCVWebcamFrameSource(cameraDeviceIndex, 640, 480, 30.0);
        } else if ("rtsp".equalsIgnoreCase(cameraType) && !rtspUrl.isBlank()) {
            source = new RtspStreamFrameSource(rtspUrl);
        } else {
            source = new SimulatedFrameSource(640, 480, 30.0);
        }
        try {
            source.start();
        } catch (Exception e) {
            log.warn("Failed to start requested frame source '{}'. Falling back to SimulatedFrameSource: {}",
                    cameraType, e.getMessage());
            source = new SimulatedFrameSource(640, 480, 30.0);
            source.start();
        }
        return source;
    }

    @Bean
    public CommandLineRunner initDatabaseProfiles(BiometricPersistenceService persistenceService,
                                                  VectorDatabase vectorDatabase) {
        return args -> {
            int synced = persistenceService.syncToVectorDatabase(vectorDatabase);
            if (synced == 0) {
                log.info("No profiles in database. Initializing default biometric demo profiles...");

                // Demo Profile 1: Carlos Silva
                float[] carlosVector = new float[128];
                carlosVector[0] = 0.85f;
                carlosVector[1] = 0.40f;
                carlosVector[2] = 0.20f;
                BiometricProfile carlos = BiometricProfile.builder()
                        .id("USR-DEMO-01")
                        .name("Carlos Silva")
                        .documentId("111.222.333-44")
                        .role("Security Lead")
                        .department("DevOps & Infra")
                        .active(true)
                        .embeddings(new ArrayList<>(List.of(carlosVector)))
                        .build();
                persistenceService.saveProfile(carlos, vectorDatabase);

                // Demo Profile 2: Mariana Souza
                float[] marianaVector = new float[128];
                marianaVector[10] = 0.90f;
                marianaVector[11] = 0.35f;
                BiometricProfile mariana = BiometricProfile.builder()
                        .id("USR-DEMO-02")
                        .name("Mariana Souza")
                        .documentId("555.666.777-88")
                        .role("AI Architect")
                        .department("Computer Vision Lab")
                        .active(true)
                        .embeddings(new ArrayList<>(List.of(marianaVector)))
                        .build();
                persistenceService.saveProfile(mariana, vectorDatabase);

                log.info("Initialized {} demo profiles in persistent database and VectorDatabase.", 2);
            }
        };
    }
}
