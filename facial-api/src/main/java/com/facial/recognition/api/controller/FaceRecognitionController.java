package com.facial.recognition.api.controller;

import com.facial.recognition.api.dto.DetectionResponse;
import com.facial.recognition.api.dto.EnrollmentResponse;
import com.facial.recognition.api.dto.FaceDto;
import com.facial.recognition.api.dto.RecognitionResponse;
import com.facial.recognition.core.FacialRecognitionEngine;
import com.facial.recognition.core.biometrics.VectorDatabase;
import com.facial.recognition.core.capture.OpenCVUtils;
import com.facial.recognition.core.model.BiometricProfile;
import com.facial.recognition.core.model.DetectedFace;
import com.facial.recognition.core.model.RecognitionResult;
import com.facial.recognition.messaging.event.BiometricAccessEvent;
import com.facial.recognition.messaging.publisher.BiometricEventPublisher;
import com.facial.recognition.persistence.service.BiometricPersistenceService;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.bytedeco.opencv.opencv_core.Mat;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.util.ArrayList;
import java.util.List;

/**
 * REST controller for biometric face detection, enrollment, 1:1 verification, and 1:N identification.
 */
@Slf4j
@RestController
@RequestMapping("/api/v1/faces")
@RequiredArgsConstructor
@Tag(name = "Biometric Face Recognition API", description = "Endpoints for DNN face detection, enrollment, and biometric verification")
public class FaceRecognitionController {

    private final FacialRecognitionEngine recognitionEngine;
    private final BiometricPersistenceService persistenceService;
    private final BiometricEventPublisher eventPublisher;

    @Operation(summary = "Detects faces and 5 facial landmarks in an uploaded image")
    @PostMapping(value = "/detect", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
    public ResponseEntity<DetectionResponse> detectFaces(@RequestParam("file") MultipartFile file) throws Exception {
        long start = System.currentTimeMillis();
        byte[] bytes = file.getBytes();
        Mat image = OpenCVUtils.byteArrayToMat(bytes);

        List<DetectedFace> faces = recognitionEngine.detectFaces(image);
        List<FaceDto> dtoList = faces.stream().map(FaceDto::fromDetectedFace).toList();

        return ResponseEntity.ok(DetectionResponse.builder()
                .faceCount(faces.size())
                .faces(dtoList)
                .latencyMs(System.currentTimeMillis() - start)
                .build());
    }

    @Operation(summary = "Enrolls a subject into the biometric database from an uploaded portrait photo")
    @PostMapping(value = "/enroll", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
    public ResponseEntity<EnrollmentResponse> enrollFace(
            @RequestParam("file") MultipartFile file,
            @RequestParam("id") String id,
            @RequestParam("name") String name,
            @RequestParam(value = "documentId", required = false) String documentId,
            @RequestParam(value = "role", required = false) String role,
            @RequestParam(value = "department", required = false) String department) throws Exception {

        byte[] bytes = file.getBytes();
        Mat image = OpenCVUtils.byteArrayToMat(bytes);

        List<DetectedFace> detectedFaces = recognitionEngine.detectAndExtractEmbeddings(image);
        if (detectedFaces.isEmpty()) {
            return ResponseEntity.badRequest().body(EnrollmentResponse.builder()
                    .success(false)
                    .subjectId(id)
                    .name(name)
                    .message("No face detected in the uploaded photo.")
                    .build());
        }

        DetectedFace primaryFace = detectedFaces.get(0);
        float[] embedding = primaryFace.getEmbedding();

        BiometricProfile existingProfile = persistenceService.findById(id).orElseGet(() ->
                BiometricProfile.builder()
                        .id(id)
                        .name(name)
                        .documentId(documentId)
                        .role(role)
                        .department(department)
                        .active(true)
                        .embeddings(new ArrayList<>())
                        .build());

        existingProfile.addEmbedding(embedding);
        persistenceService.saveProfile(existingProfile, recognitionEngine.getVectorDatabase());

        return ResponseEntity.ok(EnrollmentResponse.builder()
                .success(true)
                .subjectId(id)
                .name(name)
                .registeredEmbeddingsCount(existingProfile.getEmbeddings().size())
                .message("Profile enrolled and indexed successfully.")
                .build());
    }

    @Operation(summary = "Performs 1:1 biometric verification between an uploaded face and a claimed subject ID")
    @PostMapping(value = "/verify", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
    public ResponseEntity<RecognitionResponse> verifyFace(
            @RequestParam("file") MultipartFile file,
            @RequestParam("subjectId") String subjectId,
            @RequestParam(value = "threshold", defaultValue = "0.40") double threshold) throws Exception {

        long start = System.currentTimeMillis();
        byte[] bytes = file.getBytes();
        Mat image = OpenCVUtils.byteArrayToMat(bytes);

        List<DetectedFace> faces = recognitionEngine.detectAndExtractEmbeddings(image);
        if (faces.isEmpty()) {
            return ResponseEntity.ok(RecognitionResponse.builder()
                    .subjectId(subjectId)
                    .matched(false)
                    .latencyMs(System.currentTimeMillis() - start)
                    .build());
        }

        DetectedFace face = faces.get(0);
        RecognitionResult result = recognitionEngine.getVectorDatabase().verify(subjectId, face.getEmbedding(), threshold);
        result.setDetectedFace(face);

        // Publish event
        BiometricAccessEvent event = BiometricAccessEvent.builder()
                .subjectId(subjectId)
                .subjectName(result.getSubjectName())
                .accessGranted(result.isMatched())
                .similarityScore(result.getSimilarity())
                .distance(result.getDistance())
                .cameraId("REST-API-VERIFY")
                .eventType(result.isMatched() ? "ACCESS_GRANTED" : "ACCESS_DENIED_MISMATCH")
                .build();
        eventPublisher.publishEvent(event);

        return ResponseEntity.ok(RecognitionResponse.fromRecognitionResult(result));
    }

    @Operation(summary = "Performs 1:N biometric identification against all registered gallery subjects")
    @PostMapping(value = "/identify", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
    public ResponseEntity<RecognitionResponse> identifyFace(
            @RequestParam("file") MultipartFile file,
            @RequestParam(value = "threshold", defaultValue = "0.40") double threshold) throws Exception {

        long start = System.currentTimeMillis();
        byte[] bytes = file.getBytes();
        Mat image = OpenCVUtils.byteArrayToMat(bytes);

        List<DetectedFace> faces = recognitionEngine.detectAndExtractEmbeddings(image);
        if (faces.isEmpty()) {
            return ResponseEntity.ok(RecognitionResponse.builder()
                    .subjectId("NO_FACE")
                    .subjectName("Nenhum rosto detectado")
                    .matched(false)
                    .latencyMs(System.currentTimeMillis() - start)
                    .build());
        }

        DetectedFace face = faces.get(0);
        RecognitionResult result = recognitionEngine.getVectorDatabase().identify(face.getEmbedding(), threshold);
        result.setDetectedFace(face);

        // Publish event
        BiometricAccessEvent event = BiometricAccessEvent.builder()
                .subjectId(result.getSubjectId())
                .subjectName(result.getSubjectName())
                .accessGranted(result.isMatched())
                .similarityScore(result.getSimilarity())
                .distance(result.getDistance())
                .cameraId("REST-API-IDENTIFY")
                .eventType(result.isMatched() ? "ACCESS_GRANTED" : "UNKNOWN_SUBJECT")
                .build();
        eventPublisher.publishEvent(event);

        return ResponseEntity.ok(RecognitionResponse.fromRecognitionResult(result));
    }

    @Operation(summary = "Lists all enrolled biometric profiles in the database")
    @GetMapping("/profiles")
    public ResponseEntity<List<BiometricProfile>> listProfiles() {
        return ResponseEntity.ok(persistenceService.findAll());
    }

    @Operation(summary = "Deletes an enrolled biometric profile by ID")
    @DeleteMapping("/profiles/{id}")
    public ResponseEntity<Void> deleteProfile(@PathVariable("id") String id) {
        boolean deleted = persistenceService.deleteProfile(id, recognitionEngine.getVectorDatabase());
        return deleted ? ResponseEntity.noContent().build() : ResponseEntity.notFound().build();
    }
}
