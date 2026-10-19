package com.facial.recognition.api.dto;

import com.facial.recognition.core.model.RecognitionResult;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.io.Serializable;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class RecognitionResponse implements Serializable {
    private String subjectId;
    private String subjectName;
    private String department;
    private String role;
    private boolean matched;
    private double similarity;
    private double distance;
    private float detectionConfidence;
    private FaceDto face;
    private long latencyMs;

    public static RecognitionResponse fromRecognitionResult(RecognitionResult result) {
        if (result == null) return null;
        return RecognitionResponse.builder()
                .subjectId(result.getSubjectId())
                .subjectName(result.getSubjectName())
                .department(result.getDepartment())
                .role(result.getRole())
                .matched(result.isMatched())
                .similarity(result.getSimilarity())
                .distance(result.getDistance())
                .detectionConfidence(result.getDetectionConfidence())
                .face(FaceDto.fromDetectedFace(result.getDetectedFace()))
                .latencyMs(result.getInferenceTimeMs())
                .build();
    }
}
