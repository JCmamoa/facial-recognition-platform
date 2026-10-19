package com.facial.recognition.api.dto;

import com.facial.recognition.core.model.DetectedFace;
import com.facial.recognition.core.model.FaceLandmarks;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.io.Serializable;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class FaceDto implements Serializable {
    private int x;
    private int y;
    private int width;
    private int height;
    private float confidence;
    private FaceLandmarks landmarks;
    private boolean livenessPassed;

    public static FaceDto fromDetectedFace(DetectedFace face) {
        if (face == null) return null;
        return FaceDto.builder()
                .x(face.getX())
                .y(face.getY())
                .width(face.getWidth())
                .height(face.getHeight())
                .confidence(face.getConfidence())
                .landmarks(face.getLandmarks())
                .livenessPassed(face.isLivenessPassed())
                .build();
    }
}
