package com.facial.recognition.api.dto;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.io.Serializable;
import java.util.List;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class DetectionResponse implements Serializable {
    private int faceCount;
    private List<FaceDto> faces;
    private long latencyMs;
}
