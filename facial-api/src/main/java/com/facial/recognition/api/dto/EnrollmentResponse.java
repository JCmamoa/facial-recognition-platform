package com.facial.recognition.api.dto;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.io.Serializable;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class EnrollmentResponse implements Serializable {
    private boolean success;
    private String subjectId;
    private String name;
    private int registeredEmbeddingsCount;
    private String message;
}
