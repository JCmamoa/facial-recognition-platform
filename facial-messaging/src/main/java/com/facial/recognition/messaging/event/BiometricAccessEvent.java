package com.facial.recognition.messaging.event;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.io.Serializable;
import java.time.Instant;
import java.util.HashMap;
import java.util.Map;
import java.util.UUID;

/**
 * Event published when a facial recognition decision occurs in real-time.
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class BiometricAccessEvent implements Serializable {
    private static final long serialVersionUID = 1L;

    @Builder.Default
    private String eventId = UUID.randomUUID().toString();

    @Builder.Default
    private Instant timestamp = Instant.now();

    private String subjectId;
    private String subjectName;
    private String department;
    private String role;

    private boolean accessGranted;
    private double similarityScore;
    private double distance;

    private String cameraId;
    private String eventType; // ACCESS_GRANTED, UNKNOWN_SUBJECT, LIVENESS_FAILURE

    @Builder.Default
    private Map<String, String> telemetry = new HashMap<>();
}
