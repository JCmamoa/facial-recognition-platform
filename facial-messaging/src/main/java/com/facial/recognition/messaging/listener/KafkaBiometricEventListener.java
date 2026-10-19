package com.facial.recognition.messaging.listener;

import com.facial.recognition.messaging.event.BiometricAccessEvent;
import lombok.extern.slf4j.Slf4j;
import org.springframework.boot.autoconfigure.condition.ConditionalOnClass;
import org.springframework.kafka.annotation.KafkaListener;
import org.springframework.stereotype.Component;

/**
 * Enterprise Apache Kafka consumer listening for downstream biometric access decisions,
 * security door relay triggers, and audit logging.
 */
@Slf4j
@Component
@ConditionalOnClass(KafkaListener.class)
public class KafkaBiometricEventListener {

    @KafkaListener(
            topics = "${facial.kafka.topic:biometric-access-events}",
            groupId = "${facial.kafka.group-id:biometric-access-group}",
            autoStartup = "${facial.kafka.listener.auto-startup:false}"
    )
    public void onBiometricEvent(BiometricAccessEvent event) {
        log.info("Received Biometric Event from Kafka: eventId={}, subject={}, granted={}, camera={}",
                event.getEventId(), event.getSubjectName(), event.isAccessGranted(), event.getCameraId());

        if (!event.isAccessGranted()) {
            log.warn("SECURITY ALERT: Unauthorized or unknown subject detected at camera '{}' (similarity: {})",
                    event.getCameraId(), event.getSimilarityScore());
        } else {
            log.info("ACCESS GRANTED: Profile verified for {} (role: {}, dept: {})",
                    event.getSubjectName(), event.getRole(), event.getDepartment());
        }
    }
}
