package com.facial.recognition.messaging.publisher;

import com.facial.recognition.messaging.event.BiometricAccessEvent;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.boot.autoconfigure.condition.ConditionalOnClass;
import org.springframework.kafka.core.KafkaTemplate;
import org.springframework.stereotype.Service;

import java.util.concurrent.CompletableFuture;

/**
 * Enterprise Apache Kafka producer that publishes biometric recognition events
 * and security alerts to distributed event streams.
 */
@Slf4j
@Service
@org.springframework.context.annotation.Primary
@ConditionalOnClass(KafkaTemplate.class)
public class KafkaBiometricEventPublisher implements BiometricEventPublisher {

    private final KafkaTemplate<String, BiometricAccessEvent> kafkaTemplate;
    private final String topicName;
    private final InMemoryBiometricEventBus fallbackBus;

    @Autowired
    public KafkaBiometricEventPublisher(
            @Autowired(required = false) KafkaTemplate<String, BiometricAccessEvent> kafkaTemplate,
            @Value("${facial.kafka.topic:biometric-access-events}") String topicName,
            InMemoryBiometricEventBus fallbackBus) {
        this.kafkaTemplate = kafkaTemplate;
        this.topicName = topicName;
        this.fallbackBus = fallbackBus;
    }

    @Override
    public CompletableFuture<Void> publishEvent(BiometricAccessEvent event) {
        if (event == null) {
            return CompletableFuture.completedFuture(null);
        }

        // Always notify local in-memory subscribers (UI, WebSocket, live monitoring)
        if (fallbackBus != null) {
            fallbackBus.publishEvent(event);
        }

        // If Kafka cluster template is available, stream to Kafka topic
        if (kafkaTemplate != null) {
            String key = event.getSubjectId() != null ? event.getSubjectId() : event.getEventId();
            return kafkaTemplate.send(topicName, key, event)
                    .thenAccept(result -> log.debug("Published biometric event to Kafka topic '{}': {}", topicName, event.getEventId()))
                    .exceptionally(ex -> {
                        log.warn("Could not publish event to Kafka cluster (offline/unreachable): {}", ex.getMessage());
                        return null;
                    });
        } else {
            return CompletableFuture.completedFuture(null);
        }
    }
}
