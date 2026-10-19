package com.facial.recognition.messaging.publisher;

import com.facial.recognition.messaging.event.BiometricAccessEvent;

import java.util.concurrent.CompletableFuture;

/**
 * Universal publisher interface for biometric security events (Kafka, WebSocket, EventBus).
 */
public interface BiometricEventPublisher {

    /**
     * Publishes a biometric access or security alert event.
     */
    CompletableFuture<Void> publishEvent(BiometricAccessEvent event);
}
