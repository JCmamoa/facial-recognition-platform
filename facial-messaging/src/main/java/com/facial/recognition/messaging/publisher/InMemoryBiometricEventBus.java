package com.facial.recognition.messaging.publisher;

import com.facial.recognition.messaging.event.BiometricAccessEvent;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;

import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.function.Consumer;

/**
 * Thread-safe in-memory biometric event bus for local pub/sub, UI updates, and SSE/WebSockets.
 */
@Slf4j
@Component
public class InMemoryBiometricEventBus implements BiometricEventPublisher {

    private final List<Consumer<BiometricAccessEvent>> subscribers = new CopyOnWriteArrayList<>();

    public void subscribe(Consumer<BiometricAccessEvent> consumer) {
        if (consumer != null) {
            subscribers.add(consumer);
        }
    }

    public void unsubscribe(Consumer<BiometricAccessEvent> consumer) {
        subscribers.remove(consumer);
    }

    @Override
    public CompletableFuture<Void> publishEvent(BiometricAccessEvent event) {
        if (event == null) {
            return CompletableFuture.completedFuture(null);
        }
        return CompletableFuture.runAsync(() -> {
            for (Consumer<BiometricAccessEvent> consumer : subscribers) {
                try {
                    consumer.accept(event);
                } catch (Exception e) {
                    log.error("Error dispatching biometric event to subscriber", e);
                }
            }
        });
    }

    public int getSubscriberCount() {
        return subscribers.size();
    }
}
