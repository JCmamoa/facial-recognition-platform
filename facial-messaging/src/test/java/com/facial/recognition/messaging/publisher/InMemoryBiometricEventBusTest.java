package com.facial.recognition.messaging.publisher;

import com.facial.recognition.messaging.event.BiometricAccessEvent;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.TimeUnit;

import static org.assertj.core.api.Assertions.assertThat;

class InMemoryBiometricEventBusTest {

    @Test
    @DisplayName("InMemoryBiometricEventBus dispatches events to registered subscribers")
    void testPublishAndSubscribe() throws Exception {
        InMemoryBiometricEventBus bus = new InMemoryBiometricEventBus();
        CopyOnWriteArrayList<BiometricAccessEvent> received = new CopyOnWriteArrayList<>();

        bus.subscribe(received::add);
        assertThat(bus.getSubscriberCount()).isEqualTo(1);

        BiometricAccessEvent event = BiometricAccessEvent.builder()
                .subjectId("USR-777")
                .subjectName("Luciana Ribeiro")
                .accessGranted(true)
                .similarityScore(0.85)
                .cameraId("CAM-ENTRANCE-1")
                .eventType("ACCESS_GRANTED")
                .build();

        bus.publishEvent(event).get(3, TimeUnit.SECONDS);

        assertThat(received).hasSize(1);
        assertThat(received.get(0).getSubjectName()).isEqualTo("Luciana Ribeiro");
        assertThat(received.get(0).getCameraId()).isEqualTo("CAM-ENTRANCE-1");
    }
}
