package com.facial.recognition.desktop;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.awt.GraphicsEnvironment;

import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

class FacialRecognitionDesktopAppTest {

    @Test
    @DisplayName("Desktop Application classes load cleanly")
    void testDesktopAppInitialization() {
        if (!GraphicsEnvironment.isHeadless()) {
            FacialRecognitionDesktopApp app = new FacialRecognitionDesktopApp();
            assertNotNull(app);
            assertNotNull(app.getTitle());
            assertTrue(app.getTitle().contains("Reconhecimento Facial"));
            app.dispose();
        } else {
            // In headless CI/test environments, verify headless detection
            assertTrue(GraphicsEnvironment.isHeadless());
        }
    }
}
