package com.facial.recognition.api.controller;

import com.facial.recognition.core.FacialRecognitionEngine;
import com.facial.recognition.core.capture.FrameSource;
import com.facial.recognition.core.capture.OpenCVUtils;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import jakarta.servlet.http.HttpServletResponse;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.bytedeco.opencv.opencv_core.Mat;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.servlet.mvc.method.annotation.StreamingResponseBody;

import java.io.OutputStream;

/**
 * Real-time MJPEG video streaming controller with live on-frame facial bounding boxes,
 * 5-point landmark markers, and biometric recognition overlay.
 */
@Slf4j
@RestController
@RequestMapping("/api/v1/stream")
@RequiredArgsConstructor
@Tag(name = "Live Video Streaming API", description = "Real-time annotated video feed with facial detection and biometric recognition")
public class VideoStreamController {

    private final FrameSource frameSource;
    private final FacialRecognitionEngine recognitionEngine;

    @Operation(summary = "Continuous real-time MJPEG video stream (viewable directly in browser)")
    @GetMapping(value = "/mjpeg", produces = "multipart/x-mixed-replace; boundary=--frame")
    public StreamingResponseBody streamMjpeg(HttpServletResponse response) {
        response.setHeader("Cache-Control", "no-cache, private");
        response.setHeader("Pragma", "no-cache");

        return (OutputStream outputStream) -> {
            try {
                log.info("Client connected to live MJPEG stream.");
                while (!Thread.currentThread().isInterrupted() && frameSource.isOpen()) {
                    Mat frame = frameSource.captureFrame();
                    if (frame == null || frame.empty()) {
                        Thread.sleep(30);
                        continue;
                    }

                    // Process frame: detect faces, 1:N recognize against vector DB, and draw live overlay
                    recognitionEngine.processFrame(frame, true);

                    byte[] jpegBytes = OpenCVUtils.matToByteArray(frame, ".jpg");

                    outputStream.write(("--frame\r\n").getBytes());
                    outputStream.write(("Content-Type: image/jpeg\r\n").getBytes());
                    outputStream.write(("Content-Length: " + jpegBytes.length + "\r\n\r\n").getBytes());
                    outputStream.write(jpegBytes);
                    outputStream.write(("\r\n").getBytes());
                    outputStream.flush();

                    Thread.sleep(33); // ~30 FPS throttle
                }
            } catch (Exception e) {
                log.info("MJPEG streaming client disconnected: {}", e.getMessage());
            }
        };
    }

    @Operation(summary = "Captures a single real-time snapshot with facial recognition annotations")
    @GetMapping(value = "/snapshot", produces = MediaType.IMAGE_JPEG_VALUE)
    public ResponseEntity<byte[]> getSnapshot() throws Exception {
        Mat frame = frameSource.captureFrame();
        if (frame == null || frame.empty()) {
            return ResponseEntity.noContent().build();
        }

        recognitionEngine.processFrame(frame, true);
        byte[] jpegBytes = OpenCVUtils.matToByteArray(frame, ".jpg");

        return ResponseEntity.ok()
                .contentType(MediaType.IMAGE_JPEG)
                .body(jpegBytes);
    }
}
