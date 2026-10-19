package com.facial.recognition.core.capture;

import com.facial.recognition.core.exception.FrameSourceException;
import org.bytedeco.opencv.opencv_core.Mat;

import java.awt.image.BufferedImage;

/**
 * Universal interface for video input sources (Webcam, RTSP IP stream, file, or mock).
 */
public interface FrameSource extends AutoCloseable {

    /**
     * Starts and initializes the video frame source.
     */
    void start() throws FrameSourceException;

    /**
     * Stops the video frame capture and releases native resources.
     */
    void stop();

    /**
     * Checks if the frame source is currently open and operational.
     */
    boolean isOpen();

    /**
     * Captures and returns the next frame as an OpenCV Mat (BGR format).
     * Returns null if no frame is available or stream ended.
     */
    Mat captureFrame() throws FrameSourceException;

    /**
     * Captures and returns the next frame converted to a standard Java BufferedImage.
     */
    BufferedImage captureBufferedImage() throws FrameSourceException;

    /**
     * Gets current metadata regarding the frame source.
     */
    FrameSourceInfo getInfo();

    @Override
    default void close() {
        stop();
    }
}
