package com.facial.recognition.core.exception;

/**
 * Exception thrown when a frame source (webcam, RTSP, file) encounters an error.
 */
public class FrameSourceException extends FacialRecognitionException {
    public FrameSourceException(String message) {
        super(message);
    }

    public FrameSourceException(String message, Throwable cause) {
        super(message, cause);
    }
}
