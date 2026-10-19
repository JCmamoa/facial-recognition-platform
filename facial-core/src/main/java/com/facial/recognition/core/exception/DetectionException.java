package com.facial.recognition.core.exception;

/**
 * Exception thrown when an error occurs during face detection or alignment.
 */
public class DetectionException extends FacialRecognitionException {
    public DetectionException(String message) {
        super(message);
    }

    public DetectionException(String message, Throwable cause) {
        super(message, cause);
    }
}
