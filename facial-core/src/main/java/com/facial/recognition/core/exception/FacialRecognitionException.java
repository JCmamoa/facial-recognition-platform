package com.facial.recognition.core.exception;

/**
 * Base exception for all facial recognition platform operations.
 */
public class FacialRecognitionException extends RuntimeException {
    public FacialRecognitionException(String message) {
        super(message);
    }

    public FacialRecognitionException(String message, Throwable cause) {
        super(message, cause);
    }
}
