package com.facial.recognition.core.exception;

/**
 * Exception thrown when DNN or cascade models cannot be loaded or initialized.
 */
public class ModelInitializationException extends FacialRecognitionException {
    public ModelInitializationException(String message) {
        super(message);
    }

    public ModelInitializationException(String message, Throwable cause) {
        super(message, cause);
    }
}
