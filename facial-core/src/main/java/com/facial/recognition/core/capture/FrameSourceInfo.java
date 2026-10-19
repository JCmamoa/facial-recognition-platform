package com.facial.recognition.core.capture;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.io.Serializable;

/**
 * Metadata and specifications for an active frame source.
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class FrameSourceInfo implements Serializable {
    private static final long serialVersionUID = 1L;

    private String name;
    private String type; // WEBCAM, RTSP, FILE, SIMULATED
    private int width;
    private int height;
    private double fps;
    private String uri;
    private boolean connected;
}
