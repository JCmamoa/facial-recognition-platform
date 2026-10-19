package com.facial.recognition.core.capture;

import com.facial.recognition.core.exception.FrameSourceException;
import lombok.extern.slf4j.Slf4j;
import org.bytedeco.javacv.FFmpegFrameGrabber;
import org.bytedeco.javacv.Frame;
import org.bytedeco.javacv.OpenCVFrameConverter;
import org.bytedeco.opencv.opencv_core.Mat;

import java.awt.image.BufferedImage;

/**
 * High-performance JavaCV FFmpegFrameGrabber implementation for RTSP IP cameras,
 * HLS, HTTP streams, and networked video feeds.
 */
@Slf4j
public class RtspStreamFrameSource implements FrameSource {

    private final String rtspUrl;
    private final int connectionTimeoutMs;
    private final boolean forceTcp;

    private FFmpegFrameGrabber grabber;
    private OpenCVFrameConverter.ToMat toMatConverter;
    private volatile boolean open = false;

    private int streamWidth = 1280;
    private int streamHeight = 720;
    private double streamFps = 25.0;

    public RtspStreamFrameSource(String rtspUrl) {
        this(rtspUrl, true, 5000);
    }

    public RtspStreamFrameSource(String rtspUrl, boolean forceTcp, int connectionTimeoutMs) {
        this.rtspUrl = rtspUrl;
        this.forceTcp = forceTcp;
        this.connectionTimeoutMs = connectionTimeoutMs;
    }

    @Override
    public synchronized void start() throws FrameSourceException {
        if (open) {
            log.warn("RTSP stream '{}' is already running.", rtspUrl);
            return;
        }
        try {
            log.info("Connecting to RTSP video stream: {}", rtspUrl);
            grabber = new FFmpegFrameGrabber(rtspUrl);
            if (forceTcp) {
                // Force TCP to prevent UDP packet loss in enterprise networks
                grabber.setOption("rtsp_transport", "tcp");
            }
            grabber.setOption("stimeout", String.valueOf(connectionTimeoutMs * 1000));
            grabber.setOption("max_delay", "500000"); // 500ms max jitter delay
            grabber.setOption("fflags", "nobuffer");  // Low latency streaming
            grabber.setOption("probesize", "32768");
            grabber.setOption("analyzeduration", "500000");

            grabber.start();
            toMatConverter = new OpenCVFrameConverter.ToMat();

            streamWidth = grabber.getImageWidth();
            streamHeight = grabber.getImageHeight();
            streamFps = grabber.getFrameRate() > 0 ? grabber.getFrameRate() : 25.0;

            open = true;
            log.info("RTSP stream connected successfully ({}x{} @ {} FPS)",
                    streamWidth, streamHeight, streamFps);
        } catch (Exception e) {
            open = false;
            stop();
            throw new FrameSourceException("Failed to connect to RTSP stream: " + rtspUrl, e);
        }
    }

    @Override
    public synchronized void stop() {
        if (!open && grabber == null) {
            return;
        }
        log.info("Stopping RTSP stream: {}", rtspUrl);
        open = false;
        try {
            if (grabber != null) {
                grabber.stop();
                grabber.release();
                grabber = null;
            }
            if (toMatConverter != null) {
                toMatConverter.close();
                toMatConverter = null;
            }
        } catch (Exception e) {
            log.error("Error closing RTSP frame grabber", e);
        }
    }

    @Override
    public boolean isOpen() {
        return open && grabber != null;
    }

    @Override
    public synchronized Mat captureFrame() throws FrameSourceException {
        if (!isOpen()) {
            throw new FrameSourceException("RTSP stream is not active: " + rtspUrl);
        }
        try {
            Frame frame = grabber.grabImage();
            if (frame == null) {
                return null;
            }
            Mat mat = toMatConverter.convert(frame);
            if (mat == null || mat.empty()) {
                return null;
            }
            return mat.clone();
        } catch (Exception e) {
            log.error("Error grabbing frame from RTSP stream {}", rtspUrl, e);
            throw new FrameSourceException("Frame grab failed for RTSP " + rtspUrl, e);
        }
    }

    @Override
    public BufferedImage captureBufferedImage() throws FrameSourceException {
        Mat mat = captureFrame();
        if (mat == null) {
            return null;
        }
        return OpenCVUtils.matToBufferedImage(mat);
    }

    @Override
    public FrameSourceInfo getInfo() {
        return FrameSourceInfo.builder()
                .name("RTSP-Camera")
                .type("RTSP")
                .width(streamWidth)
                .height(streamHeight)
                .fps(streamFps)
                .uri(rtspUrl)
                .connected(isOpen())
                .build();
    }
}
