package com.facial.recognition.core.capture;

import com.facial.recognition.core.exception.FrameSourceException;
import lombok.extern.slf4j.Slf4j;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_videoio.VideoCapture;
import org.bytedeco.opencv.global.opencv_videoio;

import java.awt.image.BufferedImage;

/**
 * OpenCV VideoCapture implementation for standard local webcams and V4L2 USB devices.
 */
@Slf4j
public class OpenCVWebcamFrameSource implements FrameSource {

    private final int deviceIndex;
    private final int requestedWidth;
    private final int requestedHeight;
    private final double requestedFps;

    private VideoCapture videoCapture;
    private volatile boolean open = false;

    public OpenCVWebcamFrameSource() {
        this(0, 640, 480, 30.0);
    }

    public OpenCVWebcamFrameSource(int deviceIndex, int width, int height, double fps) {
        this.deviceIndex = deviceIndex;
        this.requestedWidth = width;
        this.requestedHeight = height;
        this.requestedFps = fps;
    }

    @Override
    public synchronized void start() throws FrameSourceException {
        if (open) {
            log.warn("Webcam source index {} is already running.", deviceIndex);
            return;
        }
        try {
            log.info("Opening webcam capture device index {}", deviceIndex);
            videoCapture = new VideoCapture(deviceIndex);
            if (!videoCapture.isOpened()) {
                throw new FrameSourceException("Cannot open video capture device at index " + deviceIndex);
            }

            videoCapture.set(opencv_videoio.CAP_PROP_FRAME_WIDTH, requestedWidth);
            videoCapture.set(opencv_videoio.CAP_PROP_FRAME_HEIGHT, requestedHeight);
            videoCapture.set(opencv_videoio.CAP_PROP_FPS, requestedFps);

            open = true;
            log.info("Webcam index {} started successfully ({}x{} @ {} FPS)",
                    deviceIndex,
                    (int) videoCapture.get(opencv_videoio.CAP_PROP_FRAME_WIDTH),
                    (int) videoCapture.get(opencv_videoio.CAP_PROP_FRAME_HEIGHT),
                    videoCapture.get(opencv_videoio.CAP_PROP_FPS));
        } catch (Exception e) {
            open = false;
            if (videoCapture != null) {
                videoCapture.release();
            }
            throw new FrameSourceException("Failed to initialize webcam device " + deviceIndex, e);
        }
    }

    @Override
    public synchronized void stop() {
        if (!open) {
            return;
        }
        log.info("Stopping webcam capture device {}", deviceIndex);
        open = false;
        if (videoCapture != null) {
            try {
                videoCapture.release();
            } catch (Exception e) {
                log.error("Error releasing VideoCapture", e);
            }
        }
    }

    @Override
    public boolean isOpen() {
        return open && videoCapture != null && videoCapture.isOpened();
    }

    @Override
    public synchronized Mat captureFrame() throws FrameSourceException {
        if (!isOpen()) {
            throw new FrameSourceException("Webcam is not open");
        }
        Mat frame = new Mat();
        boolean readSuccess = videoCapture.read(frame);
        if (!readSuccess || frame.empty()) {
            return null;
        }
        return frame;
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
        int w = (videoCapture != null && open) ? (int) videoCapture.get(opencv_videoio.CAP_PROP_FRAME_WIDTH) : requestedWidth;
        int h = (videoCapture != null && open) ? (int) videoCapture.get(opencv_videoio.CAP_PROP_FRAME_HEIGHT) : requestedHeight;
        double fps = (videoCapture != null && open) ? videoCapture.get(opencv_videoio.CAP_PROP_FPS) : requestedFps;

        return FrameSourceInfo.builder()
                .name("Webcam-" + deviceIndex)
                .type("WEBCAM")
                .width(w)
                .height(h)
                .fps(fps)
                .uri("device://" + deviceIndex)
                .connected(isOpen())
                .build();
    }
}
