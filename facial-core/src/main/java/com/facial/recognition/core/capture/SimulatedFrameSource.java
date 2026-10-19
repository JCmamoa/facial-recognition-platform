package com.facial.recognition.core.capture;

import com.facial.recognition.core.exception.FrameSourceException;
import lombok.extern.slf4j.Slf4j;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Point;
import org.bytedeco.opencv.opencv_core.Scalar;
import org.bytedeco.opencv.global.opencv_core;
import org.bytedeco.opencv.global.opencv_imgproc;

import java.awt.image.BufferedImage;
import java.time.LocalTime;
import java.time.format.DateTimeFormatter;

/**
 * High-fidelity synthetic frame generator for automated testing, CI/CD pipelines,
 * and server environments without physical camera hardware.
 */
@Slf4j
public class SimulatedFrameSource implements FrameSource {

    private final int width;
    private final int height;
    private final double fps;
    private volatile boolean open = false;
    private long frameIndex = 0;

    public SimulatedFrameSource() {
        this(640, 480, 30.0);
    }

    public SimulatedFrameSource(int width, int height, double fps) {
        this.width = width;
        this.height = height;
        this.fps = fps;
    }

    @Override
    public synchronized void start() throws FrameSourceException {
        open = true;
        log.info("SimulatedFrameSource initialized ({}x{} @ {} FPS)", width, height, fps);
    }

    @Override
    public synchronized void stop() {
        open = false;
        log.info("SimulatedFrameSource stopped.");
    }

    @Override
    public boolean isOpen() {
        return open;
    }

    @Override
    public synchronized Mat captureFrame() throws FrameSourceException {
        if (!isOpen()) {
            throw new FrameSourceException("Simulated frame source is not running");
        }
        frameIndex++;

        // Base background (dark gradient / tech grid)
        Mat frame = new Mat(height, width, opencv_core.CV_8UC3, new Scalar(30, 30, 35, 0));

        // Draw animated grid
        int gridSpacing = 40;
        for (int x = 0; x < width; x += gridSpacing) {
            opencv_imgproc.line(frame, new Point(x, 0), new Point(x, height),
                    new Scalar(45, 45, 50, 0), 1, opencv_imgproc.LINE_AA, 0);
        }
        for (int y = 0; y < height; y += gridSpacing) {
            opencv_imgproc.line(frame, new Point(0, y), new Point(width, y),
                    new Scalar(45, 45, 50, 0), 1, opencv_imgproc.LINE_AA, 0);
        }

        // Draw simulated synthetic human face figure
        int centerX = width / 2 + (int) (Math.sin(frameIndex * 0.05) * 40);
        int centerY = height / 2 + (int) (Math.cos(frameIndex * 0.03) * 20);

        // Face oval / contour
        opencv_imgproc.ellipse(frame, new Point(centerX, centerY),
                new org.bytedeco.opencv.opencv_core.Size(90, 120), 0, 0, 360,
                new Scalar(175, 195, 225, 0), -1, opencv_imgproc.LINE_AA, 0);

        // Eyes (Left & Right)
        int eyeOffsetY = 25;
        int eyeOffsetX = 35;
        // Right eye
        opencv_imgproc.circle(frame, new Point(centerX - eyeOffsetX, centerY - eyeOffsetY), 10,
                new Scalar(70, 50, 40, 0), -1, opencv_imgproc.LINE_AA, 0);
        opencv_imgproc.circle(frame, new Point(centerX - eyeOffsetX, centerY - eyeOffsetY), 4,
                new Scalar(255, 255, 255, 0), -1, opencv_imgproc.LINE_AA, 0);

        // Left eye
        opencv_imgproc.circle(frame, new Point(centerX + eyeOffsetX, centerY - eyeOffsetY), 10,
                new Scalar(70, 50, 40, 0), -1, opencv_imgproc.LINE_AA, 0);
        opencv_imgproc.circle(frame, new Point(centerX + eyeOffsetX, centerY - eyeOffsetY), 4,
                new Scalar(255, 255, 255, 0), -1, opencv_imgproc.LINE_AA, 0);

        // Nose tip
        opencv_imgproc.line(frame, new Point(centerX, centerY - 5), new Point(centerX, centerY + 15),
                new Scalar(120, 140, 170, 0), 3, opencv_imgproc.LINE_AA, 0);

        // Mouth
        opencv_imgproc.ellipse(frame, new Point(centerX, centerY + 50),
                new org.bytedeco.opencv.opencv_core.Size(25, 8), 0, 0, 180,
                new Scalar(80, 80, 190, 0), 3, opencv_imgproc.LINE_AA, 0);

        // On-screen telemetry display
        String timeStr = LocalTime.now().format(DateTimeFormatter.ofPattern("HH:mm:ss.SSS"));
        opencv_imgproc.putText(frame, "SIMULATED FEED - " + timeStr, new Point(20, 30),
                opencv_imgproc.FONT_HERSHEY_SIMPLEX, 0.65, new Scalar(0, 255, 255, 0), 2, opencv_imgproc.LINE_AA, false);
        opencv_imgproc.putText(frame, "FRAME #" + frameIndex, new Point(20, 60),
                opencv_imgproc.FONT_HERSHEY_SIMPLEX, 0.55, new Scalar(0, 200, 0, 0), 1, opencv_imgproc.LINE_AA, false);

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
        return FrameSourceInfo.builder()
                .name("Simulated-Source")
                .type("SIMULATED")
                .width(width)
                .height(height)
                .fps(fps)
                .uri("simulated://local")
                .connected(isOpen())
                .build();
    }
}
