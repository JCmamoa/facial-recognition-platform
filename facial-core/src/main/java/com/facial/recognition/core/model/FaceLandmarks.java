package com.facial.recognition.core.model;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.io.Serializable;

/**
 * Encapsulates the 5 primary facial landmarks detected by DNN models (e.g., YuNet):
 * - Right Eye
 * - Left Eye
 * - Nose Tip
 * - Right Mouth Corner
 * - Left Mouth Corner
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class FaceLandmarks implements Serializable {
    private static final long serialVersionUID = 1L;

    private Point2D rightEye;
    private Point2D leftEye;
    private Point2D noseTip;
    private Point2D rightMouth;
    private Point2D leftMouth;

    /**
     * Calculates the inter-ocular distance between right and left eyes.
     */
    public double getInterOcularDistance() {
        if (rightEye == null || leftEye == null) return 0.0;
        return rightEye.distanceTo(leftEye);
    }

    /**
     * Calculates the tilt angle in radians between the two eyes relative to the horizontal axis.
     */
    public double getRollAngle() {
        if (rightEye == null || leftEye == null) return 0.0;
        return rightEye.angleTo(leftEye);
    }

    /**
     * Calculates the center midpoint between both eyes.
     */
    public Point2D getEyeCenter() {
        if (rightEye == null || leftEye == null) return new Point2D(0, 0);
        return new Point2D((rightEye.getX() + leftEye.getX()) / 2.0f,
                           (rightEye.getY() + leftEye.getY()) / 2.0f);
    }

    /**
     * Converts to flat array of floats [rx, ry, lx, ly, nx, ny, rmx, rmy, lmx, lmy].
     */
    public float[] toFlatArray() {
        return new float[]{
            rightEye != null ? rightEye.getX() : 0f, rightEye != null ? rightEye.getY() : 0f,
            leftEye != null ? leftEye.getX() : 0f, leftEye != null ? leftEye.getY() : 0f,
            noseTip != null ? noseTip.getX() : 0f, noseTip != null ? noseTip.getY() : 0f,
            rightMouth != null ? rightMouth.getX() : 0f, rightMouth != null ? rightMouth.getY() : 0f,
            leftMouth != null ? leftMouth.getX() : 0f, leftMouth != null ? leftMouth.getY() : 0f
        };
    }
}
