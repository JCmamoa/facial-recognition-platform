package com.facial.recognition.core.model;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.io.Serializable;

/**
 * Represents a 2D coordinate for facial landmarks.
 */
@Data
@NoArgsConstructor
@AllArgsConstructor
public class Point2D implements Serializable {
    private static final long serialVersionUID = 1L;

    private float x;
    private float y;

    public double distanceTo(Point2D other) {
        if (other == null) return 0.0;
        double dx = this.x - other.x;
        double dy = this.y - other.y;
        return Math.sqrt(dx * dx + dy * dy);
    }

    public double angleTo(Point2D other) {
        if (other == null) return 0.0;
        double dx = other.x - this.x;
        double dy = other.y - this.y;
        return Math.atan2(dy, dx);
    }
}
