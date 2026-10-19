package com.facial.recognition.desktop.ui;

import javax.swing.JPanel;
import java.awt.Color;
import java.awt.Dimension;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.image.BufferedImage;

/**
 * High-performance double-buffered Swing viewport for rendering real-time video frames
 * with preserved aspect ratio and smooth anti-aliased scaling.
 */
public class VideoCanvasPanel extends JPanel {

    private BufferedImage currentFrame;
    private String statusOverlayText = "Câmera Desconectada";

    public VideoCanvasPanel() {
        setBackground(new Color(25, 27, 32));
        setPreferredSize(new Dimension(640, 480));
    }

    public synchronized void updateFrame(BufferedImage frame) {
        this.currentFrame = frame;
        this.statusOverlayText = null;
        repaint();
    }

    public synchronized void setStatusText(String text) {
        this.statusOverlayText = text;
        repaint();
    }

    @Override
    protected synchronized void paintComponent(Graphics g) {
        super.paintComponent(g);
        Graphics2D g2d = (Graphics2D) g;
        g2d.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BILINEAR);
        g2d.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);

        int panelWidth = getWidth();
        int panelHeight = getHeight();

        if (currentFrame != null) {
            int frameWidth = currentFrame.getWidth();
            int frameHeight = currentFrame.getHeight();

            double scale = Math.min((double) panelWidth / frameWidth, (double) panelHeight / frameHeight);
            int drawWidth = (int) (frameWidth * scale);
            int drawHeight = (int) (frameHeight * scale);
            int x = (panelWidth - drawWidth) / 2;
            int y = (panelHeight - drawHeight) / 2;

            g2d.drawImage(currentFrame, x, y, drawWidth, drawHeight, null);
        } else if (statusOverlayText != null) {
            g2d.setColor(new Color(180, 190, 205));
            g2d.setFont(g2d.getFont().deriveFont(16.0f));
            int strWidth = g2d.getFontMetrics().stringWidth(statusOverlayText);
            g2d.drawString(statusOverlayText, (panelWidth - strWidth) / 2, panelHeight / 2);
        }
    }
}
