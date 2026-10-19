package com.facial.recognition.desktop.ui;

import com.facial.recognition.core.FacialRecognitionEngine;
import com.facial.recognition.core.biometrics.BiometricDistanceCalculator;
import com.facial.recognition.core.capture.OpenCVUtils;
import com.facial.recognition.core.model.DetectedFace;
import org.bytedeco.opencv.opencv_core.Mat;

import javax.imageio.ImageIO;
import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.util.List;

/**
 * 1:1 Biometric Comparison and Verification Inspector tool.
 * Compares two faces side-by-side with similarity gauge and Euclidean distance calculation.
 */
public class BiometricInspectorDialog extends JDialog {

    private final FacialRecognitionEngine engine;

    private final JLabel imgLabel1 = new JLabel("Foto 1 (Probe)", SwingConstants.CENTER);
    private final JLabel imgLabel2 = new JLabel("Foto 2 (Gallery)", SwingConstants.CENTER);
    private final JProgressBar similarityBar = new JProgressBar(0, 100);
    private final JLabel resultLabel = new JLabel("Aguardando imagens para comparação...", SwingConstants.CENTER);
    private final JLabel metricsLabel = new JLabel("Similaridade: - | Distância: -", SwingConstants.CENTER);

    private Mat mat1;
    private Mat mat2;

    public BiometricInspectorDialog(Frame parent, FacialRecognitionEngine engine) {
        super(parent, "Inspetor de Comparação Biométrica 1:1", true);
        this.engine = engine;

        initUI();
        pack();
        setLocationRelativeTo(parent);
    }

    private void initUI() {
        setLayout(new BorderLayout(10, 10));

        // Images panel
        JPanel imagesPanel = new JPanel(new GridLayout(1, 2, 15, 10));
        imagesPanel.setBorder(BorderFactory.createEmptyBorder(15, 15, 10, 15));

        Dimension imgSize = new Dimension(220, 220);

        JPanel p1 = new JPanel(new BorderLayout(5, 5));
        imgLabel1.setPreferredSize(imgSize);
        imgLabel1.setBorder(BorderFactory.createLineBorder(Color.GRAY));
        JButton load1Btn = new JButton("Carregar Rosto 1");
        load1Btn.addActionListener(e -> mat1 = chooseImage(imgLabel1));
        p1.add(imgLabel1, BorderLayout.CENTER);
        p1.add(load1Btn, BorderLayout.SOUTH);

        JPanel p2 = new JPanel(new BorderLayout(5, 5));
        imgLabel2.setPreferredSize(imgSize);
        imgLabel2.setBorder(BorderFactory.createLineBorder(Color.GRAY));
        JButton load2Btn = new JButton("Carregar Rosto 2");
        load2Btn.addActionListener(e -> mat2 = chooseImage(imgLabel2));
        p2.add(imgLabel2, BorderLayout.CENTER);
        p2.add(load2Btn, BorderLayout.SOUTH);

        imagesPanel.add(p1);
        imagesPanel.add(p2);

        // Results panel
        JPanel resultsPanel = new JPanel(new GridLayout(4, 1, 6, 6));
        resultsPanel.setBorder(BorderFactory.createEmptyBorder(10, 20, 15, 20));

        similarityBar.setStringPainted(true);
        similarityBar.setValue(0);
        similarityBar.setPreferredSize(new Dimension(300, 25));

        JButton compareBtn = new JButton("Comparar Biometria");
        compareBtn.setFont(compareBtn.getFont().deriveFont(Font.BOLD, 14.0f));
        compareBtn.addActionListener(e -> performComparison());

        resultsPanel.add(compareBtn);
        resultsPanel.add(similarityBar);
        resultsPanel.add(metricsLabel);
        resultsPanel.add(resultLabel);

        add(imagesPanel, BorderLayout.CENTER);
        add(resultsPanel, BorderLayout.SOUTH);
    }

    private Mat chooseImage(JLabel targetLabel) {
        JFileChooser chooser = new JFileChooser();
        if (chooser.showOpenDialog(this) == JFileChooser.APPROVE_OPTION) {
            File file = chooser.getSelectedFile();
            try {
                BufferedImage bImg = ImageIO.read(file);
                if (bImg != null) {
                    Image scaled = bImg.getScaledInstance(220, 220, Image.SCALE_SMOOTH);
                    targetLabel.setIcon(new ImageIcon(scaled));
                    targetLabel.setText(null);
                    return OpenCVUtils.bufferedImageToMat(bImg);
                }
            } catch (Exception ex) {
                JOptionPane.showMessageDialog(this, "Erro lendo arquivo: " + ex.getMessage(), "Erro", JOptionPane.ERROR_MESSAGE);
            }
        }
        return null;
    }

    private void performComparison() {
        if (mat1 == null || mat2 == null) {
            JOptionPane.showMessageDialog(this, "Carregue duas imagens para realizar a comparação.", "Aviso", JOptionPane.WARNING_MESSAGE);
            return;
        }

        List<DetectedFace> faces1 = engine.detectAndExtractEmbeddings(mat1);
        List<DetectedFace> faces2 = engine.detectAndExtractEmbeddings(mat2);

        if (faces1.isEmpty() || faces2.isEmpty()) {
            resultLabel.setText("Falha: Rosto não detectado em uma das imagens.");
            resultLabel.setForeground(Color.RED);
            return;
        }

        float[] emb1 = faces1.get(0).getEmbedding();
        float[] emb2 = faces2.get(0).getEmbedding();

        double cosineSim = BiometricDistanceCalculator.cosineSimilarity(emb1, emb2);
        double euclDist = BiometricDistanceCalculator.euclideanDistance(emb1, emb2);

        int percent = (int) Math.max(0, Math.min(100, cosineSim * 100));
        similarityBar.setValue(percent);
        similarityBar.setString(String.format("%.2f%% Similaridade", cosineSim * 100));

        metricsLabel.setText(String.format("Cosseno: %.4f | Distância Euclidiana L2: %.4f", cosineSim, euclDist));

        boolean matched = cosineSim >= engine.getSimilarityThreshold();
        if (matched) {
            resultLabel.setText("VEREDITO: MESMA PESSOA (Autenticação Aprovada)");
            resultLabel.setForeground(new Color(0, 180, 0));
        } else {
            resultLabel.setText("VEREDITO: PESSOAS DIFERENTES (Acesso Negado)");
            resultLabel.setForeground(Color.RED);
        }
    }
}
