package com.facial.recognition.desktop;

import com.facial.recognition.core.FacialRecognitionEngine;
import com.facial.recognition.core.capture.FrameSource;
import com.facial.recognition.core.capture.OpenCVUtils;
import com.facial.recognition.core.capture.OpenCVWebcamFrameSource;
import com.facial.recognition.core.capture.RtspStreamFrameSource;
import com.facial.recognition.core.capture.SimulatedFrameSource;
import com.facial.recognition.core.model.BiometricProfile;
import com.facial.recognition.core.model.RecognitionResult;
import com.facial.recognition.desktop.ui.BiometricInspectorDialog;
import com.facial.recognition.desktop.ui.EnrollmentDialog;
import com.facial.recognition.desktop.ui.VideoCanvasPanel;
import com.formdev.flatlaf.FlatDarkLaf;
import lombok.extern.slf4j.Slf4j;
import org.bytedeco.opencv.opencv_core.Mat;

import javax.swing.*;
import javax.swing.table.DefaultTableModel;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.time.LocalTime;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Modern Java Desktop Application (Swing / FlatLaf) for real-time facial recognition surveillance,
 * biometric identity verification, and profile management.
 */
@Slf4j
public class FacialRecognitionDesktopApp extends JFrame {

    private final FacialRecognitionEngine engine;
    private FrameSource activeFrameSource;

    private final VideoCanvasPanel videoPanel = new VideoCanvasPanel();
    private final JComboBox<String> sourceSelector = new JComboBox<>(new String[]{"Fonte Simulada (Mock)", "Webcam Local (0)", "Câmera IP RTSP"});
    private final JButton startStopBtn = new JButton("Iniciar Monitoramento");
    private final JSlider thresholdSlider = new JSlider(20, 80, 40);
    private final JLabel thresholdLabel = new JLabel("Limiar Cosseno: 0.40");
    private final JCheckBox livenessCheckbox = new JCheckBox("Anti-Spoofing Ativo", true);

    private final DefaultTableModel logTableModel = new DefaultTableModel(
            new String[]{"Hora", "Indivíduo", "Similaridade", "Status", "Latência"}, 0);
    private final JTable logTable = new JTable(logTableModel);

    private final ExecutorService videoExecutor = Executors.newSingleThreadExecutor();
    private final AtomicBoolean isRunning = new AtomicBoolean(false);
    private volatile Mat latestFrame;

    public FacialRecognitionDesktopApp() {
        super("Plataforma Avançada de Detecção e Reconhecimento Facial");
        this.engine = new FacialRecognitionEngine();
        initDefaultDemoProfiles();
        initUI();
    }

    private void initDefaultDemoProfiles() {
        float[] v1 = new float[128];
        v1[0] = 0.9f;
        v1[1] = 0.2f;
        engine.getVectorDatabase().enroll(BiometricProfile.builder()
                .id("USR-01")
                .name("Carlos Silva")
                .role("Security Lead")
                .department("Engenharia")
                .embeddings(new ArrayList<>(List.of(v1)))
                .build());

        float[] v2 = new float[128];
        v2[10] = 0.85f;
        v2[11] = 0.35f;
        engine.getVectorDatabase().enroll(BiometricProfile.builder()
                .id("USR-02")
                .name("Mariana Souza")
                .role("AI Architect")
                .department("R&D")
                .embeddings(new ArrayList<>(List.of(v2)))
                .build());
    }

    private void initUI() {
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        setSize(1024, 720);
        setMinimumSize(new Dimension(800, 600));
        setLayout(new BorderLayout(5, 5));

        // Top Toolbar
        JToolBar toolBar = new JToolBar();
        toolBar.setFloatable(false);
        toolBar.add(new JLabel(" Fonte de Vídeo: "));
        toolBar.add(sourceSelector);
        toolBar.addSeparator(new Dimension(15, 10));

        startStopBtn.setFont(startStopBtn.getFont().deriveFont(Font.BOLD));
        startStopBtn.addActionListener(e -> toggleCapture());
        toolBar.add(startStopBtn);

        toolBar.addSeparator(new Dimension(15, 10));
        JButton enrollBtn = new JButton("Cadastrar Rosto");
        enrollBtn.addActionListener(e -> openEnrollmentDialog());
        toolBar.add(enrollBtn);

        JButton inspectorBtn = new JButton("Comparador 1:1");
        inspectorBtn.addActionListener(e -> openInspectorDialog());
        toolBar.add(inspectorBtn);

        add(toolBar, BorderLayout.NORTH);

        // Center: Video Viewport
        add(videoPanel, BorderLayout.CENTER);

        // Right Sidebar: Settings & Profiles
        JPanel sidePanel = new JPanel();
        sidePanel.setLayout(new BoxLayout(sidePanel, BoxLayout.Y_AXIS));
        sidePanel.setPreferredSize(new Dimension(280, 0));
        sidePanel.setBorder(BorderFactory.createTitledBorder("Configurações Biométricas"));

        thresholdSlider.addChangeListener(e -> {
            double val = thresholdSlider.getValue() / 100.0;
            thresholdLabel.setText(String.format("Limiar Cosseno: %.2f", val));
            engine.setSimilarityThreshold(val);
        });

        livenessCheckbox.addActionListener(e -> engine.setLivenessCheckEnabled(livenessCheckbox.isSelected()));

        sidePanel.add(thresholdLabel);
        sidePanel.add(thresholdSlider);
        sidePanel.add(Box.createVerticalStrut(10));
        sidePanel.add(livenessCheckbox);
        sidePanel.add(Box.createVerticalStrut(15));

        JLabel profilesCountLabel = new JLabel("Perfis Cadastrados: " + engine.getVectorDatabase().size());
        sidePanel.add(profilesCountLabel);

        add(sidePanel, BorderLayout.EAST);

        // Bottom: Access Audit Log Table
        JPanel bottomPanel = new JPanel(new BorderLayout());
        bottomPanel.setPreferredSize(new Dimension(0, 160));
        bottomPanel.setBorder(BorderFactory.createTitledBorder("Registro de Acessos em Tempo Real"));
        bottomPanel.add(new JScrollPane(logTable), BorderLayout.CENTER);
        add(bottomPanel, BorderLayout.SOUTH);

        setLocationRelativeTo(null);
    }

    private void toggleCapture() {
        if (isRunning.get()) {
            stopCapture();
        } else {
            startCapture();
        }
    }

    private void startCapture() {
        int index = sourceSelector.getSelectedIndex();
        if (index == 0) {
            activeFrameSource = new SimulatedFrameSource(640, 480, 30.0);
        } else if (index == 1) {
            activeFrameSource = new OpenCVWebcamFrameSource(0, 640, 480, 30.0);
        } else {
            String url = JOptionPane.showInputDialog(this, "Digite a URL RTSP:", "rtsp://localhost:8554/live");
            if (url == null || url.isBlank()) return;
            activeFrameSource = new RtspStreamFrameSource(url);
        }

        try {
            activeFrameSource.start();
            isRunning.set(true);
            startStopBtn.setText("Parar Monitoramento");
            startStopBtn.setForeground(Color.RED);
            videoPanel.setStatusText("Conectando stream...");

            videoExecutor.submit(this::videoLoop);
        } catch (Exception ex) {
            log.error("Failed to start frame source", ex);
            JOptionPane.showMessageDialog(this, "Erro iniciando câmera: " + ex.getMessage(), "Erro", JOptionPane.ERROR_MESSAGE);
        }
    }

    private void stopCapture() {
        isRunning.set(false);
        if (activeFrameSource != null) {
            activeFrameSource.stop();
            activeFrameSource = null;
        }
        startStopBtn.setText("Iniciar Monitoramento");
        startStopBtn.setForeground(null);
        videoPanel.setStatusText("Monitoramento Pausado");
    }

    private void videoLoop() {
        DateTimeFormatter timeFormatter = DateTimeFormatter.ofPattern("HH:mm:ss");
        while (isRunning.get() && activeFrameSource != null && activeFrameSource.isOpen()) {
            try {
                Mat frame = activeFrameSource.captureFrame();
                if (frame == null || frame.empty()) {
                    Thread.sleep(30);
                    continue;
                }
                latestFrame = frame.clone();

                // Process and annotate frame
                List<RecognitionResult> results = engine.processFrame(frame, true);

                BufferedImage renderedImg = OpenCVUtils.matToBufferedImage(frame);
                SwingUtilities.invokeLater(() -> {
                    videoPanel.updateFrame(renderedImg);

                    for (RecognitionResult r : results) {
                        String time = LocalTime.now().format(timeFormatter);
                        String simStr = String.format("%.1f%%", r.getSimilarity() * 100);
                        String status = r.isMatched() ? "AUTORIZADO" : "DESCONHECIDO";
                        logTableModel.insertRow(0, new Object[]{time, r.getSubjectName(), simStr, status, r.getInferenceTimeMs() + "ms"});

                        if (logTableModel.getRowCount() > 100) {
                            logTableModel.removeRow(100);
                        }
                    }
                });

                Thread.sleep(33); // ~30 FPS
            } catch (Exception e) {
                log.warn("Video loop error: {}", e.getMessage());
            }
        }
    }

    private void openEnrollmentDialog() {
        EnrollmentDialog dialog = new EnrollmentDialog(this, engine, () -> latestFrame);
        dialog.setVisible(true);
    }

    private void openInspectorDialog() {
        BiometricInspectorDialog dialog = new BiometricInspectorDialog(this, engine);
        dialog.setVisible(true);
    }

    public static void main(String[] args) {
        try {
            FlatDarkLaf.setup();
        } catch (Exception e) {
            log.warn("FlatLaf could not be loaded, using system look and feel: {}", e.getMessage());
        }

        SwingUtilities.invokeLater(() -> {
            if (GraphicsEnvironment.isHeadless()) {
                System.out.println("Headless environment detected. Desktop UI cannot open display.");
                return;
            }
            new FacialRecognitionDesktopApp().setVisible(true);
        });
    }
}
