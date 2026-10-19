package com.facial.recognition.desktop.ui;

import com.facial.recognition.core.FacialRecognitionEngine;
import com.facial.recognition.core.capture.OpenCVUtils;
import com.facial.recognition.core.model.BiometricProfile;
import lombok.Getter;
import org.bytedeco.opencv.opencv_core.Mat;

import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.util.function.Supplier;

/**
 * Subject enrollment modal dialog for capturing face photo and registering biometric profile.
 */
public class EnrollmentDialog extends JDialog {

    private final FacialRecognitionEngine engine;
    private final Supplier<Mat> currentFrameSupplier;

    private final JTextField idField = new JTextField(15);
    private final JTextField nameField = new JTextField(15);
    private final JTextField roleField = new JTextField(15);
    private final JTextField deptField = new JTextField(15);
    private final JLabel snapshotPreview = new JLabel();

    private Mat capturedMat;

    @Getter
    private boolean enrolled = false;

    public EnrollmentDialog(Frame parent, FacialRecognitionEngine engine, Supplier<Mat> currentFrameSupplier) {
        super(parent, "Cadastrar Novo Perfil Biométrico", true);
        this.engine = engine;
        this.currentFrameSupplier = currentFrameSupplier;

        initUI();
        pack();
        setLocationRelativeTo(parent);
    }

    private void initUI() {
        setLayout(new BorderLayout(10, 10));

        JPanel formPanel = new JPanel(new GridLayout(4, 2, 8, 8));
        formPanel.setBorder(BorderFactory.createEmptyBorder(15, 15, 10, 15));

        formPanel.add(new JLabel("ID do Usuário:"));
        formPanel.add(idField);
        formPanel.add(new JLabel("Nome Completo:"));
        formPanel.add(nameField);
        formPanel.add(new JLabel("Cargo:"));
        formPanel.add(roleField);
        formPanel.add(new JLabel("Departamento:"));
        formPanel.add(deptField);

        snapshotPreview.setPreferredSize(new Dimension(160, 160));
        snapshotPreview.setHorizontalAlignment(SwingConstants.CENTER);
        snapshotPreview.setBorder(BorderFactory.createLineBorder(Color.GRAY));
        snapshotPreview.setText("Sem Foto");

        JPanel centerPanel = new JPanel(new BorderLayout(10, 10));
        centerPanel.add(formPanel, BorderLayout.CENTER);
        centerPanel.add(snapshotPreview, BorderLayout.EAST);

        JButton captureBtn = new JButton("Capturar da Câmera");
        captureBtn.addActionListener(e -> captureSnapshot());

        JButton saveBtn = new JButton("Salvar Biometria");
        saveBtn.addActionListener(e -> saveBiometricProfile());

        JButton cancelBtn = new JButton("Cancelar");
        cancelBtn.addActionListener(e -> dispose());

        JPanel buttonPanel = new JPanel(new FlowLayout(FlowLayout.RIGHT));
        buttonPanel.add(captureBtn);
        buttonPanel.add(saveBtn);
        buttonPanel.add(cancelBtn);

        add(centerPanel, BorderLayout.CENTER);
        add(buttonPanel, BorderLayout.SOUTH);
    }

    private void captureSnapshot() {
        if (currentFrameSupplier == null) return;
        Mat frame = currentFrameSupplier.get();
        if (frame != null && !frame.empty()) {
            this.capturedMat = frame.clone();
            BufferedImage bImg = OpenCVUtils.matToBufferedImage(capturedMat);
            if (bImg != null) {
                Image scaled = bImg.getScaledInstance(160, 160, Image.SCALE_SMOOTH);
                snapshotPreview.setIcon(new ImageIcon(scaled));
                snapshotPreview.setText(null);
            }
        }
    }

    private void saveBiometricProfile() {
        String id = idField.getText().trim();
        String name = nameField.getText().trim();
        String role = roleField.getText().trim();
        String dept = deptField.getText().trim();

        if (id.isEmpty() || name.isEmpty()) {
            JOptionPane.showMessageDialog(this, "ID e Nome são obrigatórios.", "Validação", JOptionPane.WARNING_MESSAGE);
            return;
        }

        if (capturedMat == null || capturedMat.empty()) {
            JOptionPane.showMessageDialog(this, "Capture uma foto do rosto primeiro.", "Validação", JOptionPane.WARNING_MESSAGE);
            return;
        }

        try {
            BiometricProfile profile = engine.enrollSubject(id, name, "", role, dept, capturedMat);
            enrolled = true;
            JOptionPane.showMessageDialog(this,
                    "Perfil de " + profile.getName() + " cadastrado com sucesso no banco de dados vetorial!",
                    "Sucesso", JOptionPane.INFORMATION_MESSAGE);
            dispose();
        } catch (Exception ex) {
            JOptionPane.showMessageDialog(this, "Erro no cadastro biométrico: " + ex.getMessage(),
                    "Erro", JOptionPane.ERROR_MESSAGE);
        }
    }
}
