@echo off
setlocal enabledelayedexpansion

set "SCRIPT_DIR=%~dp0"
for %%I in ("%SCRIPT_DIR%..") do set "BASE_DIR=%%~fI"

echo ==================================================================
echo   Iniciando Interface Grafica Desktop (Swing / FlatLaf)          
echo ==================================================================

set "FACIAL_YUNET_MODEL_PATH=%BASE_DIR%\models\face_detection_yunet.onnx"
set "FACIAL_SFACE_MODEL_PATH=%BASE_DIR%\models\face_recognition_sface.onnx"
set "FACIAL_HAARCASCADE_PATH=%BASE_DIR%\models\haarcascade_frontalface_alt2.xml"

echo [INFO] Executando Desktop via Maven...
call mvn -f "%BASE_DIR%\pom.xml" exec:java -pl facial-desktop -Dexec.mainClass="com.facial.recognition.desktop.FacialRecognitionDesktopApp" %*
