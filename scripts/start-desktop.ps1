# PowerShell script para iniciar a Interface Desktop no Windows
$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$BaseDir = Split-Path -Parent $ScriptDir

Write-Host "==================================================================" -ForegroundColor Cyan
Write-Host "  Iniciando Interface Grafica Desktop (Swing / FlatLaf)          " -ForegroundColor Cyan
Write-Host "==================================================================" -ForegroundColor Cyan

$env:FACIAL_YUNET_MODEL_PATH = "$BaseDir\models\face_detection_yunet.onnx"
$env:FACIAL_SFACE_MODEL_PATH = "$BaseDir\models\face_recognition_sface.onnx"
$env:FACIAL_HAARCASCADE_PATH = "$BaseDir\models\haarcascade_frontalface_alt2.xml"

Write-Host "[INFO] Executando Desktop via Maven..." -ForegroundColor Green
mvn -f "$BaseDir\pom.xml" exec:java -pl facial-desktop -Dexec.mainClass="com.facial.recognition.desktop.FacialRecognitionDesktopApp" $args
