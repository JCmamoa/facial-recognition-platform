# PowerShell script para iniciar a API REST no Windows
$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$BaseDir = Split-Path -Parent $ScriptDir

Write-Host "==================================================================" -ForegroundColor Cyan
Write-Host "  Iniciando Servidor Facial Recognition API & Streaming MJPEG    " -ForegroundColor Cyan
Write-Host "==================================================================" -ForegroundColor Cyan

# Define caminhos absolutos dos modelos ONNX / Haar Cascade
$env:FACIAL_YUNET_MODEL_PATH = "$BaseDir\models\face_detection_yunet.onnx"
$env:FACIAL_SFACE_MODEL_PATH = "$BaseDir\models\face_recognition_sface.onnx"
$env:FACIAL_HAARCASCADE_PATH = "$BaseDir\models\haarcascade_frontalface_alt2.xml"

# Busca o arquivo JAR executável
$JarPath = "$BaseDir\bin\facial-api-1.0.0-SNAPSHOT.jar"

if (-not (Test-Path $JarPath)) {
    $JarPath = "$BaseDir\facial-api\target\facial-api-1.0.0-SNAPSHOT.jar"
}

if (-not (Test-Path $JarPath)) {
    Write-Host "[INFO] JAR nao encontrado em bin/ nem em facial-api\target\. Compilando..." -ForegroundColor Yellow
    mvn -f "$BaseDir\pom.xml" clean install -DskipTests
    $JarPath = "$BaseDir\facial-api\target\facial-api-1.0.0-SNAPSHOT.jar"
}

Write-Host "Porta: 8080" -ForegroundColor Green
Write-Host "Swagger UI: http://localhost:8080/swagger-ui.html" -ForegroundColor Green
Write-Host "Stream MJPEG: http://localhost:8080/api/v1/stream/mjpeg" -ForegroundColor Green
Write-Host "Iniciando aplicacao..." -ForegroundColor Cyan

java -jar $JarPath $args
