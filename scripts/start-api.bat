@echo off
setlocal enabledelayedexpansion

set "SCRIPT_DIR=%~dp0"
for %%I in ("%SCRIPT_DIR%..") do set "BASE_DIR=%%~fI"

echo ==================================================================
echo   Iniciando Servidor Facial Recognition API ^& Streaming MJPEG    
echo ==================================================================

set "FACIAL_YUNET_MODEL_PATH=%BASE_DIR%\models\face_detection_yunet.onnx"
set "FACIAL_SFACE_MODEL_PATH=%BASE_DIR%\models\face_recognition_sface.onnx"
set "FACIAL_HAARCASCADE_PATH=%BASE_DIR%\models\haarcascade_frontalface_alt2.xml"

set "JAR_PATH=%BASE_DIR%\bin\facial-api-1.0.0-SNAPSHOT.jar"

if not exist "%JAR_PATH%" (
    set "JAR_PATH=%BASE_DIR%\facial-api\target\facial-api-1.0.0-SNAPSHOT.jar"
)

if not exist "%JAR_PATH%" (
    echo [INFO] JAR nao encontrado em bin/ nem em facial-api\target\. Compilando...
    call mvn -f "%BASE_DIR%\pom.xml" clean install -DskipTests
    set "JAR_PATH=%BASE_DIR%\facial-api\target\facial-api-1.0.0-SNAPSHOT.jar"
)

echo Porta: 8080
echo Swagger UI: http://localhost:8080/swagger-ui.html
echo Stream MJPEG: http://localhost:8080/api/v1/stream/mjpeg
echo Iniciando aplicacao...

java -jar "%JAR_PATH%" %*
