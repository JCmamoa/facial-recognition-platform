#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(dirname "$SCRIPT_DIR")"

echo "=================================================================="
echo "  Iniciando Servidor Facial Recognition API & Streaming MJPEG    "
echo "=================================================================="

# Detecta o Sistema Operacional (Linux ou macOS)
OS_NAME="$(uname -s)"

if [ "$OS_NAME" = "Linux" ]; then
    echo "[OS] Detectado Linux"
    export LD_LIBRARY_PATH="$BASE_DIR/lib:$LD_LIBRARY_PATH"
elif [ "$OS_NAME" = "Darwin" ]; then
    echo "[OS] Detectado macOS"
    export DYLD_LIBRARY_PATH="$BASE_DIR/lib:$DYLD_LIBRARY_PATH"
fi

# Configura caminhos dos modelos ONNX / Haar Cascade
export FACIAL_YUNET_MODEL_PATH="$BASE_DIR/models/face_detection_yunet.onnx"
export FACIAL_SFACE_MODEL_PATH="$BASE_DIR/models/face_recognition_sface.onnx"
export FACIAL_HAARCASCADE_PATH="$BASE_DIR/models/haarcascade_frontalface_alt2.xml"

# Localiza o arquivo JAR executável
JAR_PATH="$BASE_DIR/bin/facial-api-1.0.0-SNAPSHOT.jar"

if [ ! -f "$JAR_PATH" ]; then
    JAR_PATH="$BASE_DIR/facial-api/target/facial-api-1.0.0-SNAPSHOT.jar"
fi

if [ ! -f "$JAR_PATH" ]; then
    echo "[INFO] Binário pré-compilado não encontrado em bin/ nem em facial-api/target/."
    echo "[INFO] Compilando o projeto com Maven..."
    mvn -f "$BASE_DIR/pom.xml" clean install -DskipTests
    JAR_PATH="$BASE_DIR/facial-api/target/facial-api-1.0.0-SNAPSHOT.jar"
fi

echo "Porta: 8080"
echo "Swagger UI: http://localhost:8080/swagger-ui.html"
echo "Stream MJPEG: http://localhost:8080/api/v1/stream/mjpeg"
echo "Iniciando aplicação..."

java -jar "$JAR_PATH" "$@"
