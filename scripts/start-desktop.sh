#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(dirname "$SCRIPT_DIR")"

echo "=================================================================="
echo "  Iniciando Interface Gráfica Desktop (Swing / FlatLaf)          "
echo "=================================================================="

OS_NAME="$(uname -s)"

if [ "$OS_NAME" = "Linux" ]; then
    echo "[OS] Detectado Linux"
    export LD_LIBRARY_PATH="$BASE_DIR/lib:$LD_LIBRARY_PATH"
elif [ "$OS_NAME" = "Darwin" ]; then
    echo "[OS] Detectado macOS"
    export DYLD_LIBRARY_PATH="$BASE_DIR/lib:$DYLD_LIBRARY_PATH"
fi

export FACIAL_YUNET_MODEL_PATH="$BASE_DIR/models/face_detection_yunet.onnx"
export FACIAL_SFACE_MODEL_PATH="$BASE_DIR/models/face_recognition_sface.onnx"
export FACIAL_HAARCASCADE_PATH="$BASE_DIR/models/haarcascade_frontalface_alt2.xml"

echo "[INFO] Executando Desktop via Maven Classpath..."
mvn -f "$BASE_DIR/pom.xml" exec:java -pl facial-desktop -Dexec.mainClass="com.facial.recognition.desktop.FacialRecognitionDesktopApp" "$@"
