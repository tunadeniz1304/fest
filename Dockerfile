# Teslim dokumani Bolum 6 & 8'e uyumlu.
# Base: T4 GPU, CUDA 12.1. Imaj < 8GB hedefi (headless cv + no-cache + tek katman temizlik).
FROM nvidia/cuda:12.1.0-base-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
# ultralytics/easyocr otomatik indirmeyi engelle (calisma aninda internet kapali)
ENV YOLO_OFFLINE=True
ENV HF_HUB_OFFLINE=1
ENV PYTHONUNBUFFERED=1

# Sistem paketleri (tek RUN + temizlik -> kucuk katman)
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 \
        python3-pip \
        libgl1 \
        libglib2.0-0 \
        ffmpeg \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Calisma dizinleri
RUN mkdir -p /app/data/input /app/data/output /app/weights /app/src

# PyTorch (CUDA 12.1 wheel; cuDNN bundled gelir, base imaj yeterli)
RUN pip3 install --no-cache-dir \
        torch==2.2.2 torchvision==0.17.2 \
        --index-url https://download.pytorch.org/whl/cu121

# Diger bagimliliklar
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Model agirliklari (offline calisma icin imaja gomulur)
COPY weights/ /app/weights/

# Kaynak kod (selective COPY -> imaj boyutu)
COPY src/ /app/src/
COPY main.py .
COPY README.md .

# Build aninda agirlik var mi kontrolu (yoksa build erken patlar)
RUN test -f /app/weights/best_model.pt || (echo "HATA: /app/weights/best_model.pt yok!" && exit 1)

# Konteyner ayaga kalkinca otomatik calisir
CMD ["python", "main.py"]
