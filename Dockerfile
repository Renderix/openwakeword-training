FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# System deps
RUN apt-get update && apt-get install -y \
    python3.10 python3.10-venv python3.10-dev python3-dev python3-pip \
    git git-lfs curl build-essential portaudio19-dev libsndfile1 \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.10 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.10 /usr/bin/python

WORKDIR /app

# PyTorch with CUDA 12.1
RUN pip install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cu121

# Training dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Clone and install OpenWakeWord
RUN git clone https://github.com/dscripka/openWakeWord openwakeword \
    && pip install --no-cache-dir -e ./openwakeword

# Download embedding models (small, safe to bake into image)
RUN mkdir -p openwakeword/openwakeword/resources/models \
    && curl -L -o openwakeword/openwakeword/resources/models/embedding_model.onnx \
        'https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/embedding_model.onnx' \
    && curl -L -o openwakeword/openwakeword/resources/models/melspectrogram.onnx \
        'https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/melspectrogram.onnx'

# Copy training scripts
COPY train.py .
COPY setup-data.sh .
RUN chmod +x setup-data.sh
