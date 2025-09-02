FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    FORCE_CUDA=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    TRANSFORMERS_OFFLINE=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    MALLOC_ARENA_MAX=2 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    HF_HOME=/app/.hf_cache

# system deps (필요 최소)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev \
    ffmpeg libgl1 libglib2.0-0 \
    ca-certificates && \
    rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip setuptools wheel

# torch cu118
RUN pip3 install --extra-index-url https://download.pytorch.org/whl/cu118 \
    torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1

# opencv (single wheel)
RUN pip3 uninstall -y opencv-python opencv-python-headless opencv-contrib-python opencv-contrib-python-headless || true && \
    pip3 install "opencv-contrib-python-headless==4.8.1.78"

# libs
RUN pip3 install \
    numpy==1.24.4 \
    Pillow==10.3.0 \
    scikit-learn==1.3.2 \
    tqdm==4.66.4 \
    flask==2.3.3 \
    gunicorn==21.2.0 \
    decord==0.6.0 \
    transformers==4.40.2 \
    huggingface-hub==0.23.2 \
    safetensors==0.4.3

WORKDIR /app

# 코드만 이미지에 포함 (소스 디렉터리 바인딩 안 함)
COPY api_server.py /app/api_server.py

ENV API_PORT=5050
EXPOSE 5050

CMD ["bash", "-lc", "gunicorn -w 1 -b 0.0.0.0:${API_PORT} -t 600 api_server:app"]

