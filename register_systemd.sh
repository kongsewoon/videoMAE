#!/bin/bash
set -e

SERVICE_NAME="my-xclip"
IMAGE_TAG="my-xclip:20250828"
PORT=5050

# 호스트에 있는 '반드시 필요한' 것만 지정
WEIGHTS_DIR="/home/server/api_server/xclip_weights"   # config.json + (pytorch_model.bin|model.safetensors) + preprocessor_config.json + vocab.json + merges.txt
BANK_CACHE_DIR="/home/server/api_server/bank_params"  # fire_params.npz, smoke_params.npz, down_params.npz 등

# 컨테이너 내부 경로 (고정)
APP_WEIGHTS_DIR="/app/xclip_weights"
APP_BANK_CACHE="/app/bank_params"

echo "Removing old container if exists..."
docker ps -a --format '{{.Names}}' | grep -Eq "^${SERVICE_NAME}$" && sudo docker rm -f ${SERVICE_NAME} || true

mkdir -p "${BANK_CACHE_DIR}"

echo "Running new container with minimal mounts..."
sudo docker run -d --gpus all \
  --name "${SERVICE_NAME}" \
  --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -p "${PORT}:${PORT}" \
  --mount type=bind,src="${WEIGHTS_DIR}",dst="${APP_WEIGHTS_DIR}",ro \
  --mount type=bind,src="${BANK_CACHE_DIR}",dst="${APP_BANK_CACHE}" \
  --mount type=bind,src="/home/server/api_server",dst="/app" \
  --read-only \
  --tmpfs /tmp:rw,size=512m \
  --tmpfs /app/.hf_cache:rw,size=1024m \
  -e TRANSFORMERS_OFFLINE=1 \
  -e HF_HOME=/app/.hf_cache \
  -e API_PORT="${PORT}" \
  -e MODEL_ID="${APP_WEIGHTS_DIR}" \
  -e NUM_FRAMES="32" \
  "${IMAGE_TAG}"


UNIT_PATH="/etc/systemd/system/${SERVICE_NAME}.service"
sudo tee ${UNIT_PATH} >/dev/null <<EOF
[Unit]
Description=X-CLIP Inference API (/infer only; minimal binds)
After=network.target docker.service
Requires=docker.service

[Service]
Restart=always
ExecStart=/usr/bin/docker start -a ${SERVICE_NAME}
ExecStop=/usr/bin/docker stop ${SERVICE_NAME}

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reexec
sudo systemctl daemon-reload
sudo systemctl enable ${SERVICE_NAME}
sudo systemctl restart ${SERVICE_NAME}

echo "Done. Endpoint: http://<server-ip>:${PORT}/infer"

