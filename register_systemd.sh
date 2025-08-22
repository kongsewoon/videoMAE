#!/bin/bash

SERVICE_NAME="videomae-api"
IMAGE_TAG="videomae-api:20250821"
PORT=5050
SRC_DIR="/home/server/api_server"

WEIGHTS_FIRE_DIR="${SRC_DIR}/finetuned_videomae_fusion_light_fire"
WEIGHTS_SMOKE_DIR="${SRC_DIR}/finetuned_videomae_fusion_light_smoke"
WEIGHTS_DOWN_DIR="${SRC_DIR}/finetuned_videomae_fusion_light_down"

UNIT_PATH="/etc/systemd/system/${SERVICE_NAME}.service"

echo "🔧 Setting up systemd service for ${SERVICE_NAME}..."

# 기존 컨테이너 삭제 후 새로 생성
if docker ps -a --format '{{.Names}}' | grep -Eq "^${SERVICE_NAME}$"; then
  echo "🛑 기존 컨테이너 삭제 중: ${SERVICE_NAME}"
  sudo docker rm -f ${SERVICE_NAME}
fi

echo "📦 새 컨테이너 생성 중: ${SERVICE_NAME}"
sudo docker run -d --gpus all \
  --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -p ${PORT}:${PORT} \
  -v ${SRC_DIR}:/app \
  -v ${WEIGHTS_FIRE_DIR}:/app/finetuned_videomae_fusion_light_fire \
  -v ${WEIGHTS_SMOKE_DIR}:/app/finetuned_videomae_fusion_light_smoke \
  -v ${WEIGHTS_DOWN_DIR}:/app/finetuned_videomae_fusion_light_down \
  -v /home/user/vison/AlarmClips:/app/AlarmClips \
  -e API_PORT=${PORT} \
  --name ${SERVICE_NAME} \
  ${IMAGE_TAG}

# systemd 유닛 파일 생성
echo "📝 systemd 유닛 파일 생성 중: ${UNIT_PATH}"
sudo tee ${UNIT_PATH} > /dev/null <<EOF
[Unit]
Description=VideoMAE Inference API Container
After=network.target docker.service
Requires=docker.service

[Service]
Restart=always
ExecStart=/usr/bin/docker start -a ${SERVICE_NAME}
ExecStop=/usr/bin/docker stop ${SERVICE_NAME}

[Install]
WantedBy=multi-user.target
EOF

# systemctl 등록 및 재시작
echo "🔄 systemd 등록 및 서비스 시작"
sudo systemctl daemon-reexec
sudo systemctl daemon-reload
sudo systemctl enable ${SERVICE_NAME}
sudo systemctl restart ${SERVICE_NAME}

echo "🎉 등록 완료 → 'systemctl status ${SERVICE_NAME}'로 상태 확인 가능"
echo "🌐 API 접근: http://<your-server-ip>:${PORT}/infer"

