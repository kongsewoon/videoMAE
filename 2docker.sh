#!/bin/bash

CONTAINER_NAME="videomae-api"

if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
  echo "✅ 컨테이너 '${CONTAINER_NAME}'에 진입합니다..."
  docker exec -it ${CONTAINER_NAME} /bin/bash
else
  echo "❌ 컨테이너 '${CONTAINER_NAME}'가 존재하지 않습니다. 먼저 실행해주세요."
  exit 1
fi
