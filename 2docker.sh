#!/bin/bash
set -e

CONTAINER_NAME="my-xclip"

# 실행 중인지 확인
if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
  echo "✅ 실행 중인 컨테이너 '${CONTAINER_NAME}'에 진입합니다..."
  docker exec -it "${CONTAINER_NAME}" bash || docker exec -it "${CONTAINER_NAME}" sh

# 꺼져있지만 존재하는 경우
elif docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
  echo "⚠ 컨테이너 '${CONTAINER_NAME}'는 존재하지만 실행 중이 아닙니다."
  echo "   실행하려면:  docker start -ai ${CONTAINER_NAME}"
  exit 1

# 아예 없는 경우
else
  echo "❌ 컨테이너 '${CONTAINER_NAME}'가 존재하지 않습니다. 먼저 실행해주세요."
  exit 1
fi

