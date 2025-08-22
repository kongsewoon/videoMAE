#!/bin/bash

IMAGE_NAME="videomae-api"
TAG=$(date +%Y%m%d)  # 오늘 날짜로 태그 설정

echo "🚀 Building Docker image: ${IMAGE_NAME}:${TAG}"

docker build --no-cache -t ${IMAGE_NAME}:${TAG} .

if [ $? -eq 0 ]; then
    echo "✅ Build succeeded: ${IMAGE_NAME}:${TAG}"
else
    echo "❌ Build failed"
    exit 1
fi
