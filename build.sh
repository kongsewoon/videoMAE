#!/bin/bash
set -e
IMAGE_NAME="my-xclip"
TAG=$(date +%Y%m%d)
echo "Building ${IMAGE_NAME}:${TAG}"
docker build --no-cache -t ${IMAGE_NAME}:${TAG} .
echo "Built: ${IMAGE_NAME}:${TAG}"

