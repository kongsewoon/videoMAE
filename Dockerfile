FROM nvcr.io/nvidia/pytorch:22.12-py3

WORKDIR /app

# 필수 패키지 설치
RUN apt-get update && apt-get install -y libgl1-mesa-glx
RUN pip install flask decord transformers
RUN pip install opencv-python-headless==4.5.5.64   # ✅ 버전 고정 설치

EXPOSE 5050

CMD ["python", "api_server.py"]
