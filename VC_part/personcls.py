import os
import torch
import torch.nn as nn
import numpy as np
import cv2

# 설정
model_path = "/app/person_classifier_resnet50.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ 모델 로딩
_model = None

def load_person_model():
    global _model
    _model = torch.load(model_path, map_location=device)
    _model.eval()
    print("✅ [MODEL LOADED] person_classifier_resnet50")

# ✅ 추론 함수
def infer_person_from_video(crop_path):
    cap = cv2.VideoCapture(crop_path)
    probs = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_tensor = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).float().to(device)
        with torch.no_grad():
            out = _model(input_tensor)
            prob = torch.softmax(out, dim=1)[0][0].item()  # class 0: person
            probs.append(prob)
    cap.release()
    return np.array(probs)

# ✅ 평균 확률 기준 여부 반환
def is_person(prob_list, threshold=0.7):
    if len(prob_list) == 0:
        return False, 0.0
    avg = float(np.mean(prob_list))
    return avg >= threshold, avg

