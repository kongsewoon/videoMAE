import os
import torch
import torch.nn as nn
import numpy as np
import cv2
from transformers import VideoMAEModel, VideoMAEConfig
from decord import VideoReader, cpu

# ✅ 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_frames = 16
labels = ["none_fire", "fire"]
weight_dir = "./finetuned_videomae_fusion_light"
test_root = "./test_videos"

# ✅ Config 직접 정의
config = VideoMAEConfig(
    attention_probs_dropout_prob=0.0,
    decoder_hidden_size=384,
    decoder_intermediate_size=1536,
    decoder_num_attention_heads=6,
    decoder_num_hidden_layers=4,
    hidden_act="gelu",
    hidden_dropout_prob=0.0,
    hidden_size=768,
    image_size=224,
    initializer_range=0.02,
    intermediate_size=3072,
    layer_norm_eps=1e-12,
    norm_pix_loss=True,
    num_attention_heads=12,
    num_channels=3,
    num_frames=16,
    num_hidden_layers=12,
    patch_size=16,
    qkv_bias=True,
    tubelet_size=2,
    use_mean_pooling=False,
    torch_dtype="float32"
)

# ✅ 수동 전처리 함수
def manual_preprocess(frames):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    processed = []
    for img in frames:
        img = cv2.resize(img, (224, 224))
        img = img.astype(np.float32) / 255.0
        img = (img - mean) / std
        img = img.transpose(2, 0, 1)  # HWC → CHW
        processed.append(img)

    return torch.tensor(np.array(processed), dtype=torch.float32).unsqueeze(0)  # (1, T, C, H, W)

# ✅ 모델 정의 (학습과 동일)
class VideoMAEFusionAttentionClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.full_model = VideoMAEModel(config)
        self.crop_model = VideoMAEModel(config)

        feat_dim = config.hidden_size
        self.attn = nn.MultiheadAttention(embed_dim=feat_dim, num_heads=4, batch_first=True)

        self.classifier = nn.Sequential(
            nn.Linear(feat_dim * 3, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 2)
        )

    def forward(self, full_pixel_values, crop_pixel_values):
        full_feat = self.full_model(pixel_values=full_pixel_values).last_hidden_state.mean(dim=1)
        crop_feat = self.crop_model(pixel_values=crop_pixel_values).last_hidden_state.mean(dim=1)

        attn_out, _ = self.attn(query=crop_feat.unsqueeze(1),
                                key=full_feat.unsqueeze(1),
                                value=full_feat.unsqueeze(1))
        attn_out = attn_out.squeeze(1)

        fused = torch.cat([crop_feat, attn_out, full_feat], dim=1)
        return self.classifier(fused)

# ✅ 모델 로드
model = VideoMAEFusionAttentionClassifier().to(device)
model.full_model.load_state_dict(torch.load(os.path.join(weight_dir, "best_model_attn_full.pt"), map_location=device), strict=False)
model.crop_model.load_state_dict(torch.load(os.path.join(weight_dir, "best_model_attn_crop.pt"), map_location=device), strict=False)
model.attn.load_state_dict(torch.load(os.path.join(weight_dir, "best_model_attn_attn.pt"), map_location=device))
model.classifier.load_state_dict(torch.load(os.path.join(weight_dir, "best_model_attn_classifier.pt"), map_location=device))
model.eval()

# ✅ 비디오 로드 함수 (Processor 제거, manual_preprocess 사용)
def load_video_tensor(video_path):
    vr = VideoReader(video_path, ctx=cpu(0))
    total = len(vr)
    if total >= num_frames:
        idxs = np.linspace(0, total - 1, num=num_frames, dtype=int)
    else:
        idxs = np.pad(np.arange(total), (0, num_frames - total), mode='edge')

    frames = vr.get_batch(idxs).asnumpy()
    return manual_preprocess(frames).squeeze(0)

# ✅ 추론 함수
def infer_video(full_path, crop_path):
    full_tensor = load_video_tensor(full_path).unsqueeze(0).to(device)
    crop_tensor = load_video_tensor(crop_path).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(full_tensor, crop_tensor)
        probs = torch.softmax(logits, dim=-1)[0]
    return probs

# ✅ Inference 실행
print("[INFO] Inference Start")
found_any = False

for fname in os.listdir(test_root):
    if not fname.endswith("_full.mp4"):
        continue

    found_any = True
    full_path = os.path.join(test_root, fname)
    crop_path = full_path.replace("_full.mp4", "_crop.mp4")

    if not os.path.exists(crop_path):
        print(f"[SKIP] {fname} → bbox 파일 없음")
        continue

    try:
        result = infer_video(full_path, crop_path)
        pred = torch.argmax(result).item()
        confidence = result[pred].item()
        print(f"[🔥 {fname}] → Predicted: {labels[pred]} ({confidence:.4f})")
    except Exception as e:
        print(f"[⚠️ Error] {fname}: {e}")

if not found_any:
    print("[WARN] No _full.mp4 files found in test_videos/")
