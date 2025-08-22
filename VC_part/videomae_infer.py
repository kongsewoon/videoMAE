import os
import torch
import torch.nn as nn
import numpy as np
import cv2
from transformers import VideoMAEModel, VideoMAEConfig
from decord import VideoReader, cpu  # ✅ 이거 import 필요합니다.


# 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_frames = 16
labels = ["none_fire", "fire"]

weight_dirs = {
    "FIRE": "/app/finetuned_videomae_fusion_light_fire",
    "SMOKE": "/app/finetuned_videomae_fusion_light_smoke",
    "DOWN" : "/app/finetuned_videomae_fusion_light_down"
}

_model_dict = {}

# ✅ Config 코드 내 직접 정의 (from_pretrained 제거)
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

# ✅ processor 대체: 이미지 전처리 직접 구현
def manual_preprocess(frames):
    # Resize + Normalize (mean, std based on ImageNet)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    processed = []
    for img in frames:
        img = cv2.resize(img, (224, 224))
        img = img.astype(np.float32) / 255.0
        img = (img - mean) / std
        img = img.transpose(2, 0, 1)  # HWC -> CHW
        processed.append(img)

    return torch.tensor(np.array(processed), dtype=torch.float32).unsqueeze(0)   # (1, T, C, H, W)

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

# ✅ 서버 시작 시 모델 로드
def load_all_models():
    global _model_dict

    for event_type, weight_dir in weight_dirs.items():
        m = VideoMAEFusionAttentionClassifier().to(device)
        m.full_model.load_state_dict(torch.load(os.path.join(weight_dir, "final_model_attn_full.pt"), map_location=device), strict=False)
        m.crop_model.load_state_dict(torch.load(os.path.join(weight_dir, "final_model_attn_crop.pt"), map_location=device), strict=False)
        m.attn.load_state_dict(torch.load(os.path.join(weight_dir, "final_model_attn_attn.pt"), map_location=device))
        m.classifier.load_state_dict(torch.load(os.path.join(weight_dir, "final_model_attn_classifier.pt"), map_location=device))
        m.eval()
        _model_dict[event_type] = m
        print(f"🔥 [MODEL LOADED] {event_type} 모델 로딩 완료")

# ✅ 비디오 프레임 전처리 (processor 대체)
def _load_video_tensor(video_path):
    vr = VideoReader(video_path, ctx=cpu(0))
    total = len(vr)
    idxs = np.linspace(0, total - 1, num=num_frames, dtype=int) if total >= num_frames \
        else np.pad(np.arange(total), (0, num_frames - total), mode='edge')

    frames = vr.get_batch(idxs).asnumpy()
    return manual_preprocess(frames).squeeze(0)  # (T, C, H, W)

# ✅ 추론 함수
def infer_video(full_path, crop_path, event_type):
    model = _model_dict[event_type]
    full_tensor = _load_video_tensor(full_path).unsqueeze(0).float().to(device)
    crop_tensor = _load_video_tensor(crop_path).unsqueeze(0).float().to(device)

    with torch.no_grad():
        logits = model(full_tensor, crop_tensor)
        probs = torch.softmax(logits, dim=-1)[0]
    return probs
