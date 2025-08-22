import os
import torch
import torch.nn as nn
import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from decord import VideoReader, cpu
from torch.cuda.amp import GradScaler, autocast

torch.cuda.empty_cache()

# ✅ Configurations
video_dir = "./videos"
num_frames = 25
batch_size = 1
num_epochs = 50
save_dir = "./finetuned_event_model"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(save_dir, exist_ok=True)
label_map = {"none_fire": 0, "fire": 1}

# ✅ Focal Loss
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits, targets):
        ce_loss = nn.functional.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

# ✅ Preprocessing Function
def manual_preprocess(frames):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    processed = [transform(cv2.resize(img, (224, 224))) for img in frames]
    return torch.stack(processed)

# ✅ Paired Video Dataset
class PairedVideoDataset(Dataset):
    def __init__(self, root_dir, mode, num_frames):
        self.samples = []
        self.num_frames = num_frames
        target_dir = os.path.join(root_dir, mode)
        for label_name, label_id in label_map.items():
            label_dir = os.path.join(target_dir, label_name)
            if not os.path.exists(label_dir):
                continue
            files = [f for f in os.listdir(label_dir) if f.endswith("_full.mp4")]
            for full_fname in files:
                base_name = full_fname.replace("_full.mp4", "")
                crop_fname = base_name + "_crop.mp4"
                crop_path = os.path.join(label_dir, crop_fname)
                full_path = os.path.join(label_dir, full_fname)
                if os.path.exists(crop_path):
                    self.samples.append((crop_path, full_path, label_id))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        crop_path, full_path, label = self.samples[idx]
        crop_vr = VideoReader(crop_path, ctx=cpu(0))
        full_vr = VideoReader(full_path, ctx=cpu(0))
        crop_idxs = np.linspace(0, len(crop_vr) - 1, num=self.num_frames).astype(int)
        full_idxs = np.linspace(0, len(full_vr) - 1, num=self.num_frames).astype(int)
        crop_frames = crop_vr.get_batch(crop_idxs).asnumpy()
        full_frames = full_vr.get_batch(full_idxs).asnumpy()
        crop_tensor = manual_preprocess(crop_frames)
        full_tensor = manual_preprocess(full_frames)
        return crop_tensor, full_tensor, torch.tensor(label)

# ✅ InternVideo-L Fusion Model (Optimized)
class InternVideoLFusion(nn.Module):
    def __init__(self, num_classes=2, feature_dim=1024, hidden_dim=512):
        super().__init__()
        self.crop_stem = nn.Conv3d(3, feature_dim, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3))
        self.full_stem = nn.Conv3d(3, feature_dim, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3))

        encoder_layer = nn.TransformerEncoderLayer(d_model=feature_dim, nhead=8, dim_feedforward=hidden_dim, dropout=0.1, batch_first=True)
        self.transformer_crop = nn.TransformerEncoder(encoder_layer, num_layers=4)  # 줄임
        self.transformer_full = nn.TransformerEncoder(encoder_layer, num_layers=4)  # 줄임

        self.classifier = nn.Sequential(
            nn.Linear(feature_dim * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, crop_x, full_x):
        B, T, C, H, W = crop_x.shape
        crop_x = crop_x.permute(0, 2, 1, 3, 4)
        full_x = full_x.permute(0, 2, 1, 3, 4)

        crop_feat = self.crop_stem(crop_x).flatten(2).transpose(1, 2)
        full_feat = self.full_stem(full_x).flatten(2).transpose(1, 2)

        crop_feat = self.transformer_crop(crop_feat)
        full_feat = self.transformer_full(full_feat)

        crop_avg = torch.mean(crop_feat, dim=1)
        crop_max, _ = torch.max(crop_feat, dim=1)
        full_avg = torch.mean(full_feat, dim=1)
        full_max, _ = torch.max(full_feat, dim=1)

        fused = torch.cat([crop_avg, crop_max, full_avg, full_max], dim=1)
        return self.classifier(fused)

# ✅ DataLoader
train_dataset = PairedVideoDataset(video_dir, "train", num_frames)
val_dataset = PairedVideoDataset(video_dir, "val", num_frames)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# ✅ Training Setup
model = InternVideoLFusion().to(device)
criterion = FocalLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-6, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader) * num_epochs)
scaler = GradScaler()

best_val_loss = float("inf")

# ✅ Training Loop with AMP
for epoch in range(num_epochs):
    model.train()
    total_train_loss = 0
    for crop_videos, full_videos, labels in train_loader:
        crop_videos, full_videos, labels = crop_videos.to(device), full_videos.to(device), labels.to(device)

        optimizer.zero_grad()
        with autocast():
            outputs = model(crop_videos, full_videos)
            loss = criterion(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        scheduler.step()
        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_loader)

    # Validation
    model.eval()
    total_val_loss = 0
    correct, total = 0, 0
    with torch.no_grad():
        for crop_videos, full_videos, labels in val_loader:
            crop_videos, full_videos, labels = crop_videos.to(device), full_videos.to(device), labels.to(device)
            with autocast():
                outputs = model(crop_videos, full_videos)
                loss = criterion(outputs, labels)
            total_val_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_val_loss = total_val_loss / len(val_loader)
    val_accuracy = correct / total * 100
    print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f} - Val Acc: {val_accuracy:.2f}%")

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pt"))

torch.save(model.state_dict(), os.path.join(save_dir, "final_model.pt"))
