import os, json, tempfile, math, time, logging, sys
import numpy as np
import torch
from flask import Flask, request, jsonify

# ===== Config =====
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32
DEFAULT_NUM_FRAMES = int(os.getenv("NUM_FRAMES", "32"))
WEIGHTS_DIR = os.getenv("MODEL_ID", "./xclip_weights")  # 반드시 로컬 디렉터리

# 이벤트별: 은행 경로 + 임계값을 한 곳에(대문자 키만 사용)
EVENTS = {
    "FIRE": {
        "bank": os.getenv("FIRE", "./bank_params/fire_params.npz"),
        "threshold": -0.0041556287744623464,
    },
    "SMOKE": {
        "bank": os.getenv("SMOKE", "./bank_params/smoke_params.npz"),
        "threshold": 0.2510132992102228,
    },
    "DOWN": {
        "bank": os.getenv("DOWN", "./bank_params/down_params.npz"),
        "threshold": 0.051560279296166356,
    },
}

# 폼에서 threshold 덮어쓰기 허용 여부 (기본: 비허용)
ALLOW_FORM_THRESHOLD = os.getenv("ALLOW_FORM_THRESHOLD", "0") == "1"

MAX_CONTENT_LENGTH = 512 * 1024 * 1024

try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass
torch.backends.cudnn.benchmark = True

# ===== Video IO =====
try:
    from decord import VideoReader, cpu as decord_cpu
    USE_DECORD = True
except Exception:
    import cv2
    USE_DECORD = False
    cv2.setNumThreads(0)

def _read_frames_decord(path: str, num: int):
    vr = VideoReader(path, ctx=decord_cpu(0))
    if len(vr) == 0:
        raise RuntimeError(f"Empty video: {path}")
    idx = np.linspace(0, len(vr) - 1, num).astype(int)
    return [vr[i].asnumpy() for i in idx]

def _read_frames_cv2(path: str, num: int):
    import cv2
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or num
    idxs = np.linspace(0, max(length - 1, 0), num).astype(int).tolist()
    frames = []
    for t in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t))
        ret, frm = cap.read()
        if not ret:
            break
        frm = cv2.cvtColor(frm, cv2.COLOR_BGR2RGB)
        frames.append(frm)
    cap.release()
    if len(frames) == 0:
        raise RuntimeError(f"No frames read: {path}")
    if len(frames) < num:
        frames += [frames[-1]] * (num - len(frames))
    return frames

def load_frames(path: str, num_frames: int):
    return _read_frames_decord(path, num_frames) if USE_DECORD else _read_frames_cv2(path, num_frames)

# ===== Model / Processor (로컬 파일만으로 수동 로드) =====
from transformers import (
    XCLIPModel, XCLIPConfig, XCLIPProcessor,
    VideoMAEImageProcessor, CLIPTokenizer,
)
from safetensors.torch import load_file as safe_load_file

_TOKENIZER=_IMGPROC=_PROCESSOR=_MODEL=None

def _verify_weights_dir():
    must = ["config.json", "preprocessor_config.json", "vocab.json", "merges.txt"]
    miss = [f for f in must if not os.path.isfile(os.path.join(WEIGHTS_DIR, f))]
    if miss:
        raise FileNotFoundError(f"[{WEIGHTS_DIR}] missing: {', '.join(miss)}")
    has_bin  = os.path.isfile(os.path.join(WEIGHTS_DIR, "pytorch_model.bin"))
    has_safe = os.path.isfile(os.path.join(WEIGHTS_DIR, "model.safetensors"))
    if not (has_bin or has_safe):
        shard_bin  = [p for p in os.listdir(WEIGHTS_DIR) if p.startswith("pytorch_model-") and p.endswith(".bin")]
        shard_safe = [p for p in os.listdir(WEIGHTS_DIR) if p.startswith("model-") and p.endswith(".safetensors")]
        if not shard_bin and not shard_safe:
            raise FileNotFoundError("model weights not found (pytorch_model.bin/model.safetensors or shards)")

def _load_config() -> XCLIPConfig:
    with open(os.path.join(WEIGHTS_DIR, "config.json"), "r", encoding="utf-8") as f:
        cfg = json.load(f)
    return XCLIPConfig(**cfg)

def _load_model_from_local() -> XCLIPModel:
    cfg = _load_config()
    model = XCLIPModel(cfg)
    path_safe = os.path.join(WEIGHTS_DIR, "model.safetensors")
    path_bin  = os.path.join(WEIGHTS_DIR, "pytorch_model.bin")
    sd = None
    if os.path.isfile(path_safe):
        sd = safe_load_file(path_safe, device="cpu")
    elif os.path.isfile(path_bin):
        sd = torch.load(path_bin, map_location="cpu")
    else:
        shard_safes = sorted([p for p in os.listdir(WEIGHTS_DIR) if p.startswith("model-") and p.endswith(".safetensors")])
        shard_bins  = sorted([p for p in os.listdir(WEIGHTS_DIR) if p.startswith("pytorch_model-") and p.endswith(".bin")])
        if shard_safes:
            sd = {}
            for p in shard_safes:
                part = safe_load_file(os.path.join(WEIGHTS_DIR, p), device="cpu")
                sd.update(part)
        elif shard_bins:
            sd = {}
            for p in shard_bins:
                part = torch.load(os.path.join(WEIGHTS_DIR, p), map_location="cpu")
                sd.update(part)
    if sd is None:
        raise FileNotFoundError("No weight file could be loaded from local directory.")
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print(f"[XCLIP] load_state_dict: missing={len(missing)} unexpected={len(unexpected)}", file=sys.stderr)
    model.to(DEVICE, dtype=DTYPE).eval()
    return model

def _load_image_processor_from_json() -> VideoMAEImageProcessor:
    with open(os.path.join(WEIGHTS_DIR, "preprocessor_config.json"), "r", encoding="utf-8") as f:
        cfg = json.load(f)
    return VideoMAEImageProcessor(**cfg)

def _load_tokenizer_local() -> CLIPTokenizer:
    return CLIPTokenizer(
        vocab_file=os.path.join(WEIGHTS_DIR, "vocab.json"),
        merges_file=os.path.join(WEIGHTS_DIR, "merges.txt"),
        clean_up_tokenization_spaces=False,
    )

def get_tok_proc_model():
    global _TOKENIZER,_IMGPROC,_PROCESSOR,_MODEL
    if _MODEL is not None:
        return _PROCESSOR,_MODEL
    _verify_weights_dir()
    _TOKENIZER = _load_tokenizer_local()
    _IMGPROC  = _load_image_processor_from_json()
    _PROCESSOR = XCLIPProcessor(tokenizer=_TOKENIZER, image_processor=_IMGPROC)
    _MODEL    = _load_model_from_local()
    return _PROCESSOR,_MODEL

# ===== Bank =====
def load_params(path: str):
    z=np.load(path,allow_pickle=False)
    if "u" in z.files:
        u=torch.tensor(z["u"],dtype=torch.float32,device=DEVICE)
        b=float(z["b"][0]); return u,b
    W=torch.tensor(z["W"],dtype=torch.float32,device=DEVICE)
    w=torch.tensor(z["w"],dtype=torch.float32,device=DEVICE)
    b=float(z["b"][0])
    u=(W.T@w); return u,b

def svm_score(f:torch.Tensor,u:torch.Tensor,b:float)->float:
    return float(torch.dot(f,u).item()+b)

# ===== Embedding =====
@torch.inference_mode()
def embed_video(path: str, num_frames: int = DEFAULT_NUM_FRAMES) -> torch.Tensor:
    processor, model = get_tok_proc_model()
    frames = load_frames(path, num_frames)
    frames = [np.ascontiguousarray(f, dtype=np.uint8) for f in frames]
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(DEVICE=="cuda")):
        inputs = processor(videos=[frames], return_tensors="pt")
    for k, v in list(inputs.items()):
        if isinstance(v, torch.Tensor):
            if v.device.type == "cpu":
                try: v = v.pin_memory()
                except Exception: pass
            inputs[k] = v.to(DEVICE, non_blocking=True)
    feats = model.get_video_features(**inputs)  # [1,D]
    feats = feats.float().squeeze(0)            # -> fp32
    return torch.nn.functional.normalize(feats, dim=-1)

# ===== Flask =====
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH
h = logging.StreamHandler(sys.stdout)
h.setFormatter(logging.Formatter('[%(asctime)s][%(levelname)s] %(message)s','%Y-%m-%d %H:%M:%S'))
app.logger.setLevel(logging.INFO); app.logger.handlers = [h]
logging.getLogger('werkzeug').setLevel(logging.INFO); logging.getLogger('werkzeug').handlers = [h]

# 부팅 로그 (실제 적용 임계값 확인)
app.logger.info("[BOOT] EVENTS=" + json.dumps(
    {k: {"bank": v["bank"], "threshold": v["threshold"], "exists": bool(v["bank"] and os.path.exists(v["bank"]))}
     for k, v in EVENTS.items()}, ensure_ascii=False))

@app.get("/health")
def health():
    return jsonify({
        "device": DEVICE,
        "events": {k: {"threshold": v["threshold"], "bank": v["bank"], "exists": bool(v["bank"] and os.path.exists(v["bank"]))}
                   for k, v in EVENTS.items()}
    }), 200

@app.post("/infer")
def infer():
    fs = request.files
    f = fs.get("crop") or fs.get("file") or fs.get("video") or fs.get("full")
    if f is None:
        return jsonify({"error":"no video file"}), 400
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".mp4", dir="/tmp"); os.close(tmp_fd); f.save(tmp_path)
    try:
        form = request.form.to_dict()
        # 무조건 대문자 통일
        req_event = str(form.get("event_type", "")).strip()
        event_type = req_event.upper()

        ev = EVENTS.get(event_type)
        if not ev or not ev.get("bank") or not os.path.exists(ev["bank"]):
            return jsonify({"error": f"bank not found for {event_type}"}), 400

        num_frames = int(form.get("num_frames", DEFAULT_NUM_FRAMES))

        # --- threshold 결정: 기본은 고정값, 폼 덮어쓰기는 옵션 ---
        if ALLOW_FORM_THRESHOLD:
            raw_th = form.get("threshold", None)
            use_default = (raw_th is None) or (str(raw_th).strip() == "") or (str(raw_th).lower() == "auto")
            threshold = float(ev["threshold"]) if use_default else float(raw_th)
        else:
            raw_th = None
            threshold = float(ev["threshold"])

        feat = embed_video(tmp_path, num_frames=num_frames).to(DEVICE)
        u, b0 = load_params(ev["bank"])
        score = svm_score(feat, u, b0)
        pred = 1 if score >= threshold else 0
        label = "positive" if pred else "negative"

        # (참고) 단순 시그모이드, Platt 보정 아님
        pos_p = 1.0/(1.0 + math.exp(-(score - threshold)))
        probs = [float(1.0 - pos_p), float(pos_p)]

        # 디버그: 적용된 임계값을 확실히 찍는다
        app.logger.info(f"/infer dbg req='{req_event}' -> event={event_type} chosen_th={threshold} bank={ev['bank']}")
        app.logger.info(f"/infer event={event_type} pred={pred} label={label} score={score:.4f} thr={threshold}")

        return jsonify({
            "event": event_type,
            "pred": pred,
            "label": label,
            "score": float(score),
            "threshold": threshold,
            "probs": probs
        }), 200
    except Exception as e:
        import traceback; traceback.print_exc()
        app.logger.exception(f"/infer error: {e}")
        return jsonify({"error": f"infer failed: {e}"}), 500
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("API_PORT", "5050")), debug=False)

