from flask import Flask, request, jsonify
import os
import tempfile
import torch
import numpy as np
from videomae_infer import infer_video, load_all_models
from person_classifier_infer import load_person_model, infer_person_from_video, is_person

app = Flask(__name__)

# ✅ 서버 시작 시 모델 로딩
load_all_models()
load_person_model()

@app.route("/infer", methods=["POST"])
def infer():
    try:
        full_file = request.files.get('full')
        crop_file = request.files.get('crop')
        event_type = request.form.get('event_type')

        if not full_file or not crop_file:
            return jsonify({"status": "error", "message": "Both full and crop video files must be provided."}), 400

        if not event_type or event_type not in ['FIRE', 'SMOKE', 'DOWN']:
            return jsonify({"status": "error", "message": "event_type must be provided as 'FIRE', 'SMOKE' or 'DOWN'"}), 400

        temp_dir = tempfile.mkdtemp()
        full_path = os.path.join(temp_dir, full_file.filename)
        crop_path = os.path.join(temp_dir, crop_file.filename)

        full_file.save(full_path)
        crop_file.save(crop_path)

        probs = infer_video(full_path, crop_path, event_type)

        os.remove(full_path)
        os.remove(crop_path)
        os.rmdir(temp_dir)

        return jsonify({"status": "success", "probs": probs.tolist()})

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/person", methods=["POST"])
def infer_person():
    try:
        crop_file = request.files.get('crop')

        if not crop_file:
            return jsonify({"status": "error", "message": "Crop video must be provided."}), 400

        temp_dir = tempfile.mkdtemp()
        crop_path = os.path.join(temp_dir, crop_file.filename)
        crop_file.save(crop_path)

        prob_list = infer_person_from_video(crop_path)
        is_valid, avg_prob = is_person(prob_list)

        os.remove(crop_path)
        os.rmdir(temp_dir)

        return jsonify({"is_person": is_valid, "avg_prob": avg_prob, "probs": prob_list.tolist()})

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("API_PORT", 5050)))

