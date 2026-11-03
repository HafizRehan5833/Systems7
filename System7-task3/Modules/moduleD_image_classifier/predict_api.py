"""Python Bridge API for Next.js Image Classifier.

Accepts an image path via CLI and prints JSON output to stdout.
Can be executed by Next.js API route:
    python Modules/moduleD_image_classifier/predict_api.py <path_to_image>
"""
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

HERE = Path(__file__).resolve().parent
MODEL_PATH = HERE / "image_classifier.keras"

CLASS_NAMES = [
    ("airplane", "✈️"),
    ("automobile", "🚗"),
    ("bird", "🐦"),
    ("cat", "🐱"),
    ("deer", "🦌"),
    ("dog", "🐶"),
    ("frog", "🐸"),
    ("horse", "🐴"),
    ("ship", "🚢"),
    ("truck", "🚚"),
]
CLASS_LABELS = [name for name, _ in CLASS_NAMES]
CLASS_ICONS = {name: icon for name, icon in CLASS_NAMES}


def run_mock_prediction(image_path: str):
    """Provides an intelligent heuristic mock prediction when model is not yet trained."""
    import hashlib
    # Generate deterministic pseudo-probabilities based on image content hash
    with open(image_path, "rb") as f:
        file_hash = int(hashlib.md5(f.read()).hexdigest(), 16)
    
    scores = [(file_hash >> (i * 3)) % 100 + 10 for i in range(10)]
    total = sum(scores)
    probs = [round(s / total, 4) for s in scores]
    
    # Sort descending
    sorted_pairs = sorted(zip(CLASS_LABELS, probs), key=lambda x: x[1], reverse=True)
    predictions = [
        {"label": name, "icon": CLASS_ICONS[name], "confidence": prob}
        for name, prob in sorted_pairs
    ]
    
    return {
        "success": True,
        "is_mock": True,
        "latency_ms": 8.5,
        "message": "Model checkpoint (image_classifier.keras) not found. Displaying demo heuristic results. Run 'python Modules/moduleD_image_classifier/image_classifier.py' to train real weights.",
        "top_prediction": predictions[0],
        "predictions": predictions,
        "model_info": {
            "name": "MobileNetV2 (Demo Mode)",
            "trained": False,
            "classes": 10,
        }
    }


def predict_real(image_path: str):
    t0 = time.perf_counter()
    import numpy as np
    import tensorflow as tf
    from PIL import Image

    model = tf.keras.models.load_model(MODEL_PATH)
    
    # Load and preprocess image
    img = Image.open(image_path).convert("RGB")
    img_resized = img.resize((32, 32))
    arr = np.array(img_resized, dtype="float32") / 255.0
    arr = np.expand_dims(arr, axis=0)

    probs = model.predict(arr, verbose=0)[0]
    latency = round((time.perf_counter() - t0) * 1000, 1)

    sorted_indices = probs.argsort()[::-1]
    predictions = [
        {
            "label": CLASS_LABELS[i],
            "icon": CLASS_ICONS[CLASS_LABELS[i]],
            "confidence": round(float(probs[i]), 4),
        }
        for i in sorted_indices
    ]

    return {
        "success": True,
        "is_mock": False,
        "latency_ms": latency,
        "top_prediction": predictions[0],
        "predictions": predictions,
        "model_info": {
            "name": "MobileNetV2 Transfer Learning",
            "trained": True,
            "classes": 10,
        }
    }


def main():
    if len(sys.argv) < 2:
        print(json.dumps({"success": False, "error": "No image path provided."}))
        sys.exit(1)

    image_path = sys.argv[1]
    if not os.path.exists(image_path):
        print(json.dumps({"success": False, "error": f"Image file not found: {image_path}"}))
        sys.exit(1)

    try:
        if MODEL_PATH.exists():
            result = predict_real(image_path)
        else:
            result = run_mock_prediction(image_path)
        print(json.dumps(result))
    except Exception as e:
        print(json.dumps({"success": False, "error": str(e)}))
        sys.exit(1)


if __name__ == "__main__":
    main()

