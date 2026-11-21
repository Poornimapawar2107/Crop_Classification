"""
Crop Classification — Inference Script
=======================================
Load a trained model and predict crop type from a field image.

Usage:
    python predict.py <image_path>
    python predict.py D:\path\to\field_image.jpg

Output:
    - Predicted crop class
    - Confidence score
    - Top-3 predictions
    - Uncertainty status (LOW / MEDIUM / HIGH)
    - Manual review flag
"""

import sys
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from pathlib import Path


BASE_DIR = Path(r"D:\Crop_Classification")
MODEL_DIR = BASE_DIR / "models"
CONFIG_PATH = MODEL_DIR / "model_config.json"


def load_model_and_config():
    """Load trained model and config."""
    if not CONFIG_PATH.exists():
        print(f"ERROR: Model config not found at {CONFIG_PATH}")
        print("Run train_model.py first to train the model.")
        sys.exit(1)

    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)

    model_path = config["model_path"]
    if not Path(model_path).exists():
        print(f"ERROR: Model not found at {model_path}")
        sys.exit(1)

    print(f"Loading model from {model_path}...")
    model = keras.models.load_model(model_path)
    print("Model loaded successfully.")

    return model, config


def predict(model, image_path: str, config: dict) -> dict:
    """
    Predict crop type from an image.

    Returns dict with:
        - prediction: str (crop class)
        - confidence: float (0-1)
        - confidence_pct: str (e.g. "87.3%")
        - top_3_predictions: list of {class, confidence}
        - uncertainty: str (LOW / MEDIUM / HIGH)
        - needs_manual_review: bool
    """
    class_names = config["class_names"]
    img_size = tuple(config["img_size"])
    high_thresh = config["high_confidence_threshold"]
    low_thresh = config["low_confidence_threshold"]

    # Load and preprocess
    img = keras.preprocessing.image.load_img(image_path, target_size=img_size)
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    probs = model.predict(img_array, verbose=0)[0]

    # Top-3
    top3_idx = np.argsort(probs)[::-1][:3]
    top3 = [
        {"class": class_names[i], "confidence": round(float(probs[i]), 4)}
        for i in top3_idx
    ]

    pred_idx = top3_idx[0]
    pred_class = class_names[pred_idx]
    confidence = float(probs[pred_idx])

    # Uncertainty
    if confidence >= high_thresh:
        uncertainty = "LOW"
        needs_review = False
    elif confidence >= low_thresh:
        uncertainty = "MEDIUM"
        needs_review = True
    else:
        uncertainty = "HIGH"
        needs_review = True

    return {
        "image": str(image_path),
        "prediction": pred_class,
        "confidence": round(confidence, 4),
        "confidence_pct": f"{confidence * 100:.1f}%",
        "top_3_predictions": top3,
        "uncertainty": uncertainty,
        "needs_manual_review": needs_review,
    }


def print_result(result: dict):
    """Pretty-print prediction result."""
    print("\n" + "=" * 50)
    print("  CROP CLASSIFICATION RESULT")
    print("=" * 50)
    print(f"  Image:       {result['image']}")
    print(f"  Prediction:  {result['prediction']}")
    print(f"  Confidence:  {result['confidence_pct']}")
    print(f"  Uncertainty: {result['uncertainty']}")

    if result["needs_manual_review"]:
        print(f"  ** FLAGGED FOR MANUAL REVIEW **")

    print(f"\n  Top-3 Predictions:")
    for i, p in enumerate(result["top_3_predictions"], 1):
        bar = "#" * int(p["confidence"] * 30)
        print(f"    {i}. {p['class']:<10} {p['confidence']*100:5.1f}%  {bar}")
    print("=" * 50 + "\n")


def main():
    if len(sys.argv) < 2:
        print("Usage: python predict.py <image_path>")
        print("Example: python predict.py D:\\photos\\field.jpg")
        sys.exit(1)

    image_path = sys.argv[1]
    if not Path(image_path).exists():
        print(f"ERROR: Image not found: {image_path}")
        sys.exit(1)

    model, config = load_model_and_config()
    result = predict(model, image_path, config)
    print_result(result)

    # Also save as JSON
    output_path = BASE_DIR / "results" / "last_prediction.json"
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Result saved to {output_path}")


if __name__ == "__main__":
    main()
