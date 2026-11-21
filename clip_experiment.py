"""
Crop Classification — CLIP Zero-Shot Experiment
=================================================
EXPERIMENTAL: Uses OpenAI's CLIP (via open_clip) to compare an input field
image against text descriptions for crop type and field condition.

This is a zero-shot experiment — no training is involved. CLIP computes
image-text similarity scores (cosine similarity, NOT calibrated probabilities).
These scores indicate which text description is most visually consistent with
the image, but they are NOT confidence values or confirmed diagnoses.

** First run will download the CLIP model weights (~350 MB). **

Crop labels are read from models/model_config.json to stay consistent
with the existing EfficientNet classifier.

Usage:
    python clip_experiment.py <image_path>
    python clip_experiment.py <image_path> --save-json

Limitations:
    - Similarity scores are NOT calibrated probabilities.
    - Field-condition descriptions (flooded, drought, diseased) have NOT been
      evaluated against labelled ground-truth data. Treat them as exploratory
      signals only — not reliable detections.
    - CLIP was trained on internet image-text pairs and may not generalise
      well to Indian agricultural imagery.
"""

import sys
import json
import argparse
from pathlib import Path

import torch
import open_clip
from PIL import Image

# ─── Project paths ───────────────────────────────────────────────────────────
BASE_DIR = Path(r"D:\Crop_Classification")
CONFIG_PATH = BASE_DIR / "models" / "model_config.json"
RESULTS_DIR = BASE_DIR / "results"

# ─── CLIP model selection ────────────────────────────────────────────────────
# ViT-B/32 is small, fast, and well-suited for a quick experiment.
CLIP_MODEL_NAME = "ViT-B-32"
CLIP_PRETRAINED = "laion2b_s34b_b79k"


def load_clip_model():
    """Load the pretrained CLIP model, tokenizer, and image preprocessor."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading CLIP model: {CLIP_MODEL_NAME} ({CLIP_PRETRAINED})")
    print(f"  Device: {device}")
    print("  (First run will download model weights — this may take a few minutes.)\n")

    model, _, preprocess = open_clip.create_model_and_transforms(
        CLIP_MODEL_NAME, pretrained=CLIP_PRETRAINED, device=device,
    )
    tokenizer = open_clip.get_tokenizer(CLIP_MODEL_NAME)
    model.eval()
    return model, preprocess, tokenizer, device


def load_crop_labels() -> list[str]:
    """Read crop class names from the existing EfficientNet model config."""
    if not CONFIG_PATH.exists():
        print(f"WARNING: {CONFIG_PATH} not found. Using default crop labels.")
        return ["Cotton", "Maize", "Other", "Rice", "Soybean", "Wheat"]

    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)
    return config.get("class_names", ["Cotton", "Maize", "Other", "Rice", "Soybean", "Wheat"])


def build_text_prompts(crop_labels: list[str]) -> dict[str, list[str]]:
    """
    Build two groups of text descriptions for CLIP comparison.

    1. Crop type — one prompt per crop label.
    2. Field condition — five descriptive phrases.
    """
    crop_prompts = [f"a photo of a {crop.lower()} field" for crop in crop_labels]

    condition_prompts = [
        "a photo of a healthy green crop field",
        "a photo of a flooded agricultural field with standing water",
        "a photo of a drought-affected dry crop field",
        "a photo of a crop field with visibly diseased or damaged plants",
        "a photo of a crop field with unclear or mixed condition",
    ]

    condition_labels = [
        "Healthy",
        "Flooded",
        "Drought-affected",
        "Visibly diseased or damaged",
        "Unclear / mixed condition",
    ]

    return {
        "crop": {"labels": crop_labels, "prompts": crop_prompts},
        "condition": {"labels": condition_labels, "prompts": condition_prompts},
    }


@torch.no_grad()
def compute_similarities(
    model, preprocess, tokenizer, device, image_path: str, prompts: list[str],
) -> list[float]:
    """
    Compute cosine-similarity scores between a single image and a list of
    text prompts using CLIP.

    Returns a list of similarity scores (one per prompt), normalised to
    sum to 1.0 via softmax so they are easier to compare — but they remain
    similarity scores, NOT calibrated probabilities.
    """
    # Preprocess image
    image = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)

    # Tokenise text
    text_tokens = tokenizer(prompts).to(device)

    # Encode
    image_features = model.encode_image(image)
    text_features = model.encode_text(text_tokens)

    # Normalise
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    # Cosine similarity → softmax
    logits = (image_features @ text_features.T).squeeze(0) * 100.0  # CLIP default scale
    scores = torch.softmax(logits, dim=-1).cpu().tolist()

    return scores


def print_results(category_name: str, labels: list[str], scores: list[float]):
    """Pretty-print one category of similarity results."""
    ranked = sorted(zip(labels, scores), key=lambda x: x[1], reverse=True)
    print(f"  {category_name}:")
    for label, score in ranked:
        bar = "#" * int(score * 40)
        marker = " << highest" if score == ranked[0][1] else ""
        print(f"    {label:<35} {score*100:5.1f}%  {bar}{marker}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="CLIP zero-shot crop & condition experiment (similarity scores, NOT probabilities)."
    )
    parser.add_argument("image_path", help="Path to a crop field image.")
    parser.add_argument(
        "--save-json", action="store_true",
        help="Save results as JSON to results/last_clip_experiment.json",
    )
    args = parser.parse_args()

    image_path = Path(args.image_path)
    if not image_path.exists():
        print(f"ERROR: Image not found: {image_path}")
        sys.exit(1)

    # Load CLIP
    model, preprocess, tokenizer, device = load_clip_model()

    # Build prompts from existing config
    crop_labels = load_crop_labels()
    prompt_groups = build_text_prompts(crop_labels)

    # Compute similarities for each group
    results = {}
    for group_key, group_data in prompt_groups.items():
        scores = compute_similarities(
            model, preprocess, tokenizer, device,
            str(image_path), group_data["prompts"],
        )
        results[group_key] = {
            "labels": group_data["labels"],
            "similarity_scores": [round(s, 4) for s in scores],
            "highest_match": group_data["labels"][scores.index(max(scores))],
            "highest_score": round(max(scores), 4),
        }

    # Display
    print("=" * 60)
    print("  CLIP ZERO-SHOT EXPERIMENT -- Image-Text Similarity Scores")
    print("=" * 60)
    print(f"  Image:  {image_path}")
    print(f"  Model:  {CLIP_MODEL_NAME} ({CLIP_PRETRAINED})")
    print()
    print("  [!] These are SIMILARITY SCORES, not calibrated probabilities.")
    print("      Field-condition scores have NOT been validated against")
    print("      labelled ground-truth data.\n")

    print_results("Crop Type (similarity to text description)", results["crop"]["labels"], results["crop"]["similarity_scores"])
    print_results("Field Condition (similarity to text description)", results["condition"]["labels"], results["condition"]["similarity_scores"])

    # Compact summary
    print("  Summary:")
    print(f"    Closest crop type:      {results['crop']['highest_match']}"
          f"  (score: {results['crop']['highest_score']*100:.1f}%)")
    print(f"    Closest field condition: {results['condition']['highest_match']}"
          f"  (score: {results['condition']['highest_score']*100:.1f}%)")
    print("=" * 60)

    # Optionally save JSON
    if args.save_json:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        output = {
            "image": str(image_path),
            "clip_model": f"{CLIP_MODEL_NAME} ({CLIP_PRETRAINED})",
            "note": (
                "All values are image-text cosine similarity scores (softmax-normalised). "
                "They are NOT calibrated probabilities or confirmed diagnoses. "
                "Field-condition scores have not been evaluated against labelled examples."
            ),
            "crop_type": {
                label: score
                for label, score in zip(results["crop"]["labels"], results["crop"]["similarity_scores"])
            },
            "crop_type_highest": results["crop"]["highest_match"],
            "field_condition": {
                label: score
                for label, score in zip(results["condition"]["labels"], results["condition"]["similarity_scores"])
            },
            "field_condition_highest": results["condition"]["highest_match"],
        }
        out_path = RESULTS_DIR / "last_clip_experiment.json"
        with open(out_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()
