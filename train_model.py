"""
Crop Classification — EfficientNet-B0 Baseline Transfer Learning Pipeline
========================================================================
- Model: EfficientNet-B0 (ImageNet pretrained)
- Input: 224x224 RGB
- Classes: Cotton, Maize, Other, Rice, Soybean, Wheat (6 classes)
- Training: Two-phase transfer learning (frozen backbone -> fine-tune top layers)
- Imbalance handling: Class-weighted categorical cross-entropy
- Augmentation: Dynamic on-the-fly augmentation during training ONLY
- Validation/Test: Clean, unaugmented images with standard preprocessing
- Evaluation: Accuracy, Balanced Accuracy, Macro-F1, per-class Precision/Recall/F1, Confusion Matrix
- Model & Config: Saved to models/
- History & Plots: Saved to results/
"""

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_recall_fscore_support,
    accuracy_score,
    balanced_accuracy_score,
)
from sklearn.utils.class_weight import compute_class_weight
from collections import Counter

# ─── Configuration ───────────────────────────────────────────────────────────
BASE_DIR = Path(r"D:\Crop_Classification")
DATA_DIR = BASE_DIR / "data" / "dataset"
MODEL_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Training hyperparameters
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS_FROZEN = 10       # Phase 1: Train classification head
EPOCHS_FINETUNE = 15     # Phase 2: Fine-tune top layers of backbone
LEARNING_RATE_HEAD = 1e-3
LEARNING_RATE_FINETUNE = 1e-4
DROPOUT_RATE = 0.3

# Confidence thresholds for uncertainty flagging
HIGH_CONFIDENCE_THRESHOLD = 0.70
LOW_CONFIDENCE_THRESHOLD = 0.40

# ─── Data Augmentation & Generators ──────────────────────────────────────────
# On-the-fly augmentation ONLY for training.
# Note: EfficientNetB0 has internal Rescaling layer, so preprocess_input leaves pixel values in [0, 255].
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=25,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=False,
    brightness_range=[0.8, 1.2],
    fill_mode='reflect',
)

# Clean, unaugmented generator for validation and test sets
val_test_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
)


def create_generators():
    """Create train, validation, and test data generators."""
    train_gen = train_datagen.flow_from_directory(
        DATA_DIR / "train",
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=True,
        seed=42,
    )

    val_gen = val_test_datagen.flow_from_directory(
        DATA_DIR / "val",
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False,
    )

    test_gen = val_test_datagen.flow_from_directory(
        DATA_DIR / "test",
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False,
    )

    return train_gen, val_gen, test_gen


def get_class_weights(train_gen) -> dict:
    """Compute balanced class weights for training."""
    classes = np.unique(train_gen.classes)
    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=train_gen.classes,
    )
    class_weights = {int(c): float(w) for c, w in zip(classes, weights)}

    print("\nClass weights (computed for imbalanced training set):")
    idx_to_class = {v: k for k, v in train_gen.class_indices.items()}
    counter = Counter(train_gen.classes)
    for idx, weight in class_weights.items():
        print(f"  {idx_to_class[idx]:<10}: weight={weight:.4f}  ({counter[idx]} samples)")

    return class_weights


def build_model(num_classes: int):
    """
    Build transfer learning model with ImageNet-pretrained EfficientNet-B0 backbone.
    """
    base_model = EfficientNetB0(
        weights="imagenet",
        include_top=False,
        input_shape=(*IMG_SIZE, 3),
    )
    base_model.trainable = False  # Frozen for Phase 1

    inputs = keras.Input(shape=(*IMG_SIZE, 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D(name="avg_pool")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(DROPOUT_RATE)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(DROPOUT_RATE)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = keras.Model(inputs, outputs, name="EfficientNetB0_Crop_Baseline")

    print("\nModel Architecture:")
    print(f"  Backbone: EfficientNet-B0 (ImageNet pretrained, {len(base_model.layers)} layers)")
    print(f"  Total params: {model.count_params():,}")
    trainable_p1 = sum(tf.keras.backend.count_params(w) for w in model.trainable_weights)
    print(f"  Trainable params (Phase 1): {trainable_p1:,}")

    return model, base_model


def get_callbacks(phase: str) -> list:
    """Early stopping, learning rate reduction, and model checkpointing."""
    best_path = str(MODEL_DIR / f"best_model_{phase}.keras")
    return [
        callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5 if phase == "finetune" else 4,
            restore_best_weights=True,
            verbose=1,
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1,
        ),
        callbacks.ModelCheckpoint(
            best_path,
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
    ]


def train_pipeline(model, base_model, train_gen, val_gen, class_weights):
    """Two-phase training: Head training followed by top-layer fine-tuning."""

    # ── Phase 1: Classifier head training ────────────────────────────────────
    print("\n" + "=" * 65)
    print("  PHASE 1: Training Classification Head (Backbone Frozen)")
    print("=" * 65)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE_HEAD),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    history1 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS_FROZEN,
        class_weight=class_weights,
        callbacks=get_callbacks("head"),
        verbose=1,
    )

    # ── Phase 2: Fine-tuning top backbone layers ─────────────────────────────
    print("\n" + "=" * 65)
    print("  PHASE 2: Fine-Tuning Top Backbone Layers")
    print("=" * 65)

    # Unfreeze top ~25% of the backbone
    base_model.trainable = True
    fine_tune_from = int(len(base_model.layers) * 0.75)
    for layer in base_model.layers[:fine_tune_from]:
        layer.trainable = False

    trainable_p2 = sum(tf.keras.backend.count_params(w) for w in model.trainable_weights)
    print(f"  Unfrozen from layer {fine_tune_from}/{len(base_model.layers)}")
    print(f"  Trainable params (Phase 2): {trainable_p2:,}")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE_FINETUNE),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    history2 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS_FINETUNE,
        class_weight=class_weights,
        callbacks=get_callbacks("finetune"),
        verbose=1,
    )

    # Merge history metrics
    combined_history = {}
    for k in history1.history:
        combined_history[k] = history1.history[k] + history2.history.get(k, [])

    return combined_history


def evaluate_test_set(model, test_gen, class_names):
    """
    Comprehensive test set evaluation:
    - Overall accuracy
    - Balanced accuracy
    - Macro-F1, Macro-Precision, Macro-Recall
    - Per-class precision, recall, F1, and support
    - Confusion Matrix
    """
    print("\n" + "=" * 65)
    print("  FINAL EVALUATION ON UNSEEN TEST SET")
    print("=" * 65)

    test_gen.reset()
    y_pred_probs = model.predict(test_gen, verbose=1)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = test_gen.classes

    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=range(len(class_names))
    )
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro"
    )
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted"
    )

    print(f"\n  Overall Accuracy:  {acc:.4f} ({acc*100:.2f}%)")
    print(f"  Balanced Accuracy: {bal_acc:.4f} ({bal_acc*100:.2f}%)")
    print(f"  Macro-F1 Score:    {macro_f1:.4f}")
    print(f"  Macro-Precision:   {macro_p:.4f}")
    print(f"  Macro-Recall:      {macro_r:.4f}")

    print(f"\n  Per-Class Performance:")
    print(f"  {'Class':<12} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}")
    print("  " + "-" * 56)
    for i, cls in enumerate(class_names):
        print(f"  {cls:<12} {precision[i]:>10.4f} {recall[i]:>10.4f} {f1[i]:>10.4f} {support[i]:>10}")
    print("  " + "-" * 56)

    # Classification report text
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    report_path = RESULTS_DIR / "classification_report.txt"
    with open(report_path, "w") as f:
        f.write("Crop Classification — Test Set Evaluation Report\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Overall Accuracy:  {acc:.4f} ({acc*100:.2f}%)\n")
        f.write(f"Balanced Accuracy: {bal_acc:.4f} ({bal_acc*100:.2f}%)\n")
        f.write(f"Macro-F1 Score:    {macro_f1:.4f}\n\n")
        f.write(report)
    print(f"\n  Classification report saved to {report_path}")

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6.5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    ax.set_title("EfficientNet-B0 Baseline — Confusion Matrix", fontsize=13, fontweight="bold")
    plt.tight_layout()
    cm_path = RESULTS_DIR / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"  Confusion matrix plot saved to {cm_path}")

    metrics = {
        "model_type": "EfficientNet-B0 Baseline (Transfer Learning)",
        "dataset_notice": "Soybean is predominantly leaf close-ups; Rice/Wheat/Maize have small sample sizes (~78-90 total).",
        "overall_accuracy": float(acc),
        "balanced_accuracy": float(bal_acc),
        "macro_f1": float(macro_f1),
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "weighted_f1": float(weighted_f1),
        "confusion_matrix": cm.tolist(),
        "per_class": {
            cls: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1_score": float(f1[i]),
                "support": int(support[i]),
            }
            for i, cls in enumerate(class_names)
        },
    }
    metrics_path = RESULTS_DIR / "evaluation_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Metrics JSON saved to {metrics_path}")

    return metrics, cm


def plot_curves(history):
    """Plot and save training and validation curves."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Accuracy curve
    axes[0].plot(history["accuracy"], label="Train Accuracy", linewidth=2, color="#1f77b4")
    axes[0].plot(history["val_accuracy"], label="Val Accuracy", linewidth=2, color="#ff7f0e")
    axes[0].set_title("Model Accuracy across Epochs", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.5)

    # Loss curve
    axes[1].plot(history["loss"], label="Train Loss", linewidth=2, color="#1f77b4")
    axes[1].plot(history["val_loss"], label="Val Loss", linewidth=2, color="#ff7f0e")
    axes[1].set_title("Model Loss across Epochs", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    curve_path = RESULTS_DIR / "training_history.png"
    plt.savefig(curve_path, dpi=150)
    plt.close()
    print(f"  Training curves plot saved to {curve_path}")


def main():
    print("\n" + "=" * 65)
    print("  CROP CLASSIFICATION — EfficientNet-B0 Baseline Training")
    print("=" * 65)
    print("  Notice: This is a BASELINE model. Soybean dataset is predominantly leaf close-ups.")

    # 1. Generators
    train_gen, val_gen, test_gen = create_generators()
    class_names = list(train_gen.class_indices.keys())
    print(f"  Classes ({len(class_names)}): {class_names}")
    print(f"  Samples -> Train: {train_gen.samples}, Val: {val_gen.samples}, Test: {test_gen.samples}")

    # 2. Class weights
    class_weights = get_class_weights(train_gen)

    # 3. Build model
    model, base_model = build_model(len(class_names))

    # 4. Train model
    history = train_pipeline(model, base_model, train_gen, val_gen, class_weights)

    # Save final model
    final_model_path = MODEL_DIR / "best_model.keras"
    model.save(final_model_path)
    print(f"\n  Trained model saved to {final_model_path}")

    # Plot curves
    plot_curves(history)

    # 5. Evaluate on test set
    metrics, cm = evaluate_test_set(model, test_gen, class_names)

    # 6. Save model config
    config = {
        "model_name": "EfficientNet-B0 Baseline",
        "model_path": str(final_model_path),
        "class_names": class_names,
        "class_indices": train_gen.class_indices,
        "img_size": list(IMG_SIZE),
        "high_confidence_threshold": HIGH_CONFIDENCE_THRESHOLD,
        "low_confidence_threshold": LOW_CONFIDENCE_THRESHOLD,
        "is_baseline": True,
        "notes": "Trained with balanced class weights. Soybean data is predominantly leaf close-ups.",
    }
    config_path = MODEL_DIR / "model_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"  Model configuration and class mapping saved to {config_path}")

    print("\n" + "=" * 65)
    print("  TRAINING & EVALUATION COMPLETE")
    print(f"  Accuracy:         {metrics['overall_accuracy']*100:.2f}%")
    print(f"  Balanced Accuracy:{metrics['balanced_accuracy']*100:.2f}%")
    print(f"  Macro-F1 Score:   {metrics['macro_f1']:.4f}")
    print("=" * 65)


if __name__ == "__main__":
    main()
