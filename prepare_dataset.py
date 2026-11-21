"""
Crop Classification - Dataset Preparation (Fixed)
===================================================
Handles all 4 datasets:
  DS1: Agricultural Crops (30 classes, 829 images)
  DS2: Agriculture Crop Images (kag2 originals only + some_more_images)
  DS3: Cotton Disease Dataset (2,310 images -> Cotton)
  DS4: Soybean Diseased Leaf (701 images -> Soybean)

Key safeguards:
  - DS2/crop_images EXCLUDED (100% duplicate of kag2 originals)
  - DS2/kag2 pre-augmented variants (ahf, ahs, arot) EXCLUDED
  - DS3 pre-existing train/val/test split IGNORED (re-split fresh)
  - Deduplication by MD5 content hash
  - Class capping to prevent imbalance
  - Stratified split 70/15/15 with zero data leakage
  - RGBA -> RGB conversion
"""

import os
import shutil
import hashlib
import zipfile
import random
from pathlib import Path
from collections import defaultdict
from PIL import Image

# === Configuration ============================================================
BASE_DIR = Path(r"D:\Crop_Classification")
RAW_DIR = BASE_DIR / "data" / "raw"
EXTRACT_DIR = BASE_DIR / "data" / "extracted"
FINAL_DIR = BASE_DIR / "data" / "dataset"

# Zip files
DS1_ZIP = RAW_DIR / "dataset1.zip"
DS2_ZIP = RAW_DIR / "dataset2.zip"
DS3_ZIP = RAW_DIR / "archive (2).zip"
DS4_ZIP = RAW_DIR / "archive.zip"

SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}
RANDOM_SEED = 42
MAX_PER_CLASS = 500  # Cap large classes to prevent imbalance

TARGET_CLASSES = ["Rice", "Wheat", "Maize", "Cotton", "Soybean", "Other"]

# DS1 class mapping (folder name -> target class)
DS1_CLASS_MAP = {
    "rice": "Rice", "wheat": "Wheat", "maize": "Maize",
    "cotton": "Cotton", "soyabean": "Soybean",
    # Everything else -> Other
    "almond": "Other", "banana": "Other", "cardamom": "Other",
    "cherry": "Other", "chilli": "Other", "clove": "Other",
    "coconut": "Other", "coffee-plant": "Other", "cucumber": "Other",
    "fox_nut(makhana)": "Other", "gram": "Other", "jowar": "Other",
    "jute": "Other", "lemon": "Other", "mustard-oil": "Other",
    "olive-tree": "Other", "papaya": "Other", "pearl_millet(bajra)": "Other",
    "pineapple": "Other", "sugarcane": "Other", "sunflower": "Other",
    "tea": "Other", "tobacco-plant": "Other", "tomato": "Other",
    "vigna-radiati(mung)": "Other", "pineapple": "Other",
}

# DS2 crops that map to our target classes
DS2_CLASS_MAP = {
    "rice": "Rice", "wheat": "Wheat", "maize": "Maize",
    "jute": "Other", "sugarcane": "Other",
}


# === Utilities ================================================================
def file_hash(filepath: Path) -> str:
    """Compute MD5 hash for deduplication."""
    h = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def is_image(filename: str) -> bool:
    """Check if file is an image by extension."""
    return filename.lower().endswith(
        (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".jfif")
    )


def is_ds2_original(filename: str) -> bool:
    """
    Check if a DS2/kag2 file is an original (not pre-augmented).
    Pre-augmented files have suffixes: ahf (h-flip), ahs (h-shift), arot (rotation).
    Originals end with just 'a' before the extension (e.g., rice001a.jpeg).
    """
    stem = Path(filename).stem.lower()
    if "ahf" in stem or "ahs" in stem or "arot" in stem:
        return False
    return True


def validate_image(filepath: Path) -> bool:
    """Verify the file is a valid, openable image."""
    try:
        with Image.open(filepath) as img:
            img.verify()
        return True
    except Exception:
        return False


# === Extraction ===============================================================
def extract_zips():
    """Extract all 4 zip files."""
    zips = [
        (DS1_ZIP, "DS1_Agricultural_Crops"),
        (DS2_ZIP, "DS2_Agriculture_Crop_Images"),
        (DS3_ZIP, "DS3_Cotton_Disease"),
        (DS4_ZIP, "DS4_Soybean_Leaf"),
    ]
    for zip_path, label in zips:
        dest = EXTRACT_DIR / label
        if dest.exists() and any(dest.iterdir()):
            print(f"  [SKIP] {label} already extracted")
            continue
        if not zip_path.exists():
            print(f"  [WARN] {zip_path.name} not found - skipping")
            continue
        size_mb = zip_path.stat().st_size / 1e6
        print(f"  Extracting {label} ({size_mb:.1f} MB)...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(dest)
        print(f"  Done -> {dest}")


# === Collection ===============================================================
def collect_ds1() -> dict:
    """Collect from DS1: Agricultural Crops (30 classes)."""
    collected = defaultdict(list)
    root = EXTRACT_DIR / "DS1_Agricultural_Crops" / "Agricultural-crops"
    if not root.exists():
        # Try without subfolder
        root = EXTRACT_DIR / "DS1_Agricultural_Crops"
        # Find the actual crops directory
        for item in root.rglob("*"):
            if item.is_dir() and item.name == "Agricultural-crops":
                root = item
                break

    if not root.exists():
        print("  [WARN] DS1 root not found")
        return collected

    for class_dir in sorted(root.iterdir()):
        if not class_dir.is_dir():
            continue
        target = DS1_CLASS_MAP.get(class_dir.name.lower(), "Other")
        for img in class_dir.iterdir():
            if img.is_file() and is_image(img.name):
                collected[target].append(img)

    return collected


def collect_ds2() -> dict:
    """
    Collect from DS2: Agriculture Crop Images.
    - EXCLUDE crop_images/ (100% duplicate of kag2 originals)
    - From kag2/: ONLY originals (exclude ahf, ahs, arot pre-augmented)
    - Include some_more_images/
    """
    collected = defaultdict(list)
    ds2_root = EXTRACT_DIR / "DS2_Agriculture_Crop_Images"

    # kag2 - originals only
    kag2_dir = ds2_root / "kag2"
    if kag2_dir.exists():
        for class_dir in sorted(kag2_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            target = DS2_CLASS_MAP.get(class_dir.name.lower())
            if target is None:
                target = "Other"
            for img in class_dir.iterdir():
                if img.is_file() and is_image(img.name) and is_ds2_original(img.name):
                    collected[target].append(img)
    else:
        print("  [WARN] DS2/kag2 not found")

    # some_more_images
    smi_dir = ds2_root / "some_more_images" / "some_more_images"
    if not smi_dir.exists():
        smi_dir = ds2_root / "some_more_images"
    if smi_dir.exists():
        for class_dir in sorted(smi_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            target = DS2_CLASS_MAP.get(class_dir.name.lower())
            if target is None:
                target = "Other"
            for img in class_dir.iterdir():
                if img.is_file() and is_image(img.name):
                    collected[target].append(img)

    # NOTE: crop_images/ intentionally EXCLUDED (duplicate)
    # NOTE: test_crop_image/ EXCLUDED (unlabeled)

    return collected


def collect_ds3() -> dict:
    """
    Collect from DS3: Cotton Disease Dataset.
    All subfolders (diseased/fresh, leaf/plant) -> Cotton.
    IGNORE pre-existing train/val/test split.
    """
    collected = defaultdict(list)
    ds3_root = EXTRACT_DIR / "DS3_Cotton_Disease" / "Cotton Disease"
    if not ds3_root.exists():
        ds3_root = EXTRACT_DIR / "DS3_Cotton_Disease"

    # Walk ALL subdirectories regardless of train/val/test structure
    for img_path in ds3_root.rglob("*"):
        if img_path.is_file() and is_image(img_path.name):
            collected["Cotton"].append(img_path)

    return collected


def collect_ds4() -> dict:
    """
    Collect from DS4: Soybean Diseased Leaf Dataset.
    All disease subfolders -> Soybean.
    """
    collected = defaultdict(list)
    ds4_root = EXTRACT_DIR / "DS4_Soybean_Leaf"

    for img_path in ds4_root.rglob("*"):
        if img_path.is_file() and is_image(img_path.name):
            collected["Soybean"].append(img_path)

    return collected


# === Deduplication & Capping ==================================================
def deduplicate(collected: dict) -> dict:
    """Remove duplicate images by content hash within each class."""
    deduped = {}
    total_dupes = 0
    for cls, paths in collected.items():
        seen = set()
        unique = []
        for p in paths:
            h = file_hash(p)
            if h not in seen:
                seen.add(h)
                unique.append(p)
            else:
                total_dupes += 1
        deduped[cls] = unique
    print(f"  Removed {total_dupes} duplicate images total")
    return deduped


def cap_classes(deduped: dict, max_per_class: int) -> dict:
    """Downsample large classes to max_per_class for balance."""
    random.seed(RANDOM_SEED)
    capped = {}
    for cls, paths in deduped.items():
        if len(paths) > max_per_class:
            capped[cls] = random.sample(paths, max_per_class)
            print(f"  {cls}: {len(paths)} -> {max_per_class} (capped)")
        else:
            capped[cls] = paths
            print(f"  {cls}: {len(paths)} (kept all)")
    return capped


# === Splitting ================================================================
def create_splits(capped: dict):
    """
    Stratified train/val/test split with zero data leakage.
    Each image appears in exactly one split.
    """
    random.seed(RANDOM_SEED)
    split_counts = defaultdict(lambda: defaultdict(int))
    corrupt_count = 0

    for cls in TARGET_CLASSES:
        images = sorted(capped.get(cls, []), key=lambda p: str(p))
        random.shuffle(images)

        n = len(images)
        n_train = int(n * SPLIT_RATIOS["train"])
        n_val = int(n * SPLIT_RATIOS["val"])

        splits = {
            "train": images[:n_train],
            "val": images[n_train:n_train + n_val],
            "test": images[n_train + n_val:],
        }

        for split_name, split_images in splits.items():
            dest_dir = FINAL_DIR / split_name / cls
            dest_dir.mkdir(parents=True, exist_ok=True)
            for i, src in enumerate(split_images):
                ext = src.suffix.lower()
                if ext == ".jfif":
                    ext = ".jpg"
                dst = dest_dir / f"{cls.lower()}_{i+1:04d}{ext}"
                try:
                    # Open, convert to RGB (handles RGBA), save as JPEG/PNG
                    with Image.open(src) as img:
                        if img.mode == "RGBA":
                            img = img.convert("RGB")
                        elif img.mode != "RGB":
                            img = img.convert("RGB")
                        img.save(dst)
                    split_counts[split_name][cls] += 1
                except Exception as e:
                    corrupt_count += 1

    if corrupt_count > 0:
        print(f"  [WARN] {corrupt_count} corrupt images skipped")

    return split_counts


# === Reporting ================================================================
def print_report(split_counts):
    """Print summary table."""
    print("\n" + "=" * 65)
    print("  DATASET PREPARATION REPORT")
    print("=" * 65)
    print(f"  {'Class':<12} {'Train':>7} {'Val':>7} {'Test':>7} {'Total':>7}")
    print("  " + "-" * 53)

    totals = defaultdict(int)
    for cls in TARGET_CLASSES:
        tr = split_counts["train"].get(cls, 0)
        va = split_counts["val"].get(cls, 0)
        te = split_counts["test"].get(cls, 0)
        total = tr + va + te
        totals["train"] += tr
        totals["val"] += va
        totals["test"] += te
        totals["total"] += total
        print(f"  {cls:<12} {tr:>7} {va:>7} {te:>7} {total:>7}")

    print("  " + "-" * 53)
    print(f"  {'TOTAL':<12} {totals['train']:>7} {totals['val']:>7} {totals['test']:>7} {totals['total']:>7}")
    print("=" * 65)
    print(f"\n  Output: {FINAL_DIR}")
    print(f"  Split:  {SPLIT_RATIOS}")
    print(f"  Seed:   {RANDOM_SEED}")
    print(f"  Cap:    {MAX_PER_CLASS}/class")
    print(f"  Leakage: ZERO (each image in exactly one split)\n")


# === Main =====================================================================
def main():
    print("\n" + "=" * 60)
    print("  CROP CLASSIFICATION - Dataset Preparation (Fixed)")
    print("=" * 60)

    # Step 1: Extract
    print("\n[1/6] Extracting zip files...")
    extract_zips()

    # Step 2: Collect from all 4 datasets
    print("\n[2/6] Collecting images from all datasets...")
    all_collected = defaultdict(list)

    print("  DS1: Agricultural Crops...")
    ds1 = collect_ds1()
    for cls, imgs in ds1.items():
        all_collected[cls].extend(imgs)
        print(f"    {cls}: {len(imgs)}")

    print("  DS2: Agriculture Crop Images (originals only)...")
    ds2 = collect_ds2()
    for cls, imgs in ds2.items():
        all_collected[cls].extend(imgs)
        print(f"    {cls}: {len(imgs)}")

    print("  DS3: Cotton Disease...")
    ds3 = collect_ds3()
    for cls, imgs in ds3.items():
        all_collected[cls].extend(imgs)
        print(f"    {cls}: {len(imgs)}")

    print("  DS4: Soybean Leaf...")
    ds4 = collect_ds4()
    for cls, imgs in ds4.items():
        all_collected[cls].extend(imgs)
        print(f"    {cls}: {len(imgs)}")

    print("\n  Combined totals:")
    for cls in TARGET_CLASSES:
        print(f"    {cls}: {len(all_collected.get(cls, []))}")

    # Step 3: Deduplicate
    print("\n[3/6] Deduplicating by content hash...")
    deduped = deduplicate(all_collected)
    print("  After dedup:")
    for cls in TARGET_CLASSES:
        print(f"    {cls}: {len(deduped.get(cls, []))}")

    # Step 4: Cap classes
    print(f"\n[4/6] Capping large classes to {MAX_PER_CLASS}...")
    capped = cap_classes(deduped, MAX_PER_CLASS)

    # Step 5: Create splits
    print("\n[5/6] Creating stratified train/val/test splits...")
    if FINAL_DIR.exists():
        shutil.rmtree(FINAL_DIR)
    split_counts = create_splits(capped)

    # Step 6: Report
    print("\n[6/6] Report")
    print_report(split_counts)
    print("Done!\n")


if __name__ == "__main__":
    main()
