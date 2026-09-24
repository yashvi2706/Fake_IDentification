"""
precompute_ela.py — One-time ELA cache generation

Location: backend/services/tamper/cloud/

Reads normalized/images/, computes ELA for each, saves to normalized/ela_cache/.
This is THE critical budget-saver: avoids re-computing ELA every epoch.

78K images × 10 epochs = 780K redundant ELA ops eliminated.
One-time cost: ~15-30 min.
"""

import os
import io
import sys
from pathlib import Path
from PIL import Image, ImageChops, ImageEnhance
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).parent.resolve()
IMAGES_DIR = SCRIPT_DIR / "normalized" / "images"
ELA_DIR = SCRIPT_DIR / "normalized" / "ela_cache"

# ELA parameters (must match inference service exactly)
ELA_QUALITY = 90
ELA_SCALE = 10


def compute_ela_from_path(img_path, quality=ELA_QUALITY, scale=ELA_SCALE):
    """
    Compute Error Level Analysis for a single image file.
    Returns ELA PIL Image or None on failure.
    """
    try:
        original = Image.open(img_path).convert("RGB")

        # Re-compress at specified quality
        tmp_io = io.BytesIO()
        original.save(tmp_io, "JPEG", quality=quality)
        tmp_io.seek(0)
        compressed = Image.open(tmp_io).convert("RGB")

        # Pixel-level difference
        ela_image = ImageChops.difference(original, compressed)

        # Scale to maximize contrast
        extrema = ela_image.getextrema()
        max_diff = max(ex[1] for ex in extrema)
        if max_diff == 0:
            max_diff = 1

        scale_factor = 255.0 / max_diff
        ela_image = ImageEnhance.Brightness(ela_image).enhance(scale_factor * scale)

        # Clean up
        original.close()
        compressed.close()
        tmp_io.close()

        return ela_image

    except Exception as e:
        # Return a black image on failure (rare — corrupted source)
        return Image.new("RGB", (224, 224), (0, 0, 0))


def process_single_image(args):
    """Process a single image — designed for multiprocessing."""
    src_path, dst_path = args
    
    if dst_path.exists():
        return True  # Already cached

    try:
        ela_img = compute_ela_from_path(src_path)
        if ela_img is not None:
            # Save as JPEG for space efficiency (ELA images compress well)
            ela_img.save(str(dst_path), "JPEG", quality=95)
            ela_img.close()
            return True
    except Exception as e:
        # Create a black placeholder so training doesn't crash
        placeholder = Image.new("RGB", (224, 224), (0, 0, 0))
        placeholder.save(str(dst_path), "JPEG", quality=95)
        placeholder.close()
        return False

    return False


def main():
    print("============================================")
    print("  Precomputing ELA cache")
    print("============================================")
    print()

    if not IMAGES_DIR.exists():
        print(f"ERROR: {IMAGES_DIR} does not exist!")
        print("Run normalize_datasets.py first.")
        sys.exit(1)

    ELA_DIR.mkdir(parents=True, exist_ok=True)

    # Collect all images
    all_images = sorted(IMAGES_DIR.iterdir())
    all_images = [p for p in all_images if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}]

    print(f"  Found {len(all_images)} images to process")

    # Check how many are already cached
    already_cached = sum(1 for img in all_images if (ELA_DIR / f"{img.stem}.jpg").exists())
    if already_cached > 0:
        print(f"  {already_cached} already cached, processing {len(all_images) - already_cached} remaining")

    # Prepare work items
    work_items = []
    for img_path in all_images:
        dst_path = ELA_DIR / f"{img_path.stem}.jpg"
        work_items.append((img_path, dst_path))

    # Use multiprocessing for CPU-bound ELA computation
    num_workers = min(os.cpu_count() or 4, 8)
    print(f"  Using {num_workers} parallel workers")
    print()

    success_count = 0
    fail_count = 0

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_single_image, item): item for item in work_items}
        
        with tqdm(total=len(work_items), desc="Computing ELA", unit="img") as pbar:
            for future in as_completed(futures):
                result = future.result()
                if result:
                    success_count += 1
                else:
                    fail_count += 1
                pbar.update(1)

    print()
    print("============================================")
    print(f"  ELA cache complete!")
    print(f"  Success: {success_count}")
    print(f"  Failed (black placeholder): {fail_count}")
    print(f"  Cache directory: {ELA_DIR}")
    
    # Show cache size
    cache_size_bytes = sum(f.stat().st_size for f in ELA_DIR.iterdir() if f.is_file())
    cache_size_gb = cache_size_bytes / (1024 ** 3)
    print(f"  Cache size: {cache_size_gb:.2f} GB")
    print("============================================")


if __name__ == "__main__":
    main()
