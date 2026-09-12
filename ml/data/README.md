# ML Data Pipeline: Synthetic Forgery Generator

Since no public dataset of forged government IDs exists due to legal and security reasons, this module programmatically generates a synthetic dataset.

## The Approach

We use `Pillow` and the `Faker` library to generate templates.
1. **Authentic Generation**: `generate.py` builds a template ID card layout (Republic of Fiction) with randomized text (Name, DOB, ID number), a dummy photo block, and MRZ text. It saves this as a high-quality JPEG (`quality=95`).
2. **Tampered Generation**: The generator takes the authentic ID and applies one of two attacks:
   - **Photo Swap**: Simulating copy-move/splicing, a completely new dummy photo is pasted over the original.
   - **Text Edit**: The Date of Birth field is painted over with a slightly off-color box and a new DOB is written.
   The tampered image is then saved with slightly lower JPEG quality (`quality=85`), which mimics an attacker opening the image, editing it, and resaving it.

This differential compression grid is exactly what our Error Level Analysis (ELA) model targets.

## Usage

```bash
pip install pillow faker
python generate.py
```

This will populate the `dataset/authentic/` and `dataset/tampered/` folders with 200 images each.
*(Note: The `dataset/` folder is gitignored to save repository space.)*
