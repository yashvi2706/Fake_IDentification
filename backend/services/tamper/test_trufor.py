import os
import sys
import subprocess
import numpy as np
from PIL import Image

def main():
    print("\n--- TruFor (CVPR 2023) Local Test ---")
    print("Drag & drop an image file here, or type the full path.")
    
    img_path = input("\nEnter image path (or press Enter to quit): ").strip().strip('"\'')
    if not img_path or not os.path.exists(img_path):
        return

    # Prepare output path
    out_npz = os.path.abspath("trufor_temp.npz")
    if os.path.exists(out_npz):
        os.remove(out_npz)

    # Path to TruFor's internal test script
    trufor_script = os.path.join("TruFor", "test_docker", "src", "trufor_test.py")

    print(f"\n[1/3] Running TruFor model on CPU (this takes a moment)...")
    try:
        # Run TruFor python script
        trufor_src_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "TruFor", "test_docker", "src")
        subprocess.run([
            sys.executable, "trufor_test.py",
            "-gpu", "-1",  # Run on CPU
            "-in", img_path,
            "-out", out_npz
        ], check=True, cwd=trufor_src_dir)
    except subprocess.CalledProcessError:
        print("ERROR: TruFor inference failed. Check the logs above.")
        return

    if not os.path.exists(out_npz):
        print("ERROR: Expected output file was not created.")
        return

    print("[2/3] Parsing output...")
    data = np.load(out_npz)
    score = data.get('score', 0.0)
    loc_map = data.get('map', None)

    print("\n" + "="*44)
    print("            TRUFOR RESULT")
    print("="*44)
    print(f"  Whole-Image Fraud Score : {score:.4f} (0=Clean, 1=Fake)")
    
    if loc_map is not None:
        # Save the localization map as a visible image
        out_png = img_path + "_trufor_mask.png"
        
        # Convert map (0.0 to 1.0) to (0 to 255)
        mask_img = (loc_map * 255).astype(np.uint8)
        Image.fromarray(mask_img).save(out_png)
        
        print(f"  Localization Mask Saved : {out_png}")
        print("    (White areas = manipulated pixels)")
    
    print("="*44 + "\n")
    
    # Clean up temp npz
    if os.path.exists(out_npz):
        os.remove(out_npz)

if __name__ == "__main__":
    main()
