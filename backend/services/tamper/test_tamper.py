import os
import sys
import io

# Fix Windows terminal encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import tampering_service

def main():
    print("\n--- Tampering Service Local Test ---")

    if not tampering_service._model_loaded:
        print("ERROR: Model failed to load!")
        print("Make sure tamper_multihead_resnet18.pth is in this folder:")
        print(os.path.dirname(os.path.abspath(__file__)))
        return

    print("OK: Model loaded successfully!")
    print("\nDrag & drop an image file here, or type the full path.")

    img_path = input("\nEnter image path (or press Enter to quit): ").strip().strip('"\'')

    if not img_path:
        return

    if not os.path.exists(img_path):
        print(f"ERROR: File not found: {img_path}")
        return

    print(f"\nAnalyzing: {img_path}")

    with open(img_path, "rb") as f:
        img_bytes = f.read()

    result = tampering_service.analyze_tampering(img_bytes)

    print("\n" + "="*44)
    print("            ANALYSIS RESULT")
    print("="*44)
    print(f"  Total Risk Score : {result['score']}/100")
    print(f"  Verdict          : {'[SUSPICIOUS]' if result['suspicious'] else '[CLEAN]'}")
    print()
    print("  Sub-scores:")
    for k, v in result['details'].items():
        if k == 'visualization':
            continue
        print(f"    {k:<24}: {v}")
    print()
    print("  Indicators:")
    if result['indicators']:
        for ind in result['indicators']:
            print(f"    [!] {ind}")
    else:
        print("    [OK] No tampering indicators found")
    print("="*44 + "\n")

if __name__ == "__main__":
    main()
