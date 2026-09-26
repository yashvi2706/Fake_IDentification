"""
Benchmark Script for the Fake IDentification System.

Iterates over a directory of test images, runs the full analysis pipeline,
and generates an accuracy/performance report.
"""
import os
import time
import json
import argparse
from prettytable import PrettyTable
import sys

# Add parent directory to path so we can import backend modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from services.ocr_service import extract_document_data
from services.validation_service import validate_document
from services.tamper.tampering_service import analyze_tampering
from services.face_service import verify_faces
from services.quality_service import analyze_image_quality
from services.template_service import analyze_document_layout
from services.risk_service import calculate_risk

def run_benchmark(test_dir: str):
    """Run benchmark against a directory of test images."""
    if not os.path.exists(test_dir):
        print(f"Error: Directory {test_dir} not found.")
        return

    images = [f for f in os.listdir(test_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not images:
        print(f"No images found in {test_dir}.")
        return

    print(f"Starting benchmark on {len(images)} images in {test_dir}...")
    
    table = PrettyTable()
    table.field_names = ["Image", "Time (s)", "Risk Level", "Score", "Quality OK", "Tampered", "Validation OK", "MRZ OK"]

    total_time = 0
    total_high_risk = 0

    for img_name in images:
        img_path = os.path.join(test_dir, img_name)
        
        start_time = time.time()
        
        # Determine likely doc type from filename or default to passport
        doc_type = "national_id"
        if "passport" in img_name.lower():
            doc_type = "passport"
        elif "visa" in img_name.lower():
            doc_type = "visa"
            
        # Run pipeline
        try:
            quality_data = analyze_image_quality(img_path)
            ocr_data = extract_document_data(img_path, doc_type)
            template_data = analyze_document_layout(img_path, doc_type)
            validation_data = validate_document(ocr_data, doc_type)
            tampering_data = analyze_tampering(img_path)
            
            # Simulated dummy selfie check if the filename has "selfie" (for testing)
            face_data = verify_faces(img_path, None) 
            
            risk_data = calculate_risk(
                ocr_data, validation_data, tampering_data, face_data, quality_data, template_data
            )
            
            elapsed = time.time() - start_time
            total_time += elapsed
            
            # Extract key metrics for table
            risk_level = risk_data.get("level", "UNKNOWN")
            if risk_level == "HIGH":
                total_high_risk += 1
                
            score = risk_data.get("score", 0)
            qual_ok = "Yes" if quality_data.get("acceptable") else "No"
            tamp = "Yes" if tampering_data.get("suspicious") else "No"
            val_ok = "Yes" if validation_data.get("valid") else "No"
            
            # Check MRZ specifically
            mrz_ok = "N/A"
            for check in validation_data.get("checks", []):
                if check.get("name") == "MRZ Composite check":
                    mrz_ok = "Yes" if check.get("status") == "pass" else "No"
                    break

            table.add_row([
                img_name[:20], 
                f"{elapsed:.2f}", 
                risk_level, 
                f"{score}", 
                qual_ok, 
                tamp, 
                val_ok,
                mrz_ok
            ])
            
        except Exception as e:
            print(f"Error processing {img_name}: {e}")
            table.add_row([img_name[:20], "ERROR", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A"])

    print("\n--- BENCHMARK RESULTS ---")
    print(table)
    print(f"\nTotal Images Processed: {len(images)}")
    print(f"Average Time per Image: {total_time/len(images):.2f}s")
    print(f"High Risk Documents: {total_high_risk} ({total_high_risk/len(images)*100:.1f}%)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark Fake IDentification System")
    parser.add_argument("--dir", type=str, default="../tests/fixtures", help="Directory containing test images")
    args = parser.parse_args()
    
    run_benchmark(args.dir)
