from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List
import os
import io
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image, ImageChops, ImageEnhance
import exifread

app = FastAPI(title="Tamper Detection Service", description="Detects physical and digital document tampering")

class TamperResponse(BaseModel):
    status: str
    score: float
    flags: List[str]
    explainability: dict

# Global model loading
device = torch.device("cpu")
model = None

try:
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    
    # Try to load weights from the ml/training/checkpoints directory if mounted
    checkpoint_path = "/app/ml/training/checkpoints/tamper_resnet18.pth"
    # Fallback to local absolute path if running locally outside docker
    if not os.path.exists(checkpoint_path):
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        checkpoint_path = os.path.join(base_dir, "ml", "training", "checkpoints", "tamper_resnet18.pth")
        
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
        model.eval()
        print("Model loaded successfully.")
    else:
        print("Checkpoint not found, running in mock model mode.")
        model = None
except Exception as e:
    print(f"Error loading model (using mock instead): {e}")
    model = None

data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def compute_ela(img_bytes, quality=90, scale=10):
    original = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    tmp_io = io.BytesIO()
    original.save(tmp_io, 'JPEG', quality=quality)
    tmp_io.seek(0)
    compressed = Image.open(tmp_io).convert('RGB')
    
    ela_image = ImageChops.difference(original, compressed)
    extrema = ela_image.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    
    if max_diff == 0:
        max_diff = 1
    
    scale_factor = 255.0 / max_diff
    ela_image = ImageEnhance.Brightness(ela_image).enhance(scale_factor * scale)
    return ela_image, max_diff

def analyze_exif(img_bytes):
    tags = exifread.process_file(io.BytesIO(img_bytes))
    flags = []
    software = None
    
    software_tag = tags.get('Image Software')
    if software_tag:
        software = str(software_tag)
        suspicious_editors = ['photoshop', 'gimp', 'paint', 'lightroom']
        if any(ed in software.lower() for ed in suspicious_editors):
            flags.append(f"Image edited with: {software}")
            
    # Check for original vs modified datetime mismatch
    dt_orig = tags.get('EXIF DateTimeOriginal')
    dt_mod = tags.get('Image DateTime')
    if dt_orig and dt_mod and str(dt_orig) != str(dt_mod):
        flags.append("Metadata timestamps do not match (DateTimeOriginal vs DateTime)")
        
    return flags, software

@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": model is not None}

@app.post("/detect", response_model=TamperResponse)
async def detect_tampering(file: UploadFile = File(...)):
    img_bytes = await file.read()
    flags = []
    
    # 1. EXIF Analysis
    exif_flags, software_sig = analyze_exif(img_bytes)
    flags.extend(exif_flags)
    
    # 2. ELA Analysis
    try:
        ela_img, max_val = compute_ela(img_bytes)
    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid image file")
    
    # 3. Model Inference (or Mock)
    cnn_score = 0.0
    if model is not None:
        with torch.no_grad():
            tensor_img = data_transform(ela_img).unsqueeze(0).to(device)
            outputs = model(tensor_img)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            cnn_score = probs[0][1].item() # probability of 'tampered'
    else:
        # Fallback heuristic using ELA max val and EXIF
        cnn_score = min(max_val / 255.0, 1.0)
        
    if cnn_score > 0.5:
        flags.append(f"CNN detected ELA anomalies (confidence: {cnn_score:.2f})")
        
    # Fusion (Weighted Average: 70% CNN, 30% EXIF)
    exif_score = 1.0 if exif_flags else 0.0
    final_score = (cnn_score * 0.7) + (exif_score * 0.3)
    
    status = "success"
    if final_score > 0.6:
        status = "danger"
    elif final_score > 0.3:
        status = "warning"
        
    return {
        "status": status,
        "score": round(final_score, 2),
        "flags": flags,
        "explainability": {
            "ela_max_val": float(max_val),
            "metadata_software_signature": software_sig or "None",
            "heatmap_grid": [[cnn_score, cnn_score], [cnn_score, cnn_score]] # Mock grid matching score for demo
        }
    }
