import os
import random
import time
import requests
import io
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance, ImageMath, ImageChops
from faker import Faker
import numpy as np

fake = Faker()

def fetch_faces(num_faces=30, output_dir="faces"):
    os.makedirs(output_dir, exist_ok=True)
    print(f"Checking face cache in {output_dir}...")
    existing = [f for f in os.listdir(output_dir) if f.endswith('.jpg')]
    if len(existing) >= num_faces:
        print("Faces already cached.")
        return
        
    print(f"Downloading {num_faces - len(existing)} synthetic faces...")
    for i in range(len(existing), num_faces):
        gender = random.choice(['men', 'women'])
        idx = random.randint(1, 99)
        url = f"https://randomuser.me/api/portraits/{gender}/{idx}.jpg"
        try:
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200:
                with open(os.path.join(output_dir, f"face_{i:03d}.jpg"), 'wb') as f:
                    f.write(resp.content)
                time.sleep(0.1)
        except Exception as e:
            print(f"Failed to fetch face {i}: {e}")
            img = Image.effect_noise((200, 200), 50).convert('RGB')
            img.save(os.path.join(output_dir, f"face_{i:03d}.jpg"))

def get_random_face(faces_dir="faces"):
    faces = [f for f in os.listdir(faces_dir) if f.endswith('.jpg')]
    if not faces:
        img = Image.effect_noise((200, 200), 50).convert('RGB')
        return img
    face_path = os.path.join(faces_dir, random.choice(faces))
    return Image.open(face_path).convert("RGB")

def apply_paper_texture_and_lighting(img):
    # Add noise texture
    w, h = img.size
    noise = Image.effect_noise((w, h), 10).convert('RGB')
    img = Image.blend(img, noise, 0.05)
    
    # Create radial gradient for lighting
    gradient = Image.new('L', (w, h), color=0)
    draw = ImageDraw.Draw(gradient)
    center_x, center_y = random.randint(w//4, 3*w//4), random.randint(h//4, 3*w//4)
    max_radius = int((w**2 + h**2)**0.5)
    
    for r in range(max_radius, 0, -5):
        alpha = int(255 * (r / max_radius))
        draw.ellipse([center_x - r, center_y - r, center_x + r, center_y + r], fill=alpha)
    
    gradient = ImageEnhance.Brightness(gradient).enhance(0.5)
    img = ImageChops.screen(img, gradient.convert('RGB'))
    return img

def create_layout_landscape(face_img, name, dob, doc_no):
    w, h = 800, 500
    img = Image.new('RGB', (w, h), (240, 245, 250))
    draw = ImageDraw.Draw(img)
    
    try:
        title_font = ImageFont.truetype("arialbd.ttf", 36)
        text_font = ImageFont.truetype("arial.ttf", 24)
        mrz_font = ImageFont.truetype("cour.ttf", 28)
    except IOError:
        title_font = ImageFont.load_default()
        text_font = ImageFont.load_default()
        mrz_font = ImageFont.load_default()
        
    draw.rectangle([0, 0, w, 80], fill=(20, 40, 100))
    draw.text((30, 20), "REPUBLIC OF FICTION - VISA", fill="white", font=title_font)
    
    # Resize face
    face_img = face_img.resize((150, 180))
    img.paste(face_img, (50, 120))
    
    draw.text((250, 120), "Name:", fill=(100,100,100), font=text_font)
    draw.text((250, 150), name, fill=(20, 20, 20), font=title_font)
    
    draw.text((250, 220), "Date of Birth:", fill=(100,100,100), font=text_font)
    draw.text((250, 250), dob, fill=(20, 20, 20), font=text_font)
    
    draw.text((500, 220), "Document No:", fill=(100,100,100), font=text_font)
    draw.text((500, 250), doc_no, fill=(20, 20, 20), font=text_font)
    
    # MRZ
    draw.rectangle([0, 400, w, h], fill=(255, 255, 255))
    draw.text((20, 415), f"V<FCT{doc_no}<<<<<<<<<<<<<<<<<<", fill="black", font=mrz_font)
    draw.text((20, 455), f"9901010M2501019FCT<<<<<<<<<<<02", fill="black", font=mrz_font)
    
    return img, {'dob_box': (250, 250, 450, 280), 'face_box': (50, 120, 200, 300)}

def create_layout_portrait(face_img, name, dob, doc_no):
    w, h = 500, 800
    img = Image.new('RGB', (w, h), (250, 240, 235))
    draw = ImageDraw.Draw(img)
    
    try:
        title_font = ImageFont.truetype("arialbd.ttf", 30)
        text_font = ImageFont.truetype("arial.ttf", 22)
    except IOError:
        title_font = ImageFont.load_default()
        text_font = ImageFont.load_default()
        
    draw.rectangle([0, 0, w, 60], fill=(120, 40, 40))
    draw.text((20, 15), "NATIONAL ID CARD", fill="white", font=title_font)
    
    face_img = face_img.resize((200, 240))
    img.paste(face_img, (150, 80))
    
    draw.text((50, 350), "Name:", fill=(100,100,100), font=text_font)
    draw.text((50, 380), name, fill=(20, 20, 20), font=title_font)
    
    draw.text((50, 450), "DOB:", fill=(100,100,100), font=text_font)
    draw.text((50, 480), dob, fill=(20, 20, 20), font=text_font)
    
    draw.text((50, 550), "ID No:", fill=(100,100,100), font=text_font)
    draw.text((50, 580), doc_no, fill=(20, 20, 20), font=text_font)
    
    return img, {'dob_box': (50, 480, 250, 510), 'face_box': (150, 80, 350, 320)}

def simulate_photo_capture(img):
    # Rotate slightly
    angle = random.uniform(-2, 2)
    img = img.rotate(angle, resample=Image.BICUBIC, expand=True, fillcolor=(200,200,200))
    # Add slight blur
    if random.random() > 0.5:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 0.8)))
    return img

def tamper_photo_swap(img, boxes, faces_dir):
    new_img = img.copy()
    face_box = boxes['face_box']
    w = face_box[2] - face_box[0]
    h = face_box[3] - face_box[1]
    
    new_face = get_random_face(faces_dir).resize((w, h))
    
    # Degrade the pasted face differently to create ELA signature
    buf = io.BytesIO()
    new_face.save(buf, format='JPEG', quality=60)
    buf.seek(0)
    new_face = Image.open(buf).convert('RGB')
    
    # Create soft feathered mask
    mask = Image.new('L', (w, h), 0)
    draw = ImageDraw.Draw(mask)
    draw.rectangle([10, 10, w-10, h-10], fill=255)
    mask = mask.filter(ImageFilter.GaussianBlur(radius=5))
    
    new_img.paste(new_face, (face_box[0], face_box[1]), mask=mask)
    return new_img

def tamper_text_edit(img, boxes):
    new_img = img.copy()
    dob_box = boxes['dob_box']
    
    # Sample background color near the text
    sample_x, sample_y = dob_box[0] - 10, dob_box[1] - 10
    if sample_x < 0 or sample_y < 0:
        sample_x, sample_y = dob_box[0], dob_box[1]
        
    bg_color = img.getpixel((sample_x, sample_y))
    
    # Create patch
    w = dob_box[2] - dob_box[0] + 20
    h = dob_box[3] - dob_box[1] + 10
    patch = Image.new('RGB', (w, h), bg_color)
    
    # Add noise to patch matching image
    noise = Image.effect_noise((w, h), 10).convert('RGB')
    patch = Image.blend(patch, noise, 0.05)
    
    # Soft mask
    mask = Image.new('L', (w, h), 0)
    draw = ImageDraw.Draw(mask)
    draw.rectangle([5, 5, w-5, h-5], fill=255)
    mask = mask.filter(ImageFilter.GaussianBlur(radius=3))
    
    new_img.paste(patch, (dob_box[0]-10, dob_box[1]-5), mask=mask)
    
    # Write new text
    draw_new = ImageDraw.Draw(new_img)
    try:
        text_font = ImageFont.truetype("arial.ttf", 24)
    except IOError:
        text_font = ImageFont.load_default()
        
    new_dob = fake.date_of_birth(minimum_age=18, maximum_age=70).strftime("%d %b %Y")
    draw_new.text((dob_box[0], dob_box[1]), new_dob, fill=(0,0,0), font=text_font)
    
    return new_img

def main():
    fetch_faces(num_faces=30)
    os.makedirs('dataset/authentic', exist_ok=True)
    os.makedirs('dataset/tampered', exist_ok=True)
    
    num_samples = 200
    print(f"Generating {num_samples} realistic authentic and {num_samples} tampered IDs...")
    
    for i in range(num_samples):
        name = fake.name().upper()
        dob = fake.date_of_birth(minimum_age=18, maximum_age=70).strftime("%d %b %Y")
        doc_no = f"ID{random.randint(1000000, 9999999)}"
        face = get_random_face()
        
        # 1. Base generation
        if random.random() > 0.5:
            base_img, boxes = create_layout_landscape(face, name, dob, doc_no)
        else:
            base_img, boxes = create_layout_portrait(face, name, dob, doc_no)
            
        base_img = apply_paper_texture_and_lighting(base_img)
        
        # 2. Simulate photo capture (authentic)
        auth_img = simulate_photo_capture(base_img)
        
        # 3. Save Authentic (Quality 95)
        auth_path = f"dataset/authentic/auth_{i:04d}.jpg"
        auth_img.save(auth_path, format="JPEG", quality=95)
        
        # 4. Tamper on the captured photo
        # First we need to map the bounding boxes through the rotation, but for simplicity
        # we apply tampering BEFORE rotation/blur in this pipeline so bounding boxes are fixed,
        # and THEN apply capture simulation. Wait, if tampering is digital, it happens AFTER capture.
        # However, to avoid complex bounding box math during rotation, we can tamper the base,
        # then apply the SAME rotation/blur to the tampered image so they align perfectly.
        
        if random.random() > 0.5:
            tamp_base = tamper_photo_swap(base_img, boxes, "faces")
        else:
            tamp_base = tamper_text_edit(base_img, boxes)
            
        tamp_img = simulate_photo_capture(tamp_base) # Apply same/similar capture effects
        
        # Save Tampered (Quality 70 to create ELA mismatch on edited pixels)
        tamp_path = f"dataset/tampered/tamp_{i:04d}.jpg"
        tamp_img.save(tamp_path, format="JPEG", quality=70)
        
    print("Generation complete!")

if __name__ == "__main__":
    main()
