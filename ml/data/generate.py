import os
import random
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from faker import Faker

fake = Faker()

def create_base_face(color=(200, 200, 200), size=(150, 180)):
    # Create a dummy "face" image block with some texture
    img = Image.new('RGB', size, color)
    draw = ImageDraw.Draw(img)
    # add some noise
    for _ in range(100):
        x = random.randint(0, size[0]-1)
        y = random.randint(0, size[1]-1)
        draw.point((x, y), fill=(color[0]-20, color[1]-20, color[2]-20))
    # draw a generic silhouette
    draw.ellipse((30, 20, 120, 100), fill=(color[0]-40, color[1]-40, color[2]-40))
    draw.ellipse((10, 100, 140, 200), fill=(color[0]-40, color[1]-40, color[2]-40))
    return img

def generate_authentic(id_num):
    width, height = 800, 500
    img = Image.new('RGB', (width, height), (245, 245, 245))
    draw = ImageDraw.Draw(img)
    
    # Try to load a default font, otherwise fallback
    try:
        title_font = ImageFont.truetype("arial.ttf", 36)
        text_font = ImageFont.truetype("arial.ttf", 24)
        mrz_font = ImageFont.truetype("cour.ttf", 28)
    except IOError:
        title_font = ImageFont.load_default()
        text_font = ImageFont.load_default()
        mrz_font = ImageFont.load_default()
    
    # Header
    draw.rectangle([0, 0, 800, 80], fill=(40, 60, 120))
    draw.text((30, 20), "REPUBLIC OF FICTION - IDENTITY CARD", fill="white", font=title_font)
    
    # Data fields
    name = fake.name().upper()
    dob = fake.date_of_birth(minimum_age=18, maximum_age=70).strftime("%d %b %Y")
    doc_no = f"ID{random.randint(1000000, 9999999)}"
    
    draw.text((250, 120), "Name:", fill="black", font=text_font)
    draw.text((250, 150), name, fill=(20, 20, 20), font=title_font)
    
    draw.text((250, 220), "Date of Birth:", fill="black", font=text_font)
    draw.text((250, 250), dob, fill=(20, 20, 20), font=text_font)
    
    draw.text((250, 300), "Document Number:", fill="black", font=text_font)
    draw.text((250, 330), doc_no, fill=(20, 20, 20), font=text_font)
    
    # Dummy Photo
    face_color = (random.randint(150, 220), random.randint(150, 220), random.randint(150, 220))
    face_img = create_base_face(color=face_color)
    img.paste(face_img, (50, 120))
    
    # MRZ Zone
    draw.rectangle([0, 400, 800, 500], fill=(230, 230, 230))
    draw.text((20, 420), f"I<FCT{doc_no}<<<<<<<<<<<<<<<<<<", fill="black", font=mrz_font)
    draw.text((20, 460), f"9901010M2501019FCT<<<<<<<<<<<02", fill="black", font=mrz_font)
    
    # Add subtle overall noise (simulating scan/photo)
    noise = Image.effect_noise((width, height), 5).convert('RGB')
    img = Image.blend(img, noise, 0.05)
    
    return img

def create_tampered(authentic_img):
    img = authentic_img.copy()
    draw = ImageDraw.Draw(img)
    tamper_type = random.choice(['photo_swap', 'text_edit'])
    
    try:
        text_font = ImageFont.truetype("arial.ttf", 24)
        title_font = ImageFont.truetype("arial.ttf", 36)
    except IOError:
        text_font = ImageFont.load_default()
        title_font = ImageFont.load_default()
        
    if tamper_type == 'photo_swap':
        # Create a new completely different face and paste it over, simulating splicing
        new_face_color = (random.randint(150, 220), random.randint(150, 220), random.randint(150, 220))
        new_face = create_base_face(color=new_face_color)
        # Maybe scale or rotate it slightly for bad forgery
        if random.random() > 0.5:
            new_face = new_face.resize((155, 185))
        img.paste(new_face, (48, 118))
        
    elif tamper_type == 'text_edit':
        # Erase DOB and write a new one
        draw.rectangle([250, 250, 450, 280], fill=(245, 245, 245)) # Base color
        # add slight mismatch color to the rectangle to simulate manual edit
        draw.rectangle([250, 250, 450, 280], fill=(250, 250, 240))
        fake_dob = fake.date_of_birth(minimum_age=18, maximum_age=70).strftime("%d %b %Y")
        draw.text((252, 252), fake_dob, fill=(0, 0, 0), font=text_font)

    # Re-save with JPEG compression to ensure the newly pasted areas have different compression grid (ELA target)
    return img

def main():
    os.makedirs('dataset/authentic', exist_ok=True)
    os.makedirs('dataset/tampered', exist_ok=True)
    
    num_samples = 200
    print(f"Generating {num_samples} authentic and {num_samples} tampered IDs...")
    
    for i in range(num_samples):
        # Generate Authentic
        auth_img = generate_authentic(i)
        auth_path = f"dataset/authentic/auth_{i:04d}.jpg"
        auth_img.save(auth_path, format="JPEG", quality=95)
        
        # Generate Tampered
        tamp_img = create_tampered(auth_img)
        tamp_path = f"dataset/tampered/tamp_{i:04d}.jpg"
        # Save at lower/different quality to simulate editing resave
        tamp_img.save(tamp_path, format="JPEG", quality=85)
        
    print("Generation complete!")

if __name__ == "__main__":
    main()
