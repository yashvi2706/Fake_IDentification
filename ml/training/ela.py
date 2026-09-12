import os
from PIL import Image, ImageChops, ImageEnhance

def compute_ela(img_path, quality=90, scale=10):
    """
    Computes the Error Level Analysis (ELA) of an image.
    This highlights areas of the image that have been saved at different compression levels,
    which is a strong indicator of splicing/copy-move forgery.
    """
    original = Image.open(img_path).convert('RGB')
    
    # Save to a temporary file at the given quality
    tmp_path = 'temp_ela.jpg'
    original.save(tmp_path, 'JPEG', quality=quality)
    
    compressed = Image.open(tmp_path).convert('RGB')
    
    # Calculate the absolute difference between original and re-compressed
    ela_image = ImageChops.difference(original, compressed)
    
    # Get the extrema (min, max) to calculate scaling
    extrema = ela_image.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    
    if max_diff == 0:
        max_diff = 1 # Avoid division by zero
        
    scale_factor = 255.0 / max_diff
    
    # Enhance the difference
    ela_image = ImageEnhance.Brightness(ela_image).enhance(scale_factor * scale)
    
    # Clean up temp file
    if os.path.exists(tmp_path):
        os.remove(tmp_path)
        
    return ela_image, max_diff
