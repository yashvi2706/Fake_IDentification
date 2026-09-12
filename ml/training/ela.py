import os
from PIL import Image, ImageChops, ImageEnhance

def compute_ela(img_path, quality=90, scale=10):
    """
    Computes the Error Level Analysis (ELA) of an image.
    This highlights areas of the image that have been saved at different compression levels,
    which is a strong indicator of splicing/copy-move forgery.
    """
    import io
    original = Image.open(img_path).convert('RGB')
    
    # Save to an in-memory buffer at the given quality
    tmp_buffer = io.BytesIO()
    original.save(tmp_buffer, 'JPEG', quality=quality)
    tmp_buffer.seek(0)
    
    compressed = Image.open(tmp_buffer).convert('RGB')
    
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
    
    # Clean up memory
    original.close()
    compressed.close()
    tmp_buffer.close()
        
    return ela_image, max_diff
