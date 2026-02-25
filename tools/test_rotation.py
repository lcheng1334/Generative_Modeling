"""
Rotation with background padding (no distortion)
Input: 1 image -> Output: 4 images
L90/R90: rotate then pad with background color to keep original size
"""
import sys
import os
import numpy as np
from PIL import Image


def get_dominant_bg_color(img):
    """Get the background color by sampling corners."""
    arr = np.array(img)
    h, w = arr.shape[:2]
    # Sample 4 corners (20x20 each)
    corners = [
        arr[:20, :20],       # top-left
        arr[:20, -20:],      # top-right
        arr[-20:, :20],      # bottom-left
        arr[-20:, -20:],     # bottom-right
    ]
    all_pixels = np.concatenate([c.reshape(-1, 3) for c in corners], axis=0)
    bg_color = tuple(np.median(all_pixels, axis=0).astype(int))
    return bg_color


def rotate_with_padding(img, angle, target_size, bg_color):
    """Rotate image and pad with bg_color to reach target_size."""
    # Rotate with expand=True to avoid cropping
    rotated = img.rotate(angle, expand=True, resample=Image.LANCZOS)
    rw, rh = rotated.size
    tw, th = target_size
    
    # Create canvas with bg_color
    canvas = Image.new('RGB', (tw, th), bg_color)
    
    # Center the rotated image on canvas
    offset_x = (tw - rw) // 2
    offset_y = (th - rh) // 2
    canvas.paste(rotated, (offset_x, offset_y))
    
    return canvas


def augment_and_save(input_path):
    img = Image.open(input_path)
    w, h = img.size
    bg_color = get_dominant_bg_color(img)
    base, ext = os.path.splitext(input_path)
    
    print(f"Input: {w}x{h}, Background color: RGB{bg_color}")
    
    # Original
    out = f"{base}_0_original{ext}"
    img.save(out)
    print(f"[1/4] Original  ({w}x{h}): {os.path.basename(out)}")
    
    # Left 90 (counterclockwise) + pad
    img_l90 = rotate_with_padding(img, 90, (w, h), bg_color)
    out = f"{base}_1_L90{ext}"
    img_l90.save(out)
    print(f"[2/4] Left 90   ({w}x{h}): {os.path.basename(out)}")
    
    # Right 90 (clockwise) + pad
    img_r90 = rotate_with_padding(img, -90, (w, h), bg_color)
    out = f"{base}_2_R90{ext}"
    img_r90.save(out)
    print(f"[3/4] Right 90  ({w}x{h}): {os.path.basename(out)}")
    
    # 180 (no padding needed, same dimensions)
    img_180 = img.rotate(180)
    out = f"{base}_3_180{ext}"
    img_180.save(out)
    print(f"[4/4] Rotate 180({w}x{h}): {os.path.basename(out)}")
    
    print(f"\nDone! Background padding color: RGB{bg_color}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_rotation.py <image_path>")
        sys.exit(1)
    
    path = sys.argv[1]
    if not os.path.exists(path):
        print(f"File not found: {path}")
        sys.exit(1)
    
    augment_and_save(path)
