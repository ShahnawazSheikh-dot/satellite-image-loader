import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

class ForestImageProcessor:
    def __init__(self):
        print("🌳 Forest Image Processor Ready!")
    
    def load_image(self, image_path):
        try:
            img = Image.open(image_path)
            return np.array(img)
        except Exception as e:
            print(f"Error loading {image_path}: {e}")
            return None
    
    def enhance_forest_image(self, image):
        enhanced = image.astype(np.float32)
        enhanced = enhanced * 1.4
        min_val = np.min(enhanced)
        max_val = np.max(enhanced)
        if max_val > min_val:
            enhanced = (enhanced - min_val) / (max_val - min_val) * 255
        return enhanced.clip(0, 255).astype(np.uint8)
    
    def process_forest_images(self):
        image_files = [f for f in os.listdir('images/raw') if f.endswith(('.jpg', '.png'))]
        if not image_files:
            print("❌ No forest images found!")
            return
        for image_file in image_files:
            print(f"🌲 Processing: {image_file}")
            original = self.load_image(f'images/raw/{image_file}')
            if original is None:
                continue
            enhanced = self.enhance_forest_image(original)
            output_name = f'enhanced_{image_file}'
            Image.fromarray(enhanced).save(f'images/processed/{output_name}')
            print(f"💾 Saved: {output_name}")

if __name__ == "__main__":
    processor = ForestImageProcessor()
    processor.process_forest_images()
