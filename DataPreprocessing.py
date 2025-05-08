import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

def add_noise(image):
    noise = np.random.normal(loc=0, scale=25, size=image.shape)
    noisy = np.clip(image + noise, 0, 255).astype(np.uint8)
    return noisy

def preprocess_tiny_imagenet(input_dir, output_dir, img_size=(128, 128)):
    os.makedirs(output_dir, exist_ok=True)
    
    # Process training data
    train_dir = os.path.join(input_dir, 'train')
    print(f"Processing training data from: {train_dir}")  # Debug
    
    for class_dir in os.listdir(train_dir):
        class_path = os.path.join(train_dir, class_dir, 'images')
        if not os.path.isdir(class_path):
            continue
        
        print(f"\nProcessing class: {class_dir}")  # Debug
        images = []
        
        # Process each image in the class directory
        for img_file in os.listdir(class_path):
            img_path = os.path.join(class_path, img_file)
            
            # Skip non-image files
            if not img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.jpeg')):
                print(f"Skipping non-image file: {img_file}")
                continue
            
            img = cv2.imread(img_path)
            if img is None:
                print(f"Failed to read: {img_path}")
                continue
            
            img = cv2.resize(img, img_size)
            images.append(img)
        
        if not images:
            print(f"No valid images found for class: {class_dir}")
            continue
        
        # Split and save data
        train, val = train_test_split(images, test_size=0.2)
        print(f"  Saving {len(train)} train images and {len(val)} validation images")
        save_split(train, output_dir, 'train', class_dir, img_size)
        save_split(val, output_dir, 'val', class_dir, img_size)
    
    # Process validation data (same as before)
    # ... [rest of the validation processing code] ...

def save_split(images, base_dir, split, class_dir, img_size):
    split_dir = os.path.join(base_dir, split, class_dir)
    os.makedirs(split_dir, exist_ok=True)  # Creates parent directories if needed
    for i, img in enumerate(images):
        cv2.imwrite(os.path.join(split_dir, f'clean_{i}.png'), img)
        cv2.imwrite(os.path.join(split_dir, f'noisy_{i}.png'), add_noise(img))

if __name__ == "__main__":
    preprocess_tiny_imagenet(
        input_dir='/home/ccx4276/GENAI/tiny-imagenet-200/',
        output_dir='/home/ccx4276/GENAI/processed/'
    )
