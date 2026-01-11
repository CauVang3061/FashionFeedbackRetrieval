"""
Importing and processing Fashion Product Image dataset
"""

import pandas as pd
import numpy as np
from PIL import Image
import os

class FashionProductDataset:
    def __init__(self, subset_size, data_dir="data"):
        self.subset_size = subset_size
        self.data_dir = data_dir
        self.img_dir = os.path.join(data_dir, "images")
        self.csv_path = os.path.join(data_dir, "styles.csv")
        self.df = None
        self.image_ids = None
        self.labels = None
        
    def load_data(self):
        print(f"Loading Fashion Product dataset from {self.csv_path}...")

        self.df = pd.read_csv(self.csv_path, on_bad_lines='skip', encoding='utf-8', encoding_errors='ignore')
        self.df['exists'] = self.df['id'].apply(lambda x: os.path.exists(os.path.join(self.img_dir, f"{int(x)}.jpg")))
        self.df = self.df[self.df['exists'] == True].reset_index(drop=True)
        
        if self.subset_size < len(self.df):
            self.df = self.df.sample(n=self.subset_size, random_state=42).reset_index(drop=True)
        
        self.image_ids = self.df['id'].values
        self.labels = self.df['articleType'].values
        
        print(f"Loaded {len(self.df)} images")
        
        return self.image_ids, self.labels
    
    def get_image(self, idx, target_size=(224, 224)):
        if self.df is None:
            raise ValueError("Dataset not loaded. Call load_data() first.")
        
        img_id = self.image_ids[idx]
        img_path = os.path.join(self.img_dir, f"{int(img_id)}.jpg")

        try:
            img = Image.open(img_path).convert('RGB')
            if target_size:
                img = img.resize(target_size, Image.LANCZOS)
            return img
        except Exception as e:
            raise RuntimeError(f"Failed to load image {img_id}: {e}")
    
    def get_label(self, idx):
        if self.labels is None:
            raise ValueError("Dataset not loaded. Call load_data() first.")
        return self.labels[idx]
    
    def preprocess_cnn(self, target_size=(224, 224)):
        print(f"Preprocessing {len(self.image_ids)} images for CNN...")
        processed_images = []
        
        for i in range(len(self.image_ids)):
            img = self.get_image(i, target_size=target_size)
            img_array = np.array(img).astype('float32') / 255.0
            processed_images.append(img_array)
            
        return np.array(processed_images)
    
    def get_random_sample(self, n=10):
        if self.image_ids is None:
            raise ValueError("Dataset not loaded. Call load_data() first.")
        return np.random.choice(len(self.image_ids), n, replace=False)

    def save_image(self, idx, filepath):
        img = self.get_image(idx, target_size=None)
        img.save(filepath)
        print(f"Saved image {idx} (ID: {self.image_ids[idx]}) to {filepath}")
    
    def get_unique_labels(self):
        if self.labels is None:
            raise ValueError("Dataset not loaded. Call load_data() first.")
        return list(set(self.labels))
