"""
Deep feature extraction from Fashion Product Images using pre-trained ResNet50 model
"""

import numpy as np
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from PIL import Image as PILImage

class FeatureExtractor:
    def __init__(self, batch_size=32):
        self.model = None
        self.feature_cache = {}
        self.feature_dim = 2048  # ResNet50 output dimension
        self.batch_size = batch_size
        
    def build_model(self):
        print("Loading pre-trained ResNet50 model (ImageNet)...")
        
        self.model = ResNet50(
            weights='imagenet', 
            include_top=False, 
            pooling='avg'
        )
        
        print("ResNet50 model loaded successfully!")
        return self.model
    
    def preprocess_image(self, image):
        if isinstance(image, PILImage.Image):
            img = image.resize((224, 224), PILImage.LANCZOS)
            img_array = np.array(img)
        else:
            if image.dtype == np.float32 or image.dtype == np.float64:
                img_uint8 = (image * 255).astype('uint8')
            else:
                img_uint8 = image.astype('uint8')
            
            img_pil = PILImage.fromarray(img_uint8)
            img_resized = img_pil.resize((224, 224), PILImage.LANCZOS)
            img_array = np.array(img_resized)
        
        img_array = np.expand_dims(img_array, axis=0)
        
        return preprocess_input(img_array.astype('float32'))
    
    def extract_features(self, dataset_obj):
        if self.model is None:
            self.build_model()
            
        batch_size = batch_size if batch_size is not None else self.batch_size
        n_images = len(dataset_obj.image_ids)
        all_features = np.zeros((n_images, self.feature_dim))
        
        print(f"Extracting features from {n_images} images (Batch size: {batch_size})...")
        
        for i in range(0, n_images, batch_size):
            end_idx = min(i + batch_size, n_images)
            batch_imgs = []
            
            for j in range(i, end_idx):
                try:
                    img = dataset_obj.get_image(j, target_size=(224, 224))
                    prep_img = self.preprocess_image(img)
                    batch_imgs.append(prep_img)
                except Exception as e:
                    print(f"Error processing image {j}: {e}")
                    batch_imgs.append(np.zeros((1, 224, 224, 3), dtype='float32'))
            
            batch_tensor = np.vstack(batch_imgs)
            features = self.model.predict(batch_tensor, verbose=0)
            all_features[i:end_idx] = self._normalize_features(features)
            
            print(f"Processed {end_idx}/{n_images} images ({100*end_idx//n_images}%)")
                
        print("Feature extraction completed!")
        return all_features
    
    def extract_single_feature(self, image, image_id=None):
        if image_id is not None and image_id in self.feature_cache:
            return self.feature_cache[image_id]
        
        if self.model is None:
            self.build_model()
        
        prep_img = self.preprocess_image(image)
        feature = self.model.predict(prep_img, verbose=0)
        feature = self._normalize_features(feature)[0]
        
        if image_id is not None:
            self.feature_cache[image_id] = feature
            
        return feature

    def _normalize_features(self, features):
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        return features / norms

    def save_features(self, features, filepath):
        np.save(filepath, features)
        print(f"Saved features to {filepath}")

    def load_features(self, filepath):
        features = np.load(filepath)
        print(f"Loaded features: {features.shape}")
        return features
