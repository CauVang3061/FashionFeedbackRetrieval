"""
Feature extraction using pre-trained CNN models: ResNet50 for extracting deep features from Fashion-MNIST images
"""

import numpy as np
from tensorflow import keras
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import Model
import tensorflow as tf
from PIL import Image as PILImage
from dataset import FashionMNISTDataset

class FeatureExtractor:
    """Extract deep features using pre-trained CNN"""
    
    def __init__(self, model_name='resnet50'):
        """
        Initialize feature extractor
        
        Args:
            model_name: Name of pre-trained model ('resnet50', 'vgg16')
        """
        self.model_name = model_name
        self.model = None
        self.feature_cache = {}
        
    def build_model(self):
        """Build feature extraction model"""
        print(f"Loading pre-trained {self.model_name} model...")
        
        if self.model_name == 'resnet50':
            # Load ResNet50 pre-trained on ImageNet
            base_model = ResNet50(weights='imagenet', include_top=False, 
                                 pooling='avg')
            self.model = base_model
            
        # elif self.model_name == 'vgg16':
        #     from tensorflow.keras.applications import VGG16
        #     base_model = VGG16(weights='imagenet', include_top=False, 
        #                       pooling='avg')
        #     self.model = base_model
        
        print("Model loaded successfully!")
        return self.model
    
    def preprocess_image(self, image):
        """
        Preprocess single image for CNN
        
        Args:
            image: NumPy array (28x28 or 28x28x3)
            
        Returns:
            Preprocessed image (224x224x3)
        """
        # Convert to RGB if grayscale
        if len(image.shape) == 2:
            image = np.stack([image] * 3, axis=-1)
        
        # Resize to 224x224 (required by ResNet/VGG)
        img_pil = PILImage.fromarray(image.astype('uint8'))
        img_resized = img_pil.resize((224, 224), PILImage.LANCZOS)
        
        # Convert back to array and preprocess
        img_array = np.array(img_resized)
        img_array = np.expand_dims(img_array, axis=0)
        img_preprocessed = preprocess_input(img_array.astype('float32'))
        
        return img_preprocessed
    
    def extract_features(self, images):
        """
        Extract features from batch of images
        
        Args:
            images: NumPy array of images (N x 28 x 28) or (N x 28 x 28 x 3)
            
        Returns:
            Feature vectors (N x feature_dim)
        """
        if self.model is None:
            self.build_model()
        
        print(f"Extracting features from {len(images)} images...")
        
        # Preprocess all images
        preprocessed = []
        for img in images:
            prep_img = self.preprocess_image(img)
            preprocessed.append(prep_img)
        
        # Stack into batch
        batch = np.vstack(preprocessed)
        
        # Extract features
        features = self.model.predict(batch, verbose=0)
        
        # Normalize features (L2 normalization)
        features = self._normalize_features(features)
        
        print(f"Extracted features shape: {features.shape}")
        return features
    
    def extract_single_feature(self, image, image_id=None):
        """
        Extract features from single image
        
        Args:
            image: Single image (28x28 or 28x28x3)
            image_id: Optional ID for caching
            
        Returns:
            Feature vector (feature_dim)
        """
        # Check cache
        if image_id is not None and image_id in self.feature_cache:
            return self.feature_cache[image_id]
        
        if self.model is None:
            self.build_model()
        
        # Preprocess and extract
        prep_img = self.preprocess_image(image)
        feature = self.model.predict(prep_img, verbose=0)
        feature = self._normalize_features(feature)[0]
        
        # Cache result
        if image_id is not None:
            self.feature_cache[image_id] = feature
        
        return feature
    
    def _normalize_features(self, features):
        """L2 normalize feature vectors"""
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)  # Avoid division by zero
        return features / norms
    
    def get_feature_dim(self):
        """Get dimension of extracted features"""
        if self.model is None:
            self.build_model()
        
        # ResNet50 outputs 2048-dim features
        if self.model_name == 'resnet50':
            return 2048
        # elif self.model_name == 'vgg16':
        #     return 512
        
        return None
    
    def save_features(self, features, filepath):
        """Save extracted features to file"""
        np.save(filepath, features)
        print(f"Saved features to {filepath}")
    
    def load_features(self, filepath):
        """Load pre-extracted features from file"""
        features = np.load(filepath)
        print(f"Loaded features from {filepath}: shape {features.shape}")
        return features


# class SimpleFeatureExtractor:
#     """Simple feature extraction without deep learning (fallback)"""
    
#     def extract_features(self, images):
#         """Extract simple features (flatten + normalize)"""
#         features = images.reshape(len(images), -1).astype('float32') / 255.0
#         # L2 normalize
#         norms = np.linalg.norm(features, axis=1, keepdims=True)
#         return features / norms


if __name__ == "__main__":
    # Load small sample
    dataset = FashionMNISTDataset(subset_size=10)
    images, labels = dataset.load_data()
    
    # Test CNN feature extraction
    extractor = FeatureExtractor('resnet50')
    features = extractor.extract_features(images)
    
    print(f"\nFeature extraction results:")
    print(f"Input images shape: {images.shape}")
    print(f"Output features shape: {features.shape}")
    print(f"Feature vector for image 0: {features[0][:10]}...")  # First 10 dims
    