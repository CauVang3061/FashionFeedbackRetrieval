"""
Deep feature extraction from FashionMNIST images using pre-trained ResNet50 model
"""

import numpy as np
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from PIL import Image as PILImage
from dataset import FashionMNISTDataset

class FeatureExtractor:
    """Extract deep features using pre-trained ResNet50"""
    
    def __init__(self):
        self.model = None
        self.feature_cache = {}
        self.feature_dim = 2048  # ResNet50 output dimension
        
    def build_model(self):
        """Build ResNet50 feature extraction model"""
        print("Loading pre-trained ResNet50 model...")
        
        # Load ResNet50 pre-trained on ImageNet
        # include_top=False: remove classification layer
        # pooling='avg': global average pooling to get fixed-size features
        self.model = ResNet50(
            weights='imagenet', 
            include_top=False, 
            pooling='avg'
        )
        
        print("ResNet50 model loaded successfully!")
        return self.model
    
    def preprocess_image(self, image):
        """
        Preprocess single image for ResNet50
        Args:
            image: NumPy array (28x28 or 28x28x3)
        Returns:
            Preprocessed image (224x224x3) ready for ResNet50
        """
        # Convert grayscale to RGB if needed
        if len(image.shape) == 2:
            image = np.stack([image] * 3, axis=-1)
        
        # Resize from 28x28 to 224x224 (ResNet50 input size)
        img_pil = PILImage.fromarray(image.astype('uint8'))
        img_resized = img_pil.resize((224, 224), PILImage.LANCZOS)
        
        # Convert to array and add batch dimension
        img_array = np.array(img_resized)
        img_array = np.expand_dims(img_array, axis=0)
        
        # Apply ResNet50 preprocessing (normalizes to [-1, 1])
        img_preprocessed = preprocess_input(img_array.astype('float32'))
        
        return img_preprocessed
    
    def extract_features(self, images):
        """
        Extract features from batch of images
        Args:
            images: NumPy array of images (N x 28 x 28) or (N x 28 x 28 x 3)
        Returns:
            Feature vectors (N x 2048) - L2 normalized
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
        
        # Extract features using ResNet50
        features = self.model.predict(batch, verbose=0)
        
        # L2 normalize features for cosine similarity
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
            Feature vector (2048,) - L2 normalized
        """
        # Check cache first
        if image_id is not None and image_id in self.feature_cache:
            return self.feature_cache[image_id]
        
        if self.model is None:
            self.build_model()
        
        # Preprocess and extract
        prep_img = self.preprocess_image(image)
        feature = self.model.predict(prep_img, verbose=0)
        feature = self._normalize_features(feature)[0]
        
        # Cache result for future use
        if image_id is not None:
            self.feature_cache[image_id] = feature
        
        return feature
    
    def _normalize_features(self, features):
        """
        L2 normalize feature vectors to ensure all have unit length, making cosine similarity equivalent to dot product
        """
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)  # Avoid division by zero
        return features / norms
    
    def get_feature_dim(self):
        """Get dimension of extracted features (2048 for ResNet50)"""
        return self.feature_dim
    
    def save_features(self, features, filepath):
        """Save extracted features to .npy file"""
        np.save(filepath, features)
        print(f"Saved features to {filepath}")
    
    def load_features(self, filepath):
        """Load pre-extracted features from .npy file"""
        features = np.load(filepath)
        print(f"Loaded features from {filepath}: shape {features.shape}")
        return features


if __name__ == "__main__":
    # Load small sample
    dataset = FashionMNISTDataset(subset_size=10)
    images, labels = dataset.load_data()
    
    # Test ResNet50 feature extraction
    extractor = FeatureExtractor()
    features = extractor.extract_features(images)
    
    print(f"\n{'='*50}")
    print("Feature Extraction Test Results:")
    print(f"{'='*50}")
    print(f"Input images shape: {images.shape}")
    print(f"Output features shape: {features.shape}")
    print(f"Feature dimension: {extractor.get_feature_dim()}")
    print(f"\nFirst 10 dimensions of feature vector 0:")
    print(features[0][:10])
    print(f"\nFeature vector norm (should be ~1.0): {np.linalg.norm(features[0]):.6f}")
