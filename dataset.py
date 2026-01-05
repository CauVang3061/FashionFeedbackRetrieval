"""
Dataset loader for Fashion-MNIST: 10 categories of clothing, 28x28 grayscale images.
Handles downloading, loading, and preprocessing of Fashion-MNIST dataset
"""

import numpy as np
from tensorflow import keras
from PIL import Image
import os

class FashionMNISTDataset:
    """Load and manage Fashion-MNIST dataset"""
    
    # Fashion-MNIST class labels
    CLASS_NAMES = [
        'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
        'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
    ]
    
    def __init__(self, subset_size=1000):
        """
        Initialize dataset loader
        Args:
            subset_size: Number of images to use from dataset (default: 1000)
        """
        self.subset_size = subset_size
        self.images = None
        self.labels = None
        self.image_ids = None
        
    def load_data(self):
        """Load Fashion-MNIST dataset from Keras"""
        print("Loading Fashion-MNIST dataset...")
        
        # Load from Keras datasets
        (train_images, train_labels), (test_images, test_labels) = \
            keras.datasets.fashion_mnist.load_data()
        
        # Combine train and test for larger pool
        all_images = np.concatenate([train_images, test_images], axis=0)
        all_labels = np.concatenate([train_labels, test_labels], axis=0)
        
        # Select subset
        indices = np.random.choice(len(all_images), self.subset_size, replace=False)
        self.images = all_images[indices]
        self.labels = all_labels[indices]
        self.image_ids = indices
        
        print(f"Loaded {len(self.images)} images")
        print(f"Image shape: {self.images[0].shape}")
        
        return self.images, self.labels
    
    def get_image(self, idx):
        """
        Get single image by index
        Args:
            idx: Image index
            
        Returns:
            PIL Image object
        """
        if self.images is None:
            raise ValueError("Dataset not loaded. Call load_data() first.")
        
        img_array = self.images[idx]
        # Convert to RGB for display (Fashion-MNIST is grayscale)
        img_rgb = np.stack([img_array] * 3, axis=-1)
        return Image.fromarray(img_rgb.astype('uint8'))
    
    def get_label(self, idx):
        """Get label name for image index"""
        if self.labels is None:
            raise ValueError("Dataset not loaded. Call load_data() first.")
        
        label_idx = self.labels[idx]
        return self.CLASS_NAMES[label_idx]
    
    def preprocess_cnn(self, images=None):
        """
        Preprocess images for CNN feature extraction
        
        Args:
            images: Images to preprocess (default: all dataset images)
            
        Returns:
            Preprocessed images ready for CNN (normalized, RGB format)
        """
        if images is None:
            images = self.images
        
        # Convert grayscale to RGB (28x28 -> 28x28x3)
        images_rgb = np.stack([images] * 3, axis=-1)
        
        # Normalize to [0, 1]
        images_normalized = images_rgb.astype('float32') / 255.0
        
        return images_normalized
    
    def get_random_sample(self, n=10):
        """Get n random images for testing"""
        if self.images is None:
            raise ValueError("Dataset not loaded. Call load_data() first.")
        
        indices = np.random.choice(len(self.images), n, replace=False)
        return indices
    
    def save_image(self, idx, filepath):
        """Save image to file"""
        img = self.get_image(idx)
        img.save(filepath)
        print(f"Saved image {idx} to {filepath}")


if __name__ == "__main__":
    # Test dataset loading
    dataset = FashionMNISTDataset(subset_size=100)
    images, labels = dataset.load_data()
    
    print(f"\nDataset info:")
    print(f"Total images: {len(images)}")
    print(f"Image shape: {images[0].shape}")
    print(f"Labels: {np.unique(labels)}")
    
    # Test getting random samples
    samples = dataset.get_random_sample(5)
    print(f"\nRandom samples: {samples}")
    for idx in samples:
        print(f"Image {idx}: {dataset.get_label(idx)}")
