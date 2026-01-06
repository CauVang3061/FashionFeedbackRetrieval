"""
Main image retrieval engine
Coordinates all components: dataset, features, similarity, feedback
"""

import numpy as np
from dataset import FashionMNISTDataset
from features_extract import FeatureExtractor
from similarity import SimilarityCalculator, TextToImageSimilarity
from feedback import RocchioFeedback, InteractiveFeedbackSession
import pickle
import os

class ImageRetrievalSystem:

    def __init__(self, dataset_limit):
        """
        Initialize retrieval engine
        Args:
            dataset_size: Number of images to load from Fashion-MNIST
        """
        self.dataset_limit = dataset_limit
        self.dataset = FashionMNISTDataset(subset_size=dataset_limit)
        self.feature_extractor = FeatureExtractor()
        self.similarity_calculator = SimilarityCalculator()
        self.rocchio = RocchioFeedback(alpha=1.0, beta=0.75, gamma=0.25)
        self.text_matcher = None
        
        # Data storage
        self.images = None
        self.labels = None
        self.features = None
        self.feature_cache_path = f'features_cache_{dataset_limit}.npy'
        
        # Session management
        self.current_session = None
        self.current_query_vector = None
        self.last_results = None
    
    def initialize(self, force_recompute=False):
        """
        Initialize system: load dataset and extract features
        Args:
            force_recompute: Force recomputation of features even if cached
        """
        print("=" * 60)
        print("Initializing Image Retrieval System")
        print("=" * 60)
        
        # Load dataset
        self.images, self.labels = self.dataset.load_data()
        
        # Initialize text matcher
        self.text_matcher = TextToImageSimilarity(self.dataset.CLASS_NAMES)
        
        # Load or compute features
        if os.path.exists(self.feature_cache_path) and not force_recompute:
            print(f"\nLoading cached features from {self.feature_cache_path}...")
            self.features = np.load(self.feature_cache_path)
            print(f"Loaded features shape: {self.features.shape}")
        else:
            print("\nExtracting features (this may take a few minutes)...")
            preprocessed = self.dataset.preprocess_cnn()
            self.features = self.feature_extractor.extract_features(preprocessed)
            
            # Cache features for future use
            np.save(self.feature_cache_path, self.features)
            print(f"Cached features to {self.feature_cache_path}")
        
        print("\n" + "=" * 60)
        print("System initialized successfully!")
        print(f"Dataset size: {len(self.images)}")
        print(f"Feature dimension: {self.features.shape[1]}")
        print("=" * 60)
    
    def search_by_text(self, text_query, top_k=20):
        """
        Search using text query
        Args:
            text_query: Text description (e.g., "dress", "shoes")
            top_k: Number of results to return
        Returns:
            result_indices, similarity_scores
        """
        # Get relevant classes from text query
        relevant_classes = self.text_matcher.get_relevant_classes(text_query)
        
        if not relevant_classes:
            print(f"Warning: No matching classes found for '{text_query}'")
            # Fallback: use centroid of all images
            query_features = np.mean(self.features, axis=0)
            query_features = query_features / np.linalg.norm(query_features)
        else:
            # Get features of images in relevant classes
            relevant_mask = np.isin(self.labels, relevant_classes)
            relevant_indices = np.where(relevant_mask)[0]
            
            if len(relevant_indices) == 0:
                print("No images found in relevant classes")
                return [], []
            
            # Use centroid of relevant class features as query
            relevant_features = self.features[relevant_indices]
            query_features = np.mean(relevant_features, axis=0)
            query_features = query_features / np.linalg.norm(query_features)
        
        self.current_query_vector = query_features
        
        # Calculate similarities
        similarities = self.similarity_calculator.calculate_similarity(
            query_features, self.features
        )
        
        # Rank results
        indices, scores = self.similarity_calculator.rank_results(
            similarities, top_k=top_k
        )
        
        self.last_results = (indices, scores)
        
        # Start new feedback session
        self.current_session = InteractiveFeedbackSession(self.rocchio)
        self.current_session.start_session(query_features)
        
        return indices, scores
    
    def search_by_uploaded_image(self, image_array, top_k=20):
        """
        Search using uploaded image
        Args:
            image_array: NumPy array of image (28x28 grayscale)
            top_k: Number of results to return
            
        Returns:
            result_indices, similarity_scores
        """
        # Extract features from uploaded image
        query_features = self.feature_extractor.extract_single_feature(image_array)
        self.current_query_vector = query_features
        
        # Calculate similarities
        similarities = self.similarity_calculator.calculate_similarity(
            query_features, self.features
        )
        
        # Rank results
        indices, scores = self.similarity_calculator.rank_results(
            similarities, top_k=top_k
        )
        
        self.last_results = (indices, scores)
        
        # Start new feedback session
        self.current_session = InteractiveFeedbackSession(self.rocchio)
        self.current_session.start_session(query_features)
        
        return indices, scores
    
    def apply_relevance_feedback(self, relevant_indices, irrelevant_indices, top_k=20):
        """
        Apply relevance feedback and re-rank results using Rocchio algorithm
        Args:
            relevant_indices: List of relevant image indices
            irrelevant_indices: List of irrelevant image indices
            top_k: Number of results to return
            
        Returns:
            new_result_indices, similarity_scores
        """
        if self.current_session is None:
            raise ValueError("No active search session. Perform a search first.")
        
        # Add feedback
        self.current_session.add_feedback(relevant_indices, irrelevant_indices)
        
        # Apply feedback to update query using Rocchio
        updated_query = self.current_session.apply_feedback(self.features)
        self.current_query_vector = updated_query
        
        # Re-calculate similarities with updated query
        similarities = self.similarity_calculator.calculate_similarity(
            updated_query, self.features
        )
        
        # Rank results
        indices, scores = self.similarity_calculator.rank_results(
            similarities, top_k=top_k
        )
        
        self.last_results = (indices, scores)
        
        # Print feedback info
        session_info = self.current_session.get_session_info()
        print(f"\nFeedback applied - Iteration {session_info['iteration']}")
        print(f"Relevant: {session_info['n_relevant']}, "
              f"Irrelevant: {session_info['n_irrelevant']}")
        print(f"Query drift: {session_info['query_drift']:.4f}")
        
        return indices, scores
    
    def get_image(self, idx):
        """Get image by index"""
        return self.dataset.get_image(idx)
    
    def get_label(self, idx):
        """Get label by index"""
        return self.dataset.get_label(idx)
    
    def get_session_stats(self):
        """Get current session statistics"""
        if self.current_session:
            return self.current_session.get_session_info()
        return None
    
    def save_system_state(self, filepath):
        """Save system state for later restoration"""
        state = {
            'features': self.features,
            'labels': self.labels,
            'dataset_size': len(self.images)
        }
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
        print(f"System state saved to {filepath}")


if __name__ == "__main__":
    # Test retrieval system
    print("=" * 70)
    print("Testing Image Retrieval System")
    print("=" * 70)
    
    # Initialize with small dataset for testing
    system = ImageRetrievalSystem(dataset_size=100)
    system.initialize()
    
    # Test 1: Image-based search (using dataset image)
    print("\n" + "=" * 70)
    print("Test 1: Image-based search (dataset image)")
    print("=" * 70)
    query_idx = 5
    indices, scores = system.search_by_image(query_idx, top_k=5)
    print(f"\nQuery image: {system.get_label(query_idx)}")
    print("Top 5 results:")
    for i, (idx, score) in enumerate(zip(indices, scores), 1):
        print(f"  {i}. Image {idx} ({system.get_label(idx)}): {score:.4f}")
    
    # Test 2: Text-based search
    print("\n" + "=" * 70)
    print("Test 2: Text-based search")
    print("=" * 70)
    text_query = "dress"
    indices, scores = system.search_by_text(text_query, top_k=5)
    print(f"\nText query: '{text_query}'")
    print("Top 5 results:")
    for i, (idx, score) in enumerate(zip(indices, scores), 1):
        print(f"  {i}. Image {idx} ({system.get_label(idx)}): {score:.4f}")
    
    # Test 3: Uploaded image search (simulated)
    print("\n" + "=" * 70)
    print("Test 3: Uploaded image search (simulated)")
    print("=" * 70)
    uploaded_img = system.images[10]  # Simulate uploaded image
    indices, scores = system.search_by_uploaded_image(uploaded_img, top_k=5)
    print(f"\nSimulated upload (actually image 10): {system.get_label(10)}")
    print("Top 5 results:")
    for i, (idx, score) in enumerate(zip(indices, scores), 1):
        print(f"  {i}. Image {idx} ({system.get_label(idx)}): {score:.4f}")
    
    # Test 4: Relevance feedback
    print("\n" + "=" * 70)
    print("Test 4: Relevance feedback (Rocchio)")
    print("=" * 70)
    relevant = [indices[0], indices[1]]  # First two are relevant
    irrelevant = [indices[4]]  # Last one is irrelevant
    print(f"Marking relevant: {relevant}")
    print(f"Marking irrelevant: {irrelevant}")
    
    new_indices, new_scores = system.apply_relevance_feedback(relevant, irrelevant, top_k=5)
    print("\nTop 5 results after feedback:")
    for i, (idx, score) in enumerate(zip(new_indices, new_scores), 1):
        print(f"  {i}. Image {idx} ({system.get_label(idx)}): {score:.4f}")
    
    # Show session stats
    print("\n" + "=" * 70)
    print("Session Statistics")
    print("=" * 70)
    stats = system.get_session_stats()
    if stats:
        print(f"Iteration: {stats['iteration']}")
        print(f"Total relevant marked: {stats['n_relevant']}")
        print(f"Total irrelevant marked: {stats['n_irrelevant']}")
        print(f"Query drift: {stats['query_drift']:.4f}")
    
    print("\n" + "=" * 70)
    print("All tests completed successfully!")
    print("=" * 70)
