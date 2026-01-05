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
    """Main retrieval system coordinating all components"""
    
    def __init__(self, dataset_size=1000, feature_model='resnet50'):
        """
        Initialize retrieval system
        
        Args:
            dataset_size: Number of images to load
            feature_model: Feature extraction model name
        """
        self.dataset = FashionMNISTDataset(subset_size=dataset_size)
        self.feature_extractor = FeatureExtractor(model_name=feature_model)
        self.similarity_calculator = SimilarityCalculator(metric='cosine')
        self.rocchio = RocchioFeedback(alpha=1.0, beta=0.75, gamma=0.25)
        self.text_matcher = None
        
        # Data storage
        self.images = None
        self.labels = None
        self.features = None
        self.feature_cache_path = f'features_cache_{dataset_size}.npy'
        
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
            print("\nExtracting features...")
            preprocessed = self.dataset.preprocess_for_cnn()
            self.features = self.feature_extractor.extract_features(preprocessed)
            
            # Cache features
            np.save(self.feature_cache_path, self.features)
            print(f"Cached features to {self.feature_cache_path}")
        
        print("\n" + "=" * 60)
        print("System initialized successfully!")
        print(f"Dataset size: {len(self.images)}")
        print(f"Feature dimension: {self.features.shape[1]}")
        print("=" * 60)
    
    # def search_by_image(self, query_image_idx, top_k=20):
    #     """
    #     Search using an image from the dataset
        
    #     Args:
    #         query_image_idx: Index of query image in dataset
    #         top_k: Number of results to return
            
    #     Returns:
    #         result_indices, similarity_scores
    #     """
    #     # Get query features
    #     query_features = self.features[query_image_idx]
    #     self.current_query_vector = query_features.copy()
        
    #     # Calculate similarities
    #     similarities = self.similarity_calculator.calculate_similarity(
    #         query_features, self.features
    #     )
        
    #     # Rank results
    #     indices, scores = self.similarity_calculator.rank_results(
    #         similarities, top_k=top_k
    #     )
        
    #     self.last_results = (indices, scores)
        
    #     # Start new feedback session
    #     self.current_session = InteractiveFeedbackSession(self.rocchio)
    #     self.current_session.start_session(query_features)
        
    #     return indices, scores
    
    def search_by_text(self, text_query, top_k=20):
        """
        Search using text query
        
        Args:
            text_query: Text description
            top_k: Number of results to return
            
        Returns:
            result_indices, similarity_scores
        """
        # Get relevant classes
        relevant_classes = self.text_matcher.get_relevant_classes(text_query)
        
        if not relevant_classes:
            print(f"Warning: No matching classes found for '{text_query}'")
            # Use first image as fallback
            return self.search_by_image(0, top_k)
        
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
    
    def search_by_image(self, image_array, top_k=20):
        """
        Search using uploaded image
        
        Args:
            image_array: NumPy array of image (28x28 or 28x28x3)
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
        Apply relevance feedback and re-rank results
        
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
        
        # Apply feedback to update query
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
        """Save system state (for later restoration)"""
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
    print("Testing Image Retrieval System\n")
    
    # Initialize with small dataset for testing
    system = ImageRetrievalSystem(dataset_size=100)
    system.initialize()
    
    # Test image-based search
    print("\n--- Test 1: Image-based search ---")
    query_idx = 5
    indices, scores = system.search_by_image(query_idx, top_k=5)
    print(f"Query image: {system.get_label(query_idx)}")
    print("Top 5 results:")
    for i, (idx, score) in enumerate(zip(indices, scores), 1):
        print(f"  {i}. Image {idx} ({system.get_label(idx)}): {score:.4f}")
    
    # Test text-based search
    print("\n--- Test 2: Text-based search ---")
    text_query = "dress"
    indices, scores = system.search_by_text(text_query, top_k=5)
    print(f"Text query: '{text_query}'")
    print("Top 5 results:")
    for i, (idx, score) in enumerate(zip(indices, scores), 1):
        print(f"  {i}. Image {idx} ({system.get_label(idx)}): {score:.4f}")
    
    # Test relevance feedback
    print("\n--- Test 3: Relevance feedback ---")
    relevant = [indices[0], indices[1]]  # First two are relevant
    irrelevant = [indices[4]]  # Last one is irrelevant
    new_indices, new_scores = system.apply_relevance_feedback(relevant, irrelevant, top_k=5)
    print("Top 5 results after feedback:")
    for i, (idx, score) in enumerate(zip(new_indices, new_scores), 1):
        print(f"  {i}. Image {idx} ({system.get_label(idx)}): {score:.4f}")
