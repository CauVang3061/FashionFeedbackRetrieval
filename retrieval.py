"""
Main image retrieval engine
Coordinates: dataset, features, similarity, feedback
"""

import numpy as np
import os
from dataset import FashionProductDataset
from features_extract import FeatureExtractor
from similarity import SimilarityCalculator, TextToImageSimilarity
from feedback import RocchioFeedback, InteractiveFeedbackSession

class ImageRetrievalSystem:
    def __init__(self, dataset_limit, batch_size=32):
        self.dataset_limit = dataset_limit
        self.dataset = FashionProductDataset(subset_size=dataset_limit)
        self.feature_extractor = FeatureExtractor(batch_size=batch_size)
        self.similarity_calculator = SimilarityCalculator()
        self.rocchio = RocchioFeedback(alpha=1.0, beta=0.75, gamma=0.25)
        self.text_matcher = None
        
        # Data storage
        self.image_ids = None
        self.labels = None
        self.features = None
        self.feature_cache_path = f'features_cache_{dataset_limit}.npy'
        
        # Session management
        self.current_session = None
        self.current_query_vector = None
        self.last_results = None
    
    def initialize(self, force_recompute=False):
        print("=" * 60)
        print("Initializing Color Image Retrieval System")
        print("=" * 60)
        
        # Load metadata
        self.image_ids, self.labels = self.dataset.load_data()
        
        unique_categories = list(np.unique(self.labels))
        self.text_matcher = TextToImageSimilarity(unique_categories)
        
        if os.path.exists(self.feature_cache_path) and not force_recompute:
            print(f"\nLoading cached features from {self.feature_cache_path}...")
            self.features = np.load(self.feature_cache_path)
        else:
            print("\nExtracting features using ResNet50 (Batch mode)...")
            self.features = self.feature_extractor.extract_features(self.dataset)

            np.save(self.feature_cache_path, self.features)
            print(f"Cached features to {self.feature_cache_path}")
        
        print("\n" + "=" * 60)
        print("System initialized successfully!")
        print(f"Dataset size: {len(self.image_ids)}")
        print(f"Feature dimension: {self.features.shape[1]}")
        print("=" * 60)

    def search_by_text(self, text_query, top_k=20):
        relevant_classes = self.text_matcher.get_relevant_classes(text_query)
        
        if not relevant_classes:
            print(f"Warning: No matching classes found for '{text_query}'")
            query_features = np.mean(self.features, axis=0)
        else:
            relevant_mask = np.isin(self.labels, relevant_classes)
            relevant_indices = np.where(relevant_mask)[0]
            
            if len(relevant_indices) == 0:
                return [], []
            
            relevant_features = self.features[relevant_indices]
            query_features = np.mean(relevant_features, axis=0)
        
        query_features = query_features / (np.linalg.norm(query_features) + 1e-10)
        self.current_query_vector = query_features
        
        return self._execute_search(query_features, top_k)

    def search_by_uploaded_image(self, image_input, top_k=20):
        query_features = self.feature_extractor.extract_single_feature(image_input)
        self.current_query_vector = query_features
        
        return self._execute_search(query_features, top_k)

    def _execute_search(self, query_features, top_k):
        similarities = self.similarity_calculator.calculate_similarity(
            query_features, self.features
        )
        indices, scores = self.similarity_calculator.rank_results(
            similarities, top_k=top_k
        )
        self.last_results = (indices, scores)
        self.current_session = InteractiveFeedbackSession(self.rocchio)
        self.current_session.start_session(query_features)
        
        return indices, scores

    def apply_relevance_feedback(self, relevant_indices, irrelevant_indices, top_k=20):
        if self.current_session is None:
            raise ValueError("No active search session.")
        
        self.current_session.add_feedback(relevant_indices, irrelevant_indices)
        updated_query = self.current_session.apply_feedback(self.features)
        self.current_query_vector = updated_query
        
        similarities = self.similarity_calculator.calculate_similarity(
            updated_query, self.features
        )
        indices, scores = self.similarity_calculator.rank_results(
            similarities, top_k=top_k
        )
        
        self.last_results = (indices, scores)
        return indices, scores

    def get_image(self, idx):
        return self.dataset.get_image(idx)
    
    def get_label(self, idx):
        return self.dataset.get_label(idx)

    def get_session_stats(self):
        return self.current_session.get_session_info() if self.current_session else None
