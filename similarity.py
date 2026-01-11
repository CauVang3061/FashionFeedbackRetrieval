"""
Similarity metrics for image retrieval
"""

import numpy as np

class SimilarityCalculator:
    def calculate_similarity(self, query_features, database_features):
        if query_features.ndim == 1:
            query_features = query_features.reshape(1, -1)
        
        similarities = np.dot(database_features, query_features.T).flatten()
        
        return similarities
    
    def rank_results(self, similarities, top_k=None):
        sorted_indices = np.argsort(similarities)[::-1]
        sorted_scores = similarities[sorted_indices]
        
        if top_k is not None:
            sorted_indices = sorted_indices[:top_k]
            sorted_scores = sorted_scores[:top_k]
        
        return sorted_indices, sorted_scores
    
    def calculate_similarity_matrix(self, features):
        return np.dot(features, features.T)


class TextToImageSimilarity:
    def __init__(self, class_names):
        self.class_names = [str(name).lower() for name in class_names]
    
    def text_to_feature(self, text_query):
        text_lower = text_query.lower()
        matches = []
        
        for class_name in self.class_names:
            if class_name == text_lower:
                matches.append(1.0)
            elif class_name in text_lower or text_lower in class_name:
                matches.append(0.8)
            else:
                matches.append(0.0)
        
        matches = np.array(matches)
        if matches.sum() > 0:
            matches = matches / matches.sum()
            
        return matches
    
    def get_relevant_classes(self, text_query):
        feature = self.text_to_feature(text_query)
        relevant_indices = np.where(feature > 0)[0]
        
        return [self.class_names[i] for i in relevant_indices]
