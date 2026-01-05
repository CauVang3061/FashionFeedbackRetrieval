"""
Similarity metrics for image retrieval
Implements various distance/similarity measures
"""

import numpy as np
from scipy.spatial.distance import cosine, cdist
# from scipy.spatial.distance import euclidean

class SimilarityCalculator:
    """Calculate similarity between feature vectors"""
    
    def __init__(self, metric='cosine'):
        """
        Initialize similarity calculator
        
        Args:
            metric: Similarity metric ('cosine', 'euclidean', 'manhattan')
        """
        self.metric = metric
    
    def calculate_similarity(self, query_features, database_features):
        """
        Calculate similarity between query and database
        
        Args:
            query_features: Query feature vector (feature_dim,)
            database_features: Database feature matrix (N x feature_dim)
            
        Returns:
            Similarity scores (N) - higher is more similar
        """
        if self.metric == 'cosine':
            return self._cosine_similarity(query_features, database_features)
        # elif self.metric == 'euclidean':
        #     return self._euclidean_similarity(query_features, database_features)
        # elif self.metric == 'manhattan':
        #     return self._manhattan_similarity(query_features, database_features)
        else:
            raise ValueError(f"Unknown metric: {self.metric}")
    
    def _cosine_similarity(self, query, database):
        """
        Cosine similarity (dot product for normalized vectors)
        Returns values in [-1, 1], higher is more similar
        """
        # Ensure query is 2D
        if query.ndim == 1:
            query = query.reshape(1, -1)
        
        # Compute dot product (vectors should be L2 normalized)
        similarities = np.dot(database, query.T).flatten()
        
        return similarities
    
    # def _euclidean_similarity(self, query, database):
    #     """
    #     Euclidean distance converted to similarity
    #     Returns similarity scores (higher is more similar)
    #     """
    #     # Calculate distances
    #     distances = np.linalg.norm(database - query, axis=1)
        
    #     # Convert to similarity (inverse of distance)
    #     # Add small epsilon to avoid division by zero
    #     similarities = 1.0 / (distances + 1e-8)
        
    #     return similarities
    
    # def _manhattan_similarity(self, query, database):
    #     """
    #     Manhattan (L1) distance converted to similarity
    #     """
    #     distances = np.sum(np.abs(database - query), axis=1)
    #     similarities = 1.0 / (distances + 1e-8)
    #     return similarities
    
    def rank_results(self, similarities, top_k=None):
        """
        Rank results by similarity
        
        Args:
            similarities: Similarity scores (N)
            top_k: Return only top k results (default: all)
            
        Returns:
            indices: Indices sorted by similarity (descending)
            scores: Corresponding similarity scores
        """
        # Sort indices by similarity (descending)
        sorted_indices = np.argsort(similarities)[::-1]
        sorted_scores = similarities[sorted_indices]
        
        # Return top k if specified
        if top_k is not None:
            sorted_indices = sorted_indices[:top_k]
            sorted_scores = sorted_scores[:top_k]
        
        return sorted_indices, sorted_scores
    
    def calculate_distance_matrix(self, features):
        """
        Calculate pairwise distance matrix
        
        Args:
            features: Feature matrix (N x feature_dim)
            
        Returns:
            Distance matrix (N x N)
        """
        if self.metric == 'cosine':
            # Cosine distance = 1 - cosine similarity
            similarity_matrix = np.dot(features, features.T)
            distance_matrix = 1 - similarity_matrix
        # elif self.metric == 'euclidean':
        #     from scipy.spatial.distance import cdist
        #     distance_matrix = cdist(features, features, metric='euclidean')
        # else:
        #     from scipy.spatial.distance import cdist
        #     distance_matrix = cdist(features, features, metric=self.metric)
        
        return distance_matrix
    
    def get_similar_pairs(self, features, threshold=0.9):
        """
        Find pairs of similar images
        
        Args:
            features: Feature matrix (N x feature_dim)
            threshold: Similarity threshold
            
        Returns:
            List of (i, j, similarity) tuples
        """
        n = len(features)
        pairs = []
        
        for i in range(n):
            similarities = self.calculate_similarity(features[i], features)
            for j in range(i + 1, n):
                if similarities[j] >= threshold:
                    pairs.append((i, j, similarities[j]))
        
        return pairs


class TextToImageSimilarity:
    """Handle text query to image similarity (simplified)"""
    
    def __init__(self, class_names):
        """
        Initialize with class names for Fashion-MNIST
        
        Args:
            class_names: List of class names
        """
        self.class_names = [name.lower() for name in class_names]
    
    def text_to_feature(self, text_query):
        """
        Convert text query to pseudo-feature vector
        This is a simplified version - in real CLIP, this would use
        a text encoder. Here we use keyword matching.
        
        Args:
            text_query: Text query string
            
        Returns:
            Pseudo-feature vector
        """
        text_lower = text_query.lower()
        
        # Create a simple embedding based on keyword matching
        # In reality, you'd use CLIP or similar model
        matches = []
        for class_name in self.class_names:
            # Check if class name appears in query
            if class_name in text_lower:
                matches.append(1.0)
            else:
                # Partial match score
                words = class_name.split('/')
                score = max([0.5 if word in text_lower else 0.0 for word in words])
                matches.append(score)
        
        # Normalize
        matches = np.array(matches)
        if matches.sum() > 0:
            matches = matches / matches.sum()
        
        return matches
    
    def get_relevant_classes(self, text_query):
        """Get most relevant class indices for text query"""
        feature = self.text_to_feature(text_query)
        relevant_indices = np.where(feature > 0)[0]
        return relevant_indices.tolist()


if __name__ == "__main__":
    # Test similarity calculations
    np.random.seed(42)
    
    # Create sample features
    n_samples = 100
    feature_dim = 2048
    features = np.random.randn(n_samples, feature_dim)
    
    # Normalize features
    features = features / np.linalg.norm(features, axis=1, keepdims=True)
    
    # Test similarity calculation
    calc = SimilarityCalculator(metric='cosine')
    query = features[0]
    similarities = calc.calculate_similarity(query, features)
    
    print("Similarity calculation test:")
    print(f"Query vs itself: {similarities[0]:.4f} (should be ~1.0)")
    print(f"Query vs others: {similarities[1:5]}")
    
    # Test ranking
    top_indices, top_scores = calc.rank_results(similarities, top_k=5)
    print(f"\nTop 5 most similar:")
    for idx, score in zip(top_indices, top_scores):
        print(f"  Index {idx}: {score:.4f}")
