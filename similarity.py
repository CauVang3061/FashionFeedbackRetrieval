"""
Similarity metrics for image retrieval: implement cosine similarity for comparing feature vectors
"""

import numpy as np

class SimilarityCalculator:
    """
    Calculate cosine similarity between feature vectors
    For L2-normalized vectors, it equals their dot product.
    Range: [-1, 1], where 1 = identical, 0 = orthogonal, -1 = opposite
    """
    
    def calculate_similarity(self, query_features, database_features):
        """
        Calculate cosine similarity between query and database
        Args:
            query_features: Query feature vector (feature_dim,)
            database_features: Database feature matrix (N x feature_dim)
        Returns:
            Similarity scores (N,) - higher values indicate more similar images
        """
        # Ensure query is 2D for matrix multiplication
        if query_features.ndim == 1:
            query_features = query_features.reshape(1, -1)
        
        # Compute dot product (equivalent to cosine similarity for normalized vectors)
        similarities = np.dot(database_features, query_features.T).flatten()
        
        return similarities
    
    def rank_results(self, similarities, top_k=None):
        """
        Rank results by similarity score
        Args:
            similarities: Similarity scores (N,)
            top_k: Return only top k results (default: all)
        Returns:
            indices: Indices sorted by similarity (descending order)
            scores: Corresponding similarity scores
        """
        # Sort indices by similarity (descending: highest similarity first)
        sorted_indices = np.argsort(similarities)[::-1]
        sorted_scores = similarities[sorted_indices]
        
        # Return top k if specified
        if top_k is not None:
            sorted_indices = sorted_indices[:top_k]
            sorted_scores = sorted_scores[:top_k]
        
        return sorted_indices, sorted_scores
    
    def calculate_similarity_matrix(self, features):
        """
        Calculate pairwise similarity matrix
        
        Args:
            features: Feature matrix (N x feature_dim)
            
        Returns:
            Similarity matrix (N x N) where entry (i,j) is similarity between i and j
        """
        # Cosine similarity matrix (dot product for normalized vectors)
        similarity_matrix = np.dot(features, features.T)
        return similarity_matrix
    
    def get_similar_pairs(self, features, threshold=0.9):
        """
        Find pairs of similar images above threshold
        Args:
            features: Feature matrix (N x feature_dim)
            threshold: Similarity threshold (0.0 to 1.0)
        Returns:
            List of (i, j, similarity) tuples for pairs above threshold
        """
        n = len(features)
        pairs = []
        
        for i in range(n):
            similarities = self.calculate_similarity(features[i], features)
            for j in range(i + 1, n):  # Only check upper triangle
                if similarities[j] >= threshold:
                    pairs.append((i, j, similarities[j]))
        
        return pairs


class TextToImageSimilarity:
    """
    Handle text query to image similarity (simplified keyword matching)
    
    Note: This is a simplified version using keyword matching.
    A real implementation would use a text encoder like CLIP.
    """
    
    def __init__(self, class_names):
        """
        Initialize with Fashion-MNIST class names
        Args:
            class_names: List of class names
        """
        self.class_names = [name.lower() for name in class_names]
    
    def text_to_feature(self, text_query):
        """
        Convert text query to pseudo-feature vector using keyword matching
        Args:
            text_query: Text query string (e.g., "dress", "black shoes")
        Returns:
            Pseudo-feature vector (num_classes,) representing class relevance
        """
        text_lower = text_query.lower()
        
        # Score each class based on keyword matching
        matches = []
        for class_name in self.class_names:
            # Exact match
            if class_name in text_lower:
                matches.append(1.0)
            else:
                # Partial match (for composite names like "T-shirt/top")
                words = class_name.split('/')
                score = max([0.5 if word in text_lower else 0.0 for word in words])
                matches.append(score)
        
        # Normalize to sum to 1
        matches = np.array(matches)
        if matches.sum() > 0:
            matches = matches / matches.sum()
        
        return matches
    
    def get_relevant_classes(self, text_query):
        """
        Get indices of relevant classes for text query
        Args:
            text_query: Text query string
        Returns:
            List of relevant class indices
        """
        feature = self.text_to_feature(text_query)
        relevant_indices = np.where(feature > 0)[0]
        return relevant_indices.tolist()


if __name__ == "__main__":
    # Test similarity calculations
    print("=" * 60)
    print("Testing Cosine Similarity Calculator")
    print("=" * 60)
    
    np.random.seed(42)
    
    # Create sample features (simulating ResNet50 output)
    n_samples = 100
    feature_dim = 2048
    features = np.random.randn(n_samples, feature_dim)
    
    # L2 normalize features (as ResNet50 does)
    features = features / np.linalg.norm(features, axis=1, keepdims=True)
    
    # Initialize calculator
    calc = SimilarityCalculator(metric='cosine')
    
    # Test 1: Self-similarity
    print("\nTest 1: Self-similarity")
    query = features[0]
    similarities = calc.calculate_similarity(query, features)
    print(f"Query vs itself: {similarities[0]:.6f} (should be ~1.0)")
    print(f"Query vs 5 others: {similarities[1:6]}")
    
    # Test 2: Ranking
    print("\nTest 2: Ranking results")
    top_indices, top_scores = calc.rank_results(similarities, top_k=5)
    print("Top 5 most similar images:")
    for rank, (idx, score) in enumerate(zip(top_indices, top_scores), 1):
        print(f"  {rank}. Index {idx}: similarity = {score:.4f}")
    
    # Test 3: Similarity matrix
    print("\nTest 3: Similarity matrix")
    sim_matrix = calc.calculate_similarity_matrix(features[:10])
    print(f"Similarity matrix shape: {sim_matrix.shape}")
    print(f"Diagonal (self-similarity): {np.diag(sim_matrix)}")
    
    # Test 4: Text-to-image matching
    print("\n" + "=" * 60)
    print("Testing Text-to-Image Similarity")
    print("=" * 60)
    
    class_names = [
        'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
        'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
    ]
    
    text_matcher = TextToImageSimilarity(class_names)
    
    test_queries = ["dress", "shoes", "t-shirt", "black trouser"]
    for query in test_queries:
        relevant_classes = text_matcher.get_relevant_classes(query)
        print(f"\nQuery: '{query}'")
        print(f"Relevant classes: {[class_names[i] for i in relevant_classes]}")
    
    print("\n" + "=" * 60)
    print("All tests completed successfully!")
    print("=" * 60)
