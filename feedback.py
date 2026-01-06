"""
Relevance Feedback implementation using Rocchio algorithm: to refine search query based on user-marked relevant/irrelevant images
"""

import numpy as np

class RocchioFeedback:
    
    def __init__(self, alpha=1.0, beta=0.75, gamma=0.25):
        """
        Args:
            alpha: Weight for original query (default: 1.0)
            beta: Weight for relevant documents (default: 0.75)
            gamma: Weight for irrelevant documents (default: 0.25)
        """
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        
        # History tracking for analysis
        self.feedback_history = []
        self.query_history = []
    
    def update_query(self, original_query, relevant_features, 
                    irrelevant_features=None):
        """
        Args:
            original_query: Original query feature vector (2048,)
            relevant_features: Feature vectors of relevant images (N_rel x 2048)
            irrelevant_features: Feature vectors of irrelevant images (N_irr x 2048)
        Returns:
            Updated query vector (2048,) - L2 normalized
        """
        # Weighted original query
        new_query = self.alpha * original_query
        
        # Add contribution from relevant documents
        if relevant_features is not None and len(relevant_features) > 0:
            relevant_mean = np.mean(relevant_features, axis=0)
            new_query += self.beta * relevant_mean
            
        # Subtract contribution from irrelevant documents
        if irrelevant_features is not None and len(irrelevant_features) > 0:
            irrelevant_mean = np.mean(irrelevant_features, axis=0)
            new_query -= self.gamma * irrelevant_mean
        
        # L2 normalize the new query
        new_query = self._normalize_vector(new_query)
        
        # Store in history for tracking
        self.query_history.append(new_query.copy())
        self.feedback_history.append({
            'n_relevant': len(relevant_features) if relevant_features is not None else 0,
            'n_irrelevant': len(irrelevant_features) if irrelevant_features is not None else 0
        })
        
        return new_query
    
    def _normalize_vector(self, vector):
        """L2 normalize vector to unit length"""
        norm = np.linalg.norm(vector)
        if norm > 0:
            return vector / norm
        return vector
    
    def iterative_feedback(self, original_query, feedback_rounds):
        """
        Apply multiple rounds of feedback iteratively
        Args:
            original_query: Initial query vector
            feedback_rounds: List of (relevant_features, irrelevant_features) tuples
        Returns:
            Final query vector after all feedback rounds
        """
        current_query = original_query.copy()
        
        for relevant_feats, irrelevant_feats in feedback_rounds:
            current_query = self.update_query(
                current_query, 
                relevant_feats, 
                irrelevant_feats
            )
        
        return current_query
    
    def reset_history(self):
        """Clear all feedback history"""
        self.feedback_history = []
        self.query_history = []
    
    def get_feedback_stats(self):
        """
        Get statistics about feedback sessions
        Returns:
            Dictionary with feedback statistics or None if no history
        """
        if not self.feedback_history:
            return None
        
        total_relevant = sum(fb['n_relevant'] for fb in self.feedback_history)
        total_irrelevant = sum(fb['n_irrelevant'] for fb in self.feedback_history)
        
        return {
            'num_rounds': len(self.feedback_history),
            'total_relevant': total_relevant,
            'total_irrelevant': total_irrelevant,
            'avg_relevant_per_round': total_relevant / len(self.feedback_history),
            'avg_irrelevant_per_round': total_irrelevant / len(self.feedback_history)
        }
    
    def get_query_drift(self):
        """
        Calculate how much query has drifted from original
        Measures the cosine similarity between original and current query.
        A value close to 1.0 means queries are similar (little drift).
        A value close to 0.0 means queries are very different (significant drift).
        Returns:
            Cosine similarity between original and current query (0.0 to 1.0)
        """
        if len(self.query_history) < 2:
            return 1.0
        
        original = self.query_history[0]
        current = self.query_history[-1]
        
        # Cosine similarity (dot product of normalized vectors)
        similarity = np.dot(original, current)
        return similarity


class InteractiveFeedbackSession:
    """
    Manages an interactive feedback session with user
    Keeps track of:
    - Original query
    - Current query (after feedback)
    - Relevant/irrelevant image indices marked by user
    - Number of feedback iterations
    """
    
    def __init__(self, rocchio_feedback):
        self.rocchio = rocchio_feedback
        self.original_query = None
        self.current_query = None
        self.relevant_indices = []
        self.irrelevant_indices = []
        self.iteration = 0
    
    def start_session(self, query_vector):
        """
        Start a new feedback session
        Args:
            query_vector: Initial query feature vector
        """
        self.original_query = query_vector.copy()
        self.current_query = query_vector.copy()
        self.relevant_indices = []
        self.irrelevant_indices = []
        self.iteration = 0
        self.rocchio.reset_history()
    
    def add_feedback(self, relevant_ids=None, irrelevant_ids=None):
        """
        Add user feedback (relevant/irrelevant image indices)
        Args:
            relevant_ids: List of relevant image indices
            irrelevant_ids: List of irrelevant image indices
        """
        if relevant_ids:
            self.relevant_indices.extend(relevant_ids)
        if irrelevant_ids:
            self.irrelevant_indices.extend(irrelevant_ids)
    
    def apply_feedback(self, all_features):
        """
        Apply feedback and update query using Rocchio algorithm
        Args:
            all_features: All feature vectors in database (N x 2048)
        Returns:
            Updated query vector (2048,)
        """
        # Extract features for relevant images
        relevant_feats = None
        if self.relevant_indices:
            relevant_feats = all_features[self.relevant_indices]
        
        # Extract features for irrelevant images
        irrelevant_feats = None
        if self.irrelevant_indices:
            irrelevant_feats = all_features[self.irrelevant_indices]
        
        # Update query using Rocchio algorithm
        self.current_query = self.rocchio.update_query(
            self.original_query,
            relevant_feats,
            irrelevant_feats
        )
        
        self.iteration += 1
        return self.current_query
    
    def get_session_info(self):
        """
        Get current session information
        Returns:
            Dictionary with session statistics
        """
        return {
            'iteration': self.iteration,
            'n_relevant': len(self.relevant_indices),
            'n_irrelevant': len(self.irrelevant_indices),
            'query_drift': self.rocchio.get_query_drift()
        }


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Rocchio Relevance Feedback Algorithm")
    print("=" * 60)
    
    np.random.seed(42)
    
    # Create sample data (simulating ResNet50 features)
    feature_dim = 2048
    
    # Original query
    original_query = np.random.randn(feature_dim)
    original_query = original_query / np.linalg.norm(original_query)
    
    # Simulate 5 relevant images
    relevant_features = np.random.randn(5, feature_dim)
    relevant_features = relevant_features / np.linalg.norm(relevant_features, axis=1, keepdims=True)
    
    # Simulate 3 irrelevant images
    irrelevant_features = np.random.randn(3, feature_dim)
    irrelevant_features = irrelevant_features / np.linalg.norm(irrelevant_features, axis=1, keepdims=True)
    
    # Initialize Rocchio with standard parameters
    rocchio = RocchioFeedback(alpha=1.0, beta=0.75, gamma=0.25)
    
    print("\nOriginal query:")
    print(f"  Norm: {np.linalg.norm(original_query):.4f} (should be 1.0)")
    
    # Apply feedback
    new_query = rocchio.update_query(original_query, relevant_features, irrelevant_features)
    
    print("\nAfter Rocchio feedback:")
    print(f"  Updated query norm: {np.linalg.norm(new_query):.4f} (should be 1.0)")
    
    # Calculate similarity between original and new query
    similarity = np.dot(original_query, new_query)
    print(f"  Similarity to original: {similarity:.4f}")
    print(f"  Query drift: {1 - similarity:.4f}")
    
    # Get feedback statistics
    stats = rocchio.get_feedback_stats()
    print("\nFeedback statistics:")
    print(f"  Number of rounds: {stats['num_rounds']}")
    print(f"  Total relevant images: {stats['total_relevant']}")
    print(f"  Total irrelevant images: {stats['total_irrelevant']}")
    
    print("\n" + "=" * 60)
    print("Test completed successfully!")
    print("=" * 60)
