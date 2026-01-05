"""
Relevance Feedback implementation using Rocchio algorithm
Refines query based on user feedback (relevant/irrelevant images)
"""

import numpy as np

class RocchioFeedback:
    """
    Rocchio algorithm for relevance feedback
    """
    
    def __init__(self, alpha=1.0, beta=0.75, gamma=0.25):
        """
        Initialize Rocchio feedback parameters
        
        Args:
            alpha: Weight for original query
            beta: Weight for relevant documents
            gamma: Weight for irrelevant documents
        """
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        
        # History tracking
        self.feedback_history = []
        self.query_history = []
    
    def update_query(self, original_query, relevant_features, 
                    irrelevant_features=None):
        """
        Update query vector based on relevance feedback
        
        Args:
            original_query: Original query feature vector
            relevant_features: Feature vectors of relevant images (N_rel x dim)
            irrelevant_features: Feature vectors of irrelevant images (N_irr x dim)
            
        Returns:
            Updated query vector
        """
        # Start with original query
        new_query = self.alpha * original_query
        
        # Add relevant documents contribution
        if relevant_features is not None and len(relevant_features) > 0:
            relevant_mean = np.mean(relevant_features, axis=0)
            new_query += self.beta * relevant_mean
            
        # Subtract irrelevant documents contribution
        if irrelevant_features is not None and len(irrelevant_features) > 0:
            irrelevant_mean = np.mean(irrelevant_features, axis=0)
            new_query -= self.gamma * irrelevant_mean
        
        # Normalize the new query
        new_query = self._normalize_vector(new_query)
        
        # Store in history
        self.query_history.append(new_query.copy())
        self.feedback_history.append({
            'n_relevant': len(relevant_features) if relevant_features is not None else 0,
            'n_irrelevant': len(irrelevant_features) if irrelevant_features is not None else 0
        })
        
        return new_query
    
    def _normalize_vector(self, vector):
        """L2 normalize vector"""
        norm = np.linalg.norm(vector)
        if norm > 0:
            return vector / norm
        return vector
    
    def iterative_feedback(self, original_query, feedback_rounds):
        """
        Apply multiple rounds of feedback
        
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
        """Clear feedback history"""
        self.feedback_history = []
        self.query_history = []
    
    def get_feedback_stats(self):
        """Get statistics about feedback sessions"""
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
        
        Returns:
            Cosine similarity between original and current query
        """
        if len(self.query_history) < 2:
            return 1.0
        
        original = self.query_history[0]
        current = self.query_history[-1]
        
        similarity = np.dot(original, current)
        return similarity


class PseudoRelevanceFeedback:
    """
    Pseudo-Relevance Feedback (PRF)
    Automatically assumes top-k results are relevant (no user input needed)
    """
    
    def __init__(self, top_k=5, alpha=1.0, beta=0.5):
        """
        Initialize PRF
        
        Args:
            top_k: Number of top results to consider as relevant
            alpha: Weight for original query
            beta: Weight for pseudo-relevant documents
        """
        self.top_k = top_k
        self.alpha = alpha
        self.beta = beta
    
    def expand_query(self, original_query, top_features):
        """
        Expand query using top-k pseudo-relevant documents
        
        Args:
            original_query: Original query vector
            top_features: Features of top-k retrieved documents
            
        Returns:
            Expanded query vector
        """
        # Take only top_k
        pseudo_relevant = top_features[:self.top_k]
        
        # Compute centroid of pseudo-relevant documents
        centroid = np.mean(pseudo_relevant, axis=0)
        
        # Combine with original query
        expanded_query = self.alpha * original_query + self.beta * centroid
        
        # Normalize
        norm = np.linalg.norm(expanded_query)
        if norm > 0:
            expanded_query = expanded_query / norm
        
        return expanded_query
    

class InteractiveFeedbackSession:
    """Manage interactive feedback session"""
    
    def __init__(self, rocchio_feedback):
        """
        Initialize feedback session
        
        Args:
            rocchio_feedback: RocchioFeedback instance
        """
        self.rocchio = rocchio_feedback
        self.original_query = None
        self.current_query = None
        self.relevant_indices = []
        self.irrelevant_indices = []
        self.iteration = 0
    
    def start_session(self, query_vector):
        """Start new feedback session"""
        self.original_query = query_vector.copy()
        self.current_query = query_vector.copy()
        self.relevant_indices = []
        self.irrelevant_indices = []
        self.iteration = 0
        self.rocchio.reset_history()
    
    def add_feedback(self, relevant_ids=None, irrelevant_ids=None):
        """Add user feedback"""
        if relevant_ids:
            self.relevant_indices.extend(relevant_ids)
        if irrelevant_ids:
            self.irrelevant_indices.extend(irrelevant_ids)
    
    def apply_feedback(self, all_features):
        """
        Apply feedback and update query
        
        Args:
            all_features: All feature vectors in database
            
        Returns:
            Updated query vector
        """
        # Get features for relevant/irrelevant images
        relevant_feats = None
        if self.relevant_indices:
            relevant_feats = all_features[self.relevant_indices]
        
        irrelevant_feats = None
        if self.irrelevant_indices:
            irrelevant_feats = all_features[self.irrelevant_indices]
        
        # Update query using Rocchio
        self.current_query = self.rocchio.update_query(
            self.original_query,
            relevant_feats,
            irrelevant_feats
        )
        
        self.iteration += 1
        return self.current_query
    
    def get_session_info(self):
        """Get current session information"""
        return {
            'iteration': self.iteration,
            'n_relevant': len(self.relevant_indices),
            'n_irrelevant': len(self.irrelevant_indices),
            'query_drift': self.rocchio.get_query_drift()
        }


if __name__ == "__main__":
    # Test Rocchio feedback
    np.random.seed(42)
    
    # Create sample data
    feature_dim = 2048
    original_query = np.random.randn(feature_dim)
    original_query = original_query / np.linalg.norm(original_query)
    
    # Create relevant and irrelevant samples
    relevant_features = np.random.randn(5, feature_dim)
    relevant_features = relevant_features / np.linalg.norm(relevant_features, axis=1, keepdims=True)
    
    irrelevant_features = np.random.randn(3, feature_dim)
    irrelevant_features = irrelevant_features / np.linalg.norm(irrelevant_features, axis=1, keepdims=True)
    
    # Test Rocchio
    rocchio = RocchioFeedback(alpha=1.0, beta=0.75, gamma=0.25)
    
    print("Testing Rocchio Feedback:")
    print(f"Original query norm: {np.linalg.norm(original_query):.4f}")
    
    new_query = rocchio.update_query(original_query, relevant_features, irrelevant_features)
    print(f"Updated query norm: {np.linalg.norm(new_query):.4f}")
    
    # Check similarity
    similarity = np.dot(original_query, new_query)
    print(f"Similarity to original: {similarity:.4f}")
    
    # Test feedback stats
    stats = rocchio.get_feedback_stats()
    print(f"\nFeedback stats: {stats}")
