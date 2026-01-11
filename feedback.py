"""
Relevance Feedback implementation using Rocchio algorithm
"""

import numpy as np

class RocchioFeedback:
    def __init__(self, alpha=1.0, beta=0.75, gamma=0.25):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.feedback_history = []
        self.query_history = []
    
    def update_query(self, original_query, relevant_features, irrelevant_features=None):
        new_query = self.alpha * original_query
        
        if relevant_features is not None and len(relevant_features) > 0:
            relevant_mean = np.mean(relevant_features, axis=0)
            new_query += self.beta * relevant_mean
            
        if irrelevant_features is not None and len(irrelevant_features) > 0:
            irrelevant_mean = np.mean(irrelevant_features, axis=0)
            new_query -= self.gamma * irrelevant_mean
        
        new_query = self._normalize_vector(new_query)
        
        self.query_history.append(new_query.copy())
        self.feedback_history.append({
            'n_relevant': len(relevant_features) if relevant_features is not None else 0,
            'n_irrelevant': len(irrelevant_features) if irrelevant_features is not None else 0
        })
        
        return new_query
    
    def _normalize_vector(self, vector):
        norm = np.linalg.norm(vector)
        return vector / norm if norm > 0 else vector

    def reset_history(self):
        self.feedback_history = []
        self.query_history = []

    def get_query_drift(self):
        if len(self.query_history) < 2:
            return 1.0
        return np.dot(self.query_history[0], self.query_history[-1])


class InteractiveFeedbackSession:
    def __init__(self, rocchio_feedback):
        self.rocchio = rocchio_feedback
        self.original_query = None
        self.current_query = None
        self.relevant_indices = set()
        self.irrelevant_indices = set()
        self.iteration = 0
    
    def start_session(self, query_vector):
        self.original_query = query_vector.copy()
        self.current_query = query_vector.copy()
        self.relevant_indices = set()
        self.irrelevant_indices = set()
        self.iteration = 0
        self.rocchio.reset_history()
        self.rocchio.query_history.append(self.original_query.copy())
    
    def add_feedback(self, relevant_ids=None, irrelevant_ids=None):
        if relevant_ids:
            new_rel = set(relevant_ids)
            self.relevant_indices.update(new_rel)
            self.irrelevant_indices.difference_update(new_rel)
            
        if irrelevant_ids:
            new_irr = set(irrelevant_ids)
            self.irrelevant_indices.update(new_irr)
            self.relevant_indices.difference_update(new_irr)
    
    def apply_feedback(self, all_features):
        relevant_feats = None
        if self.relevant_indices:
            relevant_feats = all_features[list(self.relevant_indices)]
        
        irrelevant_feats = None
        if self.irrelevant_indices:
            irrelevant_feats = all_features[list(self.irrelevant_indices)]
        
        self.current_query = self.rocchio.update_query(
            self.original_query,
            relevant_feats,
            irrelevant_feats
        )
        
        self.iteration += 1
        return self.current_query
    
    def get_session_info(self):
        return {
            'iteration': self.iteration,
            'n_relevant': len(self.relevant_indices),
            'n_irrelevant': len(self.irrelevant_indices),
            'query_drift': self.rocchio.get_query_drift()
        }
