# cram/utils/metrics.py
import jax
import jax.numpy as jnp
from typing import Dict, List, Optional

class MetricsTracker:
    """Tracks and computes various metrics for model evaluation"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.total_loss = 0.0
        self.total_tokens = 0
        self.total_correct = 0
        self.num_batches = 0
    
    def update(self, predictions, labels, loss, mask=None):
        """Update metrics with new batch results"""
        if mask is None:
            mask = jnp.ones_like(labels)
            
        # Update loss
        self.total_loss += loss * jnp.sum(mask)
        self.total_tokens += jnp.sum(mask)
        
        # Update accuracy
        correct = (predictions == labels) * mask
        self.total_correct += jnp.sum(correct)
        
        self.num_batches += 1
    
    def compute(self) -> Dict[str, float]:
        """Compute final metrics"""
        metrics = {
            'loss': self.total_loss / self.total_tokens,
            'perplexity': jnp.exp(self.total_loss / self.total_tokens),
            'accuracy': self.total_correct / self.total_tokens
        }
        return metrics

