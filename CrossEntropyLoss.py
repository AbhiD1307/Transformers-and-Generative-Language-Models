"""
Cross Entropy Loss for multi-class classification.
"""

import numpy as np
from mini_torch import xp, asnumpy

class CrossEntropyLoss:
    """Cross entropy loss: -log(softmax(logits)[target])"""
    
    def __call__(self, logits, targets):
        """
        Args:
            logits: (batch, seq_len, vocab_size) or (batch, vocab_size)
            targets: (batch, seq_len) or (batch,) - class indices
        
        Returns:
            loss: scalar
        """
        return self.forward(logits, targets)
    
    def forward(self, logits, targets):
        """Compute cross entropy loss."""
        # Handle 2D and 3D cases
        original_shape = logits.shape
        is_3d = len(original_shape) == 3
        
        if is_3d:
            batch, seq_len, vocab_size = original_shape
            # Reshape to (batch*seq_len, vocab_size)
            logits_flat = logits.reshape(-1, vocab_size)
            targets_flat = targets.reshape(-1)
        else:
            logits_flat = logits
            targets_flat = targets
        
        # Numerically stable softmax
        # Subtract max for numerical stability
        logits_stable = logits_flat - xp.max(logits_flat, axis=1, keepdims=True)
        
        # Compute softmax
        exp_logits = xp.exp(logits_stable)
        softmax = exp_logits / xp.sum(exp_logits, axis=1, keepdims=True)
        
        # Get probabilities of target classes
        batch_size = logits_flat.shape[0]
        target_probs = softmax[xp.arange(batch_size), targets_flat.astype(int)]
        
        # Compute loss: -log(p)
        # Add small epsilon to avoid log(0)
        eps = 1e-10
        loss = -xp.log(target_probs + eps)
        
        # Average loss over batch
        return xp.mean(loss)

__all__ = ['CrossEntropyLoss']
