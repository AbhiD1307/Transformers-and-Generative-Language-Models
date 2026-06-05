"""
RMSprop Optimizer implementation.
"""

import numpy as np
from mini_torch import xp

class RMSprop:
    """RMSprop optimizer for gradient-based optimization."""
    
    def __init__(self, params, lr=0.001, decay=0.99, eps=1e-8):
        """
        Args:
            params: List of parameters to optimize
            lr: Learning rate
            decay: Decay rate for moving average
            eps: Small constant for numerical stability
        """
        self.params = params
        self.lr = lr
        self.decay = decay
        self.eps = eps
        
        # Initialize square averages
        self.square_avg = [xp.zeros_like(p) for p in params]
    
    def zero_grad(self):
        """Reset gradients."""
        for param in self.params:
            param.grad = None
    
    def step(self):
        """Perform one optimization step."""
        for i, param in enumerate(self.params):
            if hasattr(param, 'grad') and param.grad is not None:
                grad = param.grad
                
                # Update square average: E[g²] = decay * E[g²] + (1-decay) * g²
                self.square_avg[i] = (self.decay * self.square_avg[i] + 
                                     (1 - self.decay) * (grad ** 2))
                
                # Update parameter: param -= lr * g / (sqrt(E[g²]) + eps)
                param -= self.lr * grad / (xp.sqrt(self.square_avg[i]) + self.eps)

__all__ = ['RMSprop']
