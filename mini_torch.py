"""
Mini-Torch: Minimal deep learning framework for transformers.
Supports both NumPy (CPU) and CuPy (GPU) backends.
"""

import numpy as np
from typing import Union, List, Tuple, Optional

# Try to import CuPy for GPU support
try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False

# Global backend state
_use_gpu = False
xp = np  # Current array backend (NumPy or CuPy)

def is_gpu_available() -> bool:
    """Check if GPU (CuPy) is available."""
    return HAS_CUPY

def set_backend(use_gpu: bool = False) -> None:
    """Set the computation backend (CPU or GPU)."""
    global _use_gpu, xp
    if use_gpu and not HAS_CUPY:
        print("Warning: CuPy not available, falling back to NumPy")
        _use_gpu = False
        xp = np
    else:
        _use_gpu = use_gpu
        xp = cp if use_gpu else np

def get_backend():
    """Get current array backend."""
    return xp

def as_backend_array(arr):
    """Convert array to current backend (NumPy or CuPy)."""
    if _use_gpu:
        if HAS_CUPY:
            return cp.asarray(arr)
    return np.asarray(arr)

def asnumpy(arr):
    """Convert array to NumPy (for CPU operations/visualization)."""
    if HAS_CUPY:
        if isinstance(arr, cp.ndarray):
            return cp.asnumpy(arr)
    return np.asarray(arr)

class Dataset:
    """Base class for datasets."""
    
    def __len__(self):
        raise NotImplementedError
    
    def __getitem__(self, idx):
        raise NotImplementedError

class Module:
    """Base class for all modules."""
    
    def __init__(self):
        self._parameters = {}
        self._children = {}
    
    def parameters(self) -> List[np.ndarray]:
        """Return list of parameters."""
        params = []
        for param in self._parameters.values():
            params.append(param)
        for child in self._children.values():
            if isinstance(child, Module):
                params.extend(child.parameters())
            elif isinstance(child, list):
                for c in child:
                    if isinstance(c, Module):
                        params.extend(c.parameters())
        return params
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
    
    def forward(self, *args, **kwargs):
        raise NotImplementedError
    
    def __setattr__(self, name, value):
        if isinstance(value, Module):
            if '_children' not in self.__dict__:
                super().__setattr__('_children', {})
            self._children[name] = value
        elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], Module):
            if '_children' not in self.__dict__:
                super().__setattr__('_children', {})
            self._children[name] = value
        super().__setattr__(name, value)

class Linear(Module):
    """Fully connected layer: y = xW + b"""
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias
        
        # Initialize weights
        scale = np.sqrt(2.0 / in_features)  # He initialization
        self.weight = as_backend_array(
            np.random.randn(in_features, out_features) * scale
        )
        if bias:
            self.bias = as_backend_array(np.zeros(out_features))
        else:
            self.bias = None
        
        self._parameters['weight'] = self.weight
        if bias:
            self._parameters['bias'] = self.bias
    
    def forward(self, x):
        # x: (batch, ..., in_features)
        # output: (batch, ..., out_features)
        out = x @ self.weight
        if self.use_bias:
            out = out + self.bias
        return out

class ReLU(Module):
    """Rectified Linear Unit: max(0, x)"""
    
    def forward(self, x):
        return xp.maximum(0, x)

class Sequential(Module):
    """Container for sequential modules."""
    
    def __init__(self, *modules):
        super().__init__()
        self.modules_list = modules
        for i, module in enumerate(modules):
            setattr(self, str(i), module)
    
    def forward(self, x):
        for module in self.modules_list:
            x = module(x)
        return x

class DataLoader:
    """Simple data loader for batching."""
    
    def __init__(self, dataset, batch_size: int = 32, shuffle: bool = False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(dataset))
    
    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
        
        for i in range(0, len(self.dataset), self.batch_size):
            batch_indices = self.indices[i:i + self.batch_size]
            batch_x, batch_y = [], []
            for idx in batch_indices:
                x, y = self.dataset[idx]
                batch_x.append(x)
                batch_y.append(y)
            
            batch_x = as_backend_array(np.array(batch_x))
            batch_y = as_backend_array(np.array(batch_y))
            yield batch_x, batch_y
    
    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size

class Optimizer:
    """Base optimizer class."""
    
    def __init__(self, params: List[np.ndarray], lr: float = 0.001):
        self.params = params
        self.lr = lr
    
    def zero_grad(self):
        """Reset gradients to None."""
        for param in self.params:
            param.grad = None
    
    def step(self):
        """Update parameters."""
        raise NotImplementedError

class SGD(Optimizer):
    """Stochastic Gradient Descent."""
    
    def step(self):
        for param in self.params:
            if param.grad is not None:
                param -= self.lr * param.grad

class RMSprop(Optimizer):
    """RMSprop optimizer."""
    
    def __init__(self, params: List[np.ndarray], lr: float = 0.001, decay: float = 0.99, eps: float = 1e-8):
        super().__init__(params, lr)
        self.decay = decay
        self.eps = eps
        self.square_avg = [xp.zeros_like(p) for p in params]
    
    def step(self):
        for i, param in enumerate(self.params):
            if param.grad is not None:
                self.square_avg[i] = self.decay * self.square_avg[i] + (1 - self.decay) * (param.grad ** 2)
                param -= self.lr * param.grad / (xp.sqrt(self.square_avg[i]) + self.eps)

class Adam(Optimizer):
    """Adam optimizer."""
    
    def __init__(self, params: List[np.ndarray], lr: float = 0.001, betas: Tuple = (0.9, 0.999), eps: float = 1e-8):
        super().__init__(params, lr)
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.m = [xp.zeros_like(p) for p in params]
        self.v = [xp.zeros_like(p) for p in params]
        self.t = 0
    
    def step(self):
        self.t += 1
        for i, param in enumerate(self.params):
            if param.grad is not None:
                self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * param.grad
                self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (param.grad ** 2)
                
                m_hat = self.m[i] / (1 - self.beta1 ** self.t)
                v_hat = self.v[i] / (1 - self.beta2 ** self.t)
                
                param -= self.lr * m_hat / (xp.sqrt(v_hat) + self.eps)

__all__ = [
    'Dataset', 'Module', 'Linear', 'ReLU', 'Sequential', 'DataLoader',
    'Optimizer', 'SGD', 'RMSprop', 'Adam',
    'xp', 'set_backend', 'is_gpu_available', 'as_backend_array', 'asnumpy', 'get_backend'
]
