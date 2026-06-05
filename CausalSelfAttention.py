import numpy as np
from mini_torch import xp, Module, Linear


class CausalSelfAttention(Module):
    """
    Multi-head causal (decoder-only) self-attention.

    Computes scaled dot-product attention with a causal mask that prevents
    each position from attending to any future position (no "looking ahead").
    Supports multiple attention heads that operate in parallel.
    """

    def __init__(self, emb_dim, num_heads):
        """
        Initializes projection matrices and structural parameters.

        Args:
            emb_dim (int): Total embedding dimension. Must be divisible by num_heads.
            num_heads (int): Number of parallel attention heads.
        """
        super().__init__()

        assert emb_dim % num_heads == 0, \
            f"emb_dim ({emb_dim}) must be divisible by num_heads ({num_heads})"

        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.head_dim = emb_dim // num_heads

        # Scale factor applied to dot products to prevent softmax saturation
        self.scale = np.sqrt(self.head_dim)

        # Four linear projection matrices (Query, Key, Value, Output)
        self.W_q = Linear(emb_dim, emb_dim)
        self.W_k = Linear(emb_dim, emb_dim)
        self.W_v = Linear(emb_dim, emb_dim)
        self.W_o = Linear(emb_dim, emb_dim)

        # Forward-pass caches needed for backprop
        self.q = None           # Query: (B, num_heads, T, head_dim)
        self.k = None           # Key:   (B, num_heads, T, head_dim)
        self.v = None           # Value: (B, num_heads, T, head_dim)
        self.attn_weights = None  # Softmax output: (B, num_heads, T, T)
        self.mask = None        # Causal mask: reused across forward calls of same T
    # end method

    # ------------------------------------------------------------------
    # Helper: numerically stable softmax along the last axis
    # ------------------------------------------------------------------
    @staticmethod
    def _softmax(x):
        # Subtract max for numerical stability before exponentiation
        x_max = xp.max(x, axis=-1, keepdims=True)
        e_x = xp.exp(x - x_max)
        return e_x / xp.sum(e_x, axis=-1, keepdims=True)
    # end method

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------
    def forward(self, x):
        """
        Computes multi-head causal self-attention.

        Args:
            x (array): Input tensor of shape (Batch, Sequence_Length, Emb_Dim).

        Returns:
            array: Output tensor of shape (Batch, Sequence_Length, Emb_Dim).
        """
        B, T, D = x.shape

        # ---- Step 1: Linear projections to Q, K, V ----
        # Each projection maps (B, T, D) -> (B, T, D)
        q_flat = self.W_q.forward(x)   # (B, T, D)
        k_flat = self.W_k.forward(x)   # (B, T, D)
        v_flat = self.W_v.forward(x)   # (B, T, D)

        # ---- Step 2: Reshape into multi-head format ----
        # Split the embedding dimension into (num_heads, head_dim)
        # then move num_heads axis forward so attention math is per-head
        # (B, T, D) -> (B, T, num_heads, head_dim) -> (B, num_heads, T, head_dim)
        self.q = q_flat.reshape(B, T, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        self.k = k_flat.reshape(B, T, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        self.v = v_flat.reshape(B, T, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        # Each is now: (B, num_heads, T, head_dim)

        # ---- Step 3: Scaled dot-product attention scores ----
        # K transposed: (B, num_heads, head_dim, T)
        # Scores: (B, num_heads, T, T)
        scores = (self.q @ self.k.transpose(0, 1, 3, 2)) / self.scale

        # ---- Step 4: Causal mask ----
        # Build or reuse upper-triangular mask of shape (T, T)
        # Upper triangle (k=1 excludes diagonal) is set to -inf so future
        # positions are zeroed out after softmax
        if self.mask is None or self.mask.shape[0] != T:
            mask = xp.triu(xp.ones((T, T), dtype=xp.float32), k=1)
            self.mask = mask * (-1e9)   # large negative approximates -inf
        # end if
        scores = scores + self.mask   # broadcast over (B, num_heads, T, T)

        # ---- Step 5: Softmax and context vector ----
        self.attn_weights = self._softmax(scores)   # (B, num_heads, T, T)
        context = self.attn_weights @ self.v         # (B, num_heads, T, head_dim)

        # ---- Step 6: Recombine heads ----
        # (B, num_heads, T, head_dim) -> (B, T, num_heads, head_dim) -> (B, T, D)
        context = context.transpose(0, 2, 1, 3)                     # (B, T, num_heads, head_dim)
        context_flat = context.reshape(B, T, self.emb_dim)           # (B, T, D)

        # ---- Step 7: Output projection ----
        out = self.W_o.forward(context_flat)   # (B, T, D)

        return out
    # end method

    # ------------------------------------------------------------------
    # Backward pass
    # ------------------------------------------------------------------
    def backward(self, grad_output):
        """
        Manually backpropagates through all forward operations.

        Args:
            grad_output (array): Gradient of shape (Batch, Sequence_Length, Emb_Dim).

        Returns:
            array: Gradient w.r.t. input x, shape (Batch, Sequence_Length, Emb_Dim).
        """
        B = grad_output.shape[0]
        T = self.q.shape[2]

        # ---- Step 1: Reverse output projection ----
        # grad_output: (B, T, D)
        grad_context_flat = self.W_o.backward(grad_output)   # (B, T, D)

        # ---- Step 2: Reverse head recombination ----
        # (B, T, D) -> (B, T, num_heads, head_dim) -> (B, num_heads, T, head_dim)
        grad_context = grad_context_flat.reshape(
            B, T, self.num_heads, self.head_dim
        ).transpose(0, 2, 1, 3)   # (B, num_heads, T, head_dim)

        # ---- Step 3: Reverse context = attn_weights @ V ----
        # grad_v: (B, num_heads, T, head_dim)
        grad_v = self.attn_weights.transpose(0, 1, 3, 2) @ grad_context
        # grad_attn_weights: (B, num_heads, T, T)
        grad_attn_weights = grad_context @ self.v.transpose(0, 1, 3, 2)

        # ---- Step 4: Reverse softmax ----
        # Efficient Jacobian-vector product for softmax:
        # dL/dscores = attn_weights * (grad_attn_weights - sum(attn_weights * grad_attn_weights))
        sum_term = xp.sum(self.attn_weights * grad_attn_weights, axis=-1, keepdims=True)
        grad_scores = self.attn_weights * (grad_attn_weights - sum_term)
        # Note: -inf masked positions already have attn_weight=0, so their
        # gradients are naturally 0; no explicit un-masking needed.

        # ---- Step 5: Reverse scaled dot-product scores ----
        # scores = Q @ K^T / scale  =>  grad flows to both Q and K
        # grad_q: (B, num_heads, T, head_dim)
        grad_q = (grad_scores @ self.k) / self.scale
        # grad_k: transpose grad_scores so the matmul contracts over T_query
        grad_k = (grad_scores.transpose(0, 1, 3, 2) @ self.q) / self.scale

        # ---- Step 6: Reverse multi-head reshape ----
        # (B, num_heads, T, head_dim) -> (B, T, num_heads, head_dim) -> (B, T, D)
        grad_q_flat = grad_q.transpose(0, 2, 1, 3).reshape(B, T, self.emb_dim)
        grad_k_flat = grad_k.transpose(0, 2, 1, 3).reshape(B, T, self.emb_dim)
        grad_v_flat = grad_v.transpose(0, 2, 1, 3).reshape(B, T, self.emb_dim)

        # ---- Step 7: Reverse linear projections ----
        dx_q = self.W_q.backward(grad_q_flat)
        dx_k = self.W_k.backward(grad_k_flat)
        dx_v = self.W_v.backward(grad_v_flat)

        # ---- Step 8: Sum gradients (x branched into Q, K, V) ----
        dx = dx_q + dx_k + dx_v

        return dx
    # end method

    # ------------------------------------------------------------------
    # Parameter / gradient accessors
    # ------------------------------------------------------------------
    def parameters(self):
        return (self.W_q.parameters() + self.W_k.parameters() +
                self.W_v.parameters() + self.W_o.parameters())
    # end method

    def grads(self):
        return (self.W_q.grads() + self.W_k.grads() +
                self.W_v.grads() + self.W_o.grads())
    # end method

# end class
