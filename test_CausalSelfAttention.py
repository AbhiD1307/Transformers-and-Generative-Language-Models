import pytest
import numpy as np
from mini_torch import xp, asnumpy
from CausalSelfAttention import CausalSelfAttention


def test_causal_self_attention_forward():
    """Tests tensor shapes in forward pass."""
    B, T, D = 2, 4, 8
    num_heads = 2

    attn = CausalSelfAttention(D, num_heads)
    x = xp.random.randn(B, T, D).astype(xp.float32)

    out = attn.forward(x)
    assert out.shape == (B, T, D), f"Output shape mismatch: expected {(B, T, D)}, got {out.shape}"
    print("✓ test_causal_self_attention_forward passed")


def test_causal_masking():
    """Tests that causal mask prevents attending to future tokens."""
    B, T, D = 2, 4, 8
    num_heads = 2

    attn = CausalSelfAttention(D, num_heads)
    x = xp.random.randn(B, T, D).astype(xp.float32)

    out = attn.forward(x)

    # attn_weights shape: (B, num_heads, T, T)
    # Upper triangle (k=1 excludes diagonal) should be zero
    attn_weights_np = asnumpy(attn.attn_weights)
    upper_triangle = np.triu(attn_weights_np[0, 0], k=1)

    assert np.allclose(upper_triangle, 0.0, atol=1e-6), \
        "Causal mask failed: future positions should have zero attention"
    print("✓ test_causal_masking passed")


def test_causal_self_attention_backward():
    """Tests backward pass shapes and gradient flow."""
    B, T, D = 2, 4, 8
    num_heads = 2

    attn = CausalSelfAttention(D, num_heads)
    x = xp.random.randn(B, T, D).astype(xp.float32)

    out = attn.forward(x)
    grad_out = xp.random.randn(B, T, D).astype(xp.float32)
    dx = attn.backward(grad_out)

    assert dx.shape == (B, T, D), f"Gradient shape mismatch: expected {(B, T, D)}, got {dx.shape}"
    
    # Check parameter count: 4 Linear layers (W_q, W_k, W_v, W_o) * 2 (weight + bias) = 8
    params = attn.parameters()
    grads = attn.grads()
    assert len(params) == 8, f"Expected 8 parameters, got {len(params)}"
    assert len(grads) == 8, f"Expected 8 gradients, got {len(grads)}"
    print("✓ test_causal_self_attention_backward passed")


def test_multi_head_structure():
    """Tests that multi-head attention has correct internal structure."""
    emb_dim = 16
    num_heads = 4
    head_dim = emb_dim // num_heads

    attn = CausalSelfAttention(emb_dim, num_heads)
    assert attn.head_dim == head_dim
    assert attn.num_heads == num_heads
    print("✓ test_multi_head_structure passed")


if __name__ == "__main__":
    test_causal_self_attention_forward()
    test_causal_masking()
    test_causal_self_attention_backward()
    test_multi_head_structure()
    print("\n✓ All CausalSelfAttention tests passed!")
