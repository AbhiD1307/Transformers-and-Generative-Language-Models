import pytest
import numpy as np
from mini_torch import xp, asnumpy
from TransformerBlock import TransformerBlock


def test_transformer_block_forward():
    """Tests forward pass shape preservation."""
    B, T, D = 2, 4, 8
    num_heads = 2

    block = TransformerBlock(D, num_heads)
    x = xp.random.randn(B, T, D).astype(xp.float32)

    out = block.forward(x)
    assert out.shape == (B, T, D), f"Output shape mismatch: expected {(B, T, D)}, got {out.shape}"
    print("✓ test_transformer_block_forward passed")


def test_transformer_block_backward():
    """Tests backward pass through residual connections."""
    B, T, D = 2, 4, 8
    num_heads = 2

    block = TransformerBlock(D, num_heads)
    x = xp.random.randn(B, T, D).astype(xp.float32)

    out = block.forward(x)
    grad_out = xp.random.randn(B, T, D).astype(xp.float32)
    dx = block.backward(grad_out)

    assert dx.shape == (B, T, D), f"Gradient shape mismatch: expected {(B, T, D)}, got {dx.shape}"
    print("✓ test_transformer_block_backward passed")


def test_transformer_block_parameters():
    """Tests parameter aggregation from child modules."""
    B, T, D = 2, 4, 8
    num_heads = 2

    block = TransformerBlock(D, num_heads)

    # Parameter count breakdown:
    # - ln_1: 2 (gamma, beta)
    # - attn: 8 (4 Linear layers * 2 for W and b)
    # - ln_2: 2 (gamma, beta)
    # - ffwd: 4 (2 Linear layers * 2 for W and b)
    # Total: 2 + 8 + 2 + 4 = 16
    params = block.parameters()
    grads = block.grads()

    assert len(params) == 16, f"Expected 16 parameters, got {len(params)}"
    assert len(grads) == 16, f"Expected 16 gradients, got {len(grads)}"
    print("✓ test_transformer_block_parameters passed")


def test_residual_connections():
    """Tests that residual connections help gradient flow."""
    B, T, D = 2, 4, 8
    num_heads = 2

    block = TransformerBlock(D, num_heads)
    x = xp.random.randn(B, T, D).astype(xp.float32)

    # Forward and backward passes
    out = block.forward(x)
    grad_out = xp.ones((B, T, D), dtype=xp.float32)  # Uniform gradient
    dx = block.backward(grad_out)

    # With residual connections, gradients should flow back
    # Check that gradients are not vanishingly small
    dx_norm = float(xp.linalg.norm(dx))
    assert dx_norm > 1e-6, f"Gradient norm too small: {dx_norm}"
    print("✓ test_residual_connections passed")


if __name__ == "__main__":
    test_transformer_block_forward()
    test_transformer_block_backward()
    test_transformer_block_parameters()
    test_residual_connections()
    print("\n✓ All TransformerBlock tests passed!")
