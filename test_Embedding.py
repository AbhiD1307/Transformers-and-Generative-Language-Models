import pytest
import numpy as np
from mini_torch import xp, asnumpy
from Embedding import Embedding


def test_embedding_forward():
    """Tests the forward pass returns correct shape."""
    vocab_size = 10
    emb_dim = 8
    B, T = 2, 3

    emb = Embedding(vocab_size, emb_dim)
    indices = xp.array([[1, 2, 3], [4, 5, 6]], dtype=xp.int32)

    out = emb.forward(indices)
    assert out.shape == (B, T, emb_dim), f"Expected shape {(B, T, emb_dim)}, got {out.shape}"
    print("✓ test_embedding_forward passed")


def test_embedding_backward_accumulation():
    """Tests backward correctly accumulates gradients for duplicate indices."""
    vocab_size = 10
    emb_dim = 8
    B, T = 2, 3

    emb = Embedding(vocab_size, emb_dim)
    # Create indices with duplicates: token '1' appears twice, '2' appears twice
    indices = xp.array([[1, 2, 1], [0, 9, 2]], dtype=xp.int32)

    out = emb.forward(indices)

    # Gradient of all ones makes accumulation easy to trace
    grad_out = xp.ones((B, T, emb_dim), dtype=xp.float32)
    dx = emb.backward(grad_out)

    # Should be None since inputs are discrete
    assert dx is None, "Embedding backward should return None"

    # Token '1' appeared twice, so gradient should sum to 2
    grad_1 = asnumpy(emb.dweight[1])
    assert np.allclose(grad_1, 2.0), f"Token 1 gradient should be 2, got {grad_1}"

    # Token '2' appeared twice as well
    grad_2 = asnumpy(emb.dweight[2])
    assert np.allclose(grad_2, 2.0), f"Token 2 gradient should be 2, got {grad_2}"

    # Token '3' never appeared
    grad_3 = asnumpy(emb.dweight[3])
    assert np.allclose(grad_3, 0.0), f"Token 3 gradient should be 0, got {grad_3}"

    print("✓ test_embedding_backward_accumulation passed")


def test_embedding_parameters_grads():
    """Tests parameter and gradient accessors."""
    emb = Embedding(10, 8)

    params = emb.parameters()
    grads = emb.grads()

    assert len(params) == 1, "Should have 1 parameter"
    assert len(grads) == 1, "Should have 1 gradient"
    assert params[0].shape == (10, 8), f"Weight shape should be (10, 8), got {params[0].shape}"

    print("✓ test_embedding_parameters_grads passed")


if __name__ == "__main__":
    test_embedding_forward()
    test_embedding_backward_accumulation()
    test_embedding_parameters_grads()
    print("\n✓ All Embedding tests passed!")
