import pytest
import numpy as np
from mini_torch import xp
from GPTModel import GPTModel


def test_gpt_model_forward():
    """Tests the complete forward pass through the GPT model."""
    vocab_size = 50
    block_size = 8
    emb_dim = 16
    num_heads = 2
    num_layers = 2
    B, T = 2, 6

    model = GPTModel(vocab_size, block_size, emb_dim, num_heads, num_layers)
    indices = xp.random.randint(0, vocab_size, (B, T), dtype=xp.int32)

    logits = model.forward(indices)
    assert logits.shape == (B, T, vocab_size), \
        f"Logits shape mismatch: expected {(B, T, vocab_size)}, got {logits.shape}"
    print("✓ test_gpt_model_forward passed")


def test_gpt_model_backward():
    """Tests the backward pass through the full model."""
    vocab_size = 50
    block_size = 8
    emb_dim = 16
    num_heads = 2
    num_layers = 2
    B, T = 2, 6

    model = GPTModel(vocab_size, block_size, emb_dim, num_heads, num_layers)
    indices = xp.random.randint(0, vocab_size, (B, T), dtype=xp.int32)

    logits = model.forward(indices)
    grad_out = xp.random.randn(B, T, vocab_size).astype(xp.float32)

    dx = model.backward(grad_out)

    # Chain rule stops at discrete inputs
    assert dx is None, "Backward should return None (chain rule stops)"
    assert len(model.parameters()) == len(model.grads())
    print("✓ test_gpt_model_backward passed")


def test_gpt_model_generate():
    """Tests the autoregressive generation loop."""
    vocab_size = 20
    block_size = 5
    emb_dim = 8
    num_heads = 2
    num_layers = 1

    model = GPTModel(vocab_size, block_size, emb_dim, num_heads, num_layers)

    # Start with 2 tokens
    start_context = xp.array([[3, 5]], dtype=xp.int32)
    max_new_tokens = 8

    generated = model.generate(start_context, max_new_tokens=max_new_tokens)

    # Should have original 2 + 8 new = 10 tokens
    expected_length = 2 + max_new_tokens
    assert generated.shape == (1, expected_length), \
        f"Generated shape mismatch: expected {(1, expected_length)}, got {generated.shape}"

    # All generated tokens should be valid (< vocab_size)
    assert xp.all(generated >= 0) and xp.all(generated < vocab_size), \
        "Generated tokens out of vocabulary range"
    print("✓ test_gpt_model_generate passed")


def test_gpt_model_generation_respects_block_size():
    """Tests that generation respects the block_size constraint."""
    vocab_size = 20
    block_size = 4
    emb_dim = 8
    num_heads = 2
    num_layers = 1

    model = GPTModel(vocab_size, block_size, emb_dim, num_heads, num_layers)

    # Start with a short context
    start_context = xp.array([[1, 2]], dtype=xp.int32)
    max_new_tokens = 10

    generated = model.generate(start_context, max_new_tokens=max_new_tokens, temperature=0.5)

    # Should have 2 + 10 = 12 tokens, but internal cropping maintains block_size
    assert generated.shape[1] == 12
    print("✓ test_gpt_model_generation_respects_block_size passed")


def test_gpt_model_different_sequence_lengths():
    """Tests that model handles variable sequence lengths (< block_size)."""
    vocab_size = 30
    block_size = 8
    emb_dim = 16
    num_heads = 2
    num_layers = 2

    model = GPTModel(vocab_size, block_size, emb_dim, num_heads, num_layers)

    # Try different sequence lengths, all < block_size
    for T in [1, 2, 4, 7]:
        B = 2
        indices = xp.random.randint(0, vocab_size, (B, T), dtype=xp.int32)
        logits = model.forward(indices)
        assert logits.shape == (B, T, vocab_size)

    print("✓ test_gpt_model_different_sequence_lengths passed")


def test_gpt_model_parameters_count():
    """Tests that parameter counting is correct."""
    vocab_size = 50
    block_size = 8
    emb_dim = 16
    num_heads = 2
    num_layers = 2

    model = GPTModel(vocab_size, block_size, emb_dim, num_heads, num_layers)

    params = model.parameters()
    grads = model.grads()

    assert len(params) == len(grads), "Parameter and gradient counts mismatch"
    assert len(params) > 0, "No parameters found"
    print(f"✓ test_gpt_model_parameters_count passed ({len(params)} parameters)")


if __name__ == "__main__":
    test_gpt_model_forward()
    test_gpt_model_backward()
    test_gpt_model_generate()
    test_gpt_model_generation_respects_block_size()
    test_gpt_model_different_sequence_lengths()
    test_gpt_model_parameters_count()
    print("\n✓ All GPTModel tests passed!")
