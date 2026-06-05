import numpy as np
from mini_torch import xp, Module, Linear, Sequential
from Embedding import Embedding
from TransformerBlock import TransformerBlock
from LayerNorm import LayerNorm


class GPTModel(Module):
    """
    Decoder-only GPT-style language model for character-level text generation.

    Architecture:
        Token Embedding + Positional Embedding
        -> N x TransformerBlock (stacked via Sequential)
        -> Final LayerNorm
        -> Linear language-modeling head (emb_dim -> vocab_size)
    """

    def __init__(self, vocab_size, block_size, emb_dim, num_heads, num_layers):
        """
        Builds the full GPT architecture.

        Args:
            vocab_size  (int): Number of unique characters in the corpus.
            block_size  (int): Maximum context length (sequence length).
            emb_dim     (int): Hidden/embedding dimension.
            num_heads   (int): Number of attention heads per TransformerBlock.
            num_layers  (int): Number of stacked TransformerBlocks.
        """
        super().__init__()

        self.vocab_size = vocab_size
        self.block_size = block_size

        # ---- Embedding tables ----
        # Maps token IDs  (0 .. vocab_size-1)  to dense vectors
        self.tok_emb = Embedding(vocab_size, emb_dim)
        # Maps position indices (0 .. block_size-1) to dense vectors
        self.pos_emb = Embedding(block_size, emb_dim)

        # ---- Transformer body ----
        self.blocks = Sequential(
            *[TransformerBlock(emb_dim, num_heads) for _ in range(num_layers)]
        )

        # ---- Prediction head ----
        self.ln_f    = LayerNorm(emb_dim)
        self.lm_head = Linear(emb_dim, vocab_size)
    # end method

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------
    def forward(self, indices):
        """
        Full forward pass: indices -> logits.

        Args:
            indices (array): Integer array of shape (Batch_Size, Sequence_Length).

        Returns:
            array: Logits of shape (Batch_Size, Sequence_Length, Vocab_Size).
        """
        B, T = indices.shape

        # ---- Step 1: Embeddings ----
        tok_emb = self.tok_emb.forward(indices)   # (B, T, emb_dim)

        # Build position indices [[0,1,...,T-1]] repeated B times
        # Shape: (B, T)  – same shape as indices so Embedding lookup works
        pos_indices = xp.tile(xp.arange(T, dtype=xp.int32).reshape(1, T), (B, 1))
        pos_emb = self.pos_emb.forward(pos_indices)   # (B, T, emb_dim)

        # Combine: "what this token is" + "where it is"
        x = tok_emb + pos_emb   # (B, T, emb_dim)

        # ---- Step 2: Transformer blocks ----
        x = self.blocks.forward(x)   # (B, T, emb_dim)

        # ---- Step 3: Prediction head ----
        x = self.ln_f.forward(x)           # (B, T, emb_dim)
        logits = self.lm_head.forward(x)   # (B, T, vocab_size)

        return logits
    # end method

    # ------------------------------------------------------------------
    # Backward pass
    # ------------------------------------------------------------------
    def backward(self, grad_output):
        """
        Backpropagates gradients through the entire model.

        Args:
            grad_output (array): Gradient from the loss function,
                                  shape (Batch_Size, Sequence_Length, Vocab_Size).

        Returns:
            None: The chain rule stops here (inputs are discrete integers).
        """
        # ---- Step 1: Reverse prediction head ----
        dx = self.lm_head.backward(grad_output)   # (B, T, emb_dim)
        dx = self.ln_f.backward(dx)               # (B, T, emb_dim)

        # ---- Step 2: Reverse transformer blocks ----
        dx = self.blocks.backward(dx)   # (B, T, emb_dim)

        # ---- Step 3: Reverse embeddings ----
        # Addition in forward => gradient distributes equally to both branches
        self.tok_emb.backward(dx)   # accumulates into tok_emb.dweight
        self.pos_emb.backward(dx)   # accumulates into pos_emb.dweight

        # Chain rule stops here: original inputs were discrete integer indices
        return None
    # end method

    # ------------------------------------------------------------------
    # Text generation (autoregressive loop)
    # ------------------------------------------------------------------
    def generate(self, start_indices, max_new_tokens, temperature=1.0):
        """
        Autoregressively generates new tokens one at a time.

        Args:
            start_indices (array): Seed context of shape (1, T_seed).
            max_new_tokens (int): Number of new characters to generate.
            temperature (float): Controls randomness.
                                  0 or <= 0: greedy (argmax).
                                  > 0: sample from distribution scaled by temperature.

        Returns:
            array: Extended token sequence of shape (1, T_seed + max_new_tokens).
        """
        context = start_indices   # shape (1, current_T)

        for _ in range(max_new_tokens):
            # Crop context to the model's maximum context length
            context_crop = context[:, -self.block_size:]

            # Forward pass to get logits for every position
            logits = self.forward(context_crop)   # (1, T, vocab_size)

            # Only the last time-step predicts the next token
            last_logits = logits[0, -1, :]   # (vocab_size,)

            if temperature <= 0.0:
                # Greedy decoding: take the most probable token
                next_token = xp.argmax(last_logits).reshape(1, 1)
            else:
                # Temperature-scaled sampling
                scaled = last_logits / temperature
                # Numerically stable softmax
                scaled -= xp.max(scaled)
                probs = xp.exp(scaled)
                probs /= xp.sum(probs)

                # Sample from the probability distribution using numpy
                probs_np = np.array(probs).astype(np.float64)
                # Re-normalize to avoid floating-point sum != 1 errors
                probs_np /= probs_np.sum()
                next_idx = np.random.choice(len(probs_np), p=probs_np)
                next_token = xp.array([[next_idx]], dtype=xp.int32)
            # end if

            # Append the new token to the context
            context = xp.concatenate([context, next_token], axis=1)
        # end for

        return context
    # end method

    # ------------------------------------------------------------------
    # Parameter / gradient accessors
    # ------------------------------------------------------------------
    def parameters(self):
        return (self.tok_emb.parameters() + self.pos_emb.parameters() +
                self.blocks.parameters() + self.ln_f.parameters() +
                self.lm_head.parameters())
    # end method

    def grads(self):
        return (self.tok_emb.grads() + self.pos_emb.grads() +
                self.blocks.grads() + self.ln_f.grads() +
                self.lm_head.grads())
    # end method

# end class
