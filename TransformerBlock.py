from mini_torch import Module
from LayerNorm import LayerNorm
from CausalSelfAttention import CausalSelfAttention
from FeedForward import FeedForward


class TransformerBlock(Module):
    """
    A single Transformer decoder block using Pre-LayerNorm architecture.

    Architecture (per sub-step):
        x -> LN1 -> CausalSelfAttention -> (+x) -> x1
        x1 -> LN2 -> FeedForward         -> (+x1) -> out

    The residual (skip) connections allow gradients to flow directly from
    later layers to earlier ones, enabling deep networks to train without
    gradient vanishing.
    """

    def __init__(self, emb_dim, num_heads, drop_rate=0.1):
        """
        Instantiates the four sub-modules of the block.

        Args:
            emb_dim (int): Embedding dimension.
            num_heads (int): Number of attention heads for CausalSelfAttention.
            drop_rate (float): Dropout probability (reserved for future use).
        """
        super().__init__()

        self.ln_1  = LayerNorm(emb_dim)
        self.attn  = CausalSelfAttention(emb_dim, num_heads)
        self.ln_2  = LayerNorm(emb_dim)
        self.ffwd  = FeedForward(emb_dim)
    # end method

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------
    def forward(self, x):
        """
        Routes input through attention and feed-forward sub-layers,
        each wrapped in a Pre-LN residual connection.

        Args:
            x (array): Shape (Batch_Size, Sequence_Length, Emb_Dim).

        Returns:
            array: Same shape as input.
        """
        # ---- Attention path with residual ----
        # Pre-norm: normalize x before feeding into attention
        norm1_out = self.ln_1.forward(x)
        attn_out  = self.attn.forward(norm1_out)
        # Residual addition: add original x (skip connection)
        x1 = x + attn_out

        # ---- Feed-forward path with residual ----
        # Pre-norm: normalize x1 before feeding into FFN
        norm2_out = self.ln_2.forward(x1)
        ffwd_out  = self.ffwd.forward(norm2_out)
        # Residual addition: add x1 (second skip connection)
        out = x1 + ffwd_out

        return out
    # end method

    # ------------------------------------------------------------------
    # Backward pass
    # ------------------------------------------------------------------
    def backward(self, grad_output):
        """
        Backpropagates through both residual paths.

        At each residual branch  y = x + F(x),  the incoming gradient
        splits:  grad flows unchanged through the skip branch AND through
        F; the two returning gradients are then summed at the branch point.

        Args:
            grad_output (array): Gradient from the subsequent block,
                                  shape (Batch_Size, Sequence_Length, Emb_Dim).

        Returns:
            array: Gradient w.r.t. input x, same shape.
        """
        # ---- Step 1: Backprop through FeedForward residual ----
        # Skip connection passes gradient straight through
        grad_x1_skip = grad_output

        # Gradient flows through FFN then LN2
        grad_ffwd_in = self.ffwd.backward(grad_output)
        grad_ln2_in  = self.ln_2.backward(grad_ffwd_in)

        # Sum the two gradient paths at the x1 branch point
        grad_x1_total = grad_x1_skip + grad_ln2_in

        # ---- Step 2: Backprop through Attention residual ----
        # Skip connection passes gradient straight through
        grad_x_skip = grad_x1_total

        # Gradient flows through attention then LN1
        grad_attn_in = self.attn.backward(grad_x1_total)
        grad_ln1_in  = self.ln_1.backward(grad_attn_in)

        # Sum the two gradient paths at the x branch point
        dx = grad_x_skip + grad_ln1_in

        return dx
    # end method

    # ------------------------------------------------------------------
    # Parameter / gradient accessors (aggregate all child modules)
    # ------------------------------------------------------------------
    def parameters(self):
        return (self.ln_1.parameters() + self.attn.parameters() +
                self.ln_2.parameters() + self.ffwd.parameters())
    # end method

    def grads(self):
        return (self.ln_1.grads() + self.attn.grads() +
                self.ln_2.grads() + self.ffwd.grads())
    # end method

# end class
