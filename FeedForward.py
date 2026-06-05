from mini_torch import Module, Sequential, Linear, ReLU


class FeedForward(Module):
    """
    The Feed-Forward Network module for the Transformer.

    It expands the embedding dimension by a factor of 4, applies a
    non-linearity (ReLU), and projects it back down to the original dimension.
    
    This allows each position to have its own two-layer MLP applied independently.
    """

    def __init__(self, emb_dim):
        """
        Initializes the Feed-Forward Network.

        Args:
            emb_dim (int): The embedding dimension (input and output size).
        """
        super().__init__()

        # The hidden dimension is typically 4x the embedding dimension
        hidden_dim = 4 * emb_dim

        # Chain the layers using Sequential
        self.net = Sequential(
            Linear(emb_dim, hidden_dim),
            ReLU(),
            Linear(hidden_dim, emb_dim)
        )
    # end method

    def forward(self, x):
        """
        Forward pass through the feed-forward network.

        Args:
            x (array): Input tensor of shape (Batch, Sequence_Length, Emb_Dim).

        Returns:
            array: Output tensor of same shape (Batch, Sequence_Length, Emb_Dim).
        """
        return self.net.forward(x)
    # end method

    def backward(self, grad_output):
        """
        Backward pass through the feed-forward network.

        Args:
            grad_output (array): Gradient of shape (Batch, Sequence_Length, Emb_Dim).

        Returns:
            array: Gradient w.r.t. input, same shape.
        """
        return self.net.backward(grad_output)
    # end method

    def parameters(self):
        """Returns all trainable parameters from the internal Sequential."""
        return self.net.parameters()
    # end method

    def grads(self):
        """Returns all gradients from the internal Sequential."""
        return self.net.grads()
    # end method

# end class
