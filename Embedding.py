import numpy as np
from mini_torch import xp, Module


class Embedding(Module):
    """
    A lookup table that maps integer token IDs to dense continuous vectors.
    Used twice in GPTModel: once for token embeddings, once for positional embeddings.
    """

    def __init__(self, num_embeddings, embedding_dim):
        """
        Initializes the embedding lookup table.

        Args:
            num_embeddings (int): Total vocabulary size (number of unique tokens).
            embedding_dim (int): Size of each embedding vector.
        """
        super().__init__()

        # Trainable weight matrix: shape (num_embeddings, embedding_dim)
        # Small random initialization from a standard normal distribution
        self.weight = xp.array(
            np.random.randn(num_embeddings, embedding_dim).astype(np.float32) * 0.01
        )

        # Cache for backward pass: remembers which indices were looked up
        self.indices = None

        # Gradient accumulator: same shape as weight, initialized to zeros
        self.dweight = xp.zeros_like(self.weight)
    # end method

    def forward(self, indices):
        """
        Performs a lookup: retrieves the embedding vector for each index.

        Args:
            indices (array): 2D integer array of shape (Batch_Size, Sequence_Length).

        Returns:
            array: 3D float array of shape (Batch_Size, Sequence_Length, Embedding_Dim).
        """
        # Cache indices for use in backward pass
        self.indices = indices

        # NumPy/CuPy advanced indexing: automatically returns 3D output
        return self.weight[indices]
    # end method

    def backward(self, grad_output):
        """
        Accumulates gradients into the weight matrix rows corresponding to
        the cached indices. Duplicate indices are handled correctly via add.at.

        Args:
            grad_output (array): Gradient of shape (Batch_Size, Sequence_Length, Embedding_Dim).

        Returns:
            None: The chain rule stops here because the inputs were discrete integers.
        """
        # Reset gradient accumulator to zero before accumulating
        self.dweight = xp.zeros_like(self.weight)

        # np.add.at safely handles duplicate indices by summing gradients
        # rather than overwriting, which would lose contributions from repeated tokens
        np.add.at(self.dweight, self.indices, grad_output)

        # No gradient to pass further back: discrete integer inputs have no gradient
        return None
    # end method

    def parameters(self):
        """Returns the list of trainable parameters."""
        return [self.weight]
    # end method

    def grads(self):
        """Returns the list of gradients corresponding to parameters()."""
        return [self.dweight]
    # end method

# end class
