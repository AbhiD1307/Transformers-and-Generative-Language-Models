import numpy as np
from mini_torch import Dataset
from CharTokenizer import CharTokenizer


class TextCorpusDataset(Dataset):
    """
    A Dataset subclass that loads a plain text file, tokenizes it at the
    character level, and yields sliding-window (x, y) context-target pairs.

    This implements the autoregressive training paradigm where the model is
    trained to predict the next character given the previous `block_size`
    characters.
    """

    def __init__(self, file_path, block_size):
        """
        Initializes the dataset by loading and tokenizing the text corpus.

        Args:
            file_path (str): Path to the plain text file.
            block_size (int): The length of the context window (max sequence length).
        """
        self.block_size = block_size

        # 1. Read the entire text file
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()

        # 2. Build vocabulary and initialize tokenizer
        self.tokenizer = CharTokenizer(text)

        # 3. Encode the entire text into integer token IDs
        self.data = np.array(self.tokenizer.encode(text), dtype=np.int32)
    # end method

    def __len__(self):
        """
        Returns the total number of valid sequences that can be extracted.

        For a sliding-window model, this is total_chars - block_size because
        each sample uses block_size characters as input and 1 character as target.

        Returns:
            int: Number of valid (x, y) pairs available.
        """
        return len(self.data) - self.block_size
    # end method

    def __getitem__(self, idx):
        """
        Retrieves a single context-target pair at the specified index.

        Args:
            idx (int): The index of the sequence window to extract.

        Returns:
            tuple: (x, y) where:
                - x: input sequence of shape (block_size,) - the context
                - y: target sequence of shape (block_size,) - the next characters
        """
        # x is the input sequence (context window)
        x = self.data[idx : idx + self.block_size]
        # y is the target sequence, shifted forward by 1 position (autoregressive)
        # This means y[i] is the next character after x[i], so the model learns
        # to predict the next character given the context
        y = self.data[idx + 1 : idx + self.block_size + 1]

        return x, y
    # end method

# end class
