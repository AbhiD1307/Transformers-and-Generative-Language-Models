class CharTokenizer:
    """
    A simple character-level tokenizer that builds a vocabulary from a given
    text corpus and encodes/decodes text to and from integer IDs.
    
    This is much simpler than subword tokenizers (like BPE) because we're
    working at the character level with a small vocabulary (~50-100 chars).
    """

    def __init__(self, text):
        """
        Initializes the tokenizer by scanning the text and building vocabulary.

        Args:
            text (str): The entire training corpus as a single string.
        """
        # Extract all unique characters and sort them for consistency
        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)

        # Build bidirectional lookup dictionaries
        # stoi: string-to-integer (character -> ID)
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        # itos: integer-to-string (ID -> character)
        self.itos = {i: ch for i, ch in enumerate(chars)}
    # end method

    def encode(self, string_data):
        """
        Encodes a string into a list of integer IDs.

        Args:
            string_data (str): The string to encode.

        Returns:
            list: List of integers representing character IDs.
        """
        return [self.stoi[c] for c in string_data]
    # end method

    def decode(self, integer_list):
        """
        Decodes a list of integer IDs back into a string.

        Args:
            integer_list (list or array): List/array of integers to decode.

        Returns:
            str: The reconstructed string.
        """
        return ''.join([self.itos[i] for i in integer_list])
    # end method

# end class
