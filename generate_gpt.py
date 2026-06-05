import os
import numpy as np
from mini_torch import xp, as_backend_array, asnumpy, is_gpu_available
from CharTokenizer import CharTokenizer
from GPTModel import GPTModel


def main():
    """
    Loads a trained GPT model and generates text autoregressively.
    """
    corpus_path = "../../Transformers/Pride-and-Prejudice-Cleaned.txt"
    weights_path = "gpt_weights.pkl"

    # Verify required files exist
    if not os.path.exists(corpus_path):
        print(f"Error: Corpus file not found at {corpus_path}.")
        print("This is needed to rebuild the tokenizer vocabulary.")
        return

    if not os.path.exists(weights_path):
        print(f"Error: Weights file not found at {weights_path}.")
        print("Run train_gpt.py first to train and save the model.")
        return

    # ---------------------------------------------------------
    # 1. Rebuild the tokenizer from corpus
    # ---------------------------------------------------------
    print(f"Rebuilding tokenizer from {corpus_path}...")
    with open(corpus_path, 'r', encoding='utf-8') as f:
        text = f.read()
    tokenizer = CharTokenizer(text)
    vocab_size = tokenizer.vocab_size
    print(f"Vocabulary size: {vocab_size}")

    # ---------------------------------------------------------
    # 2. Hyperparameters (Must match training!)
    # ---------------------------------------------------------
    block_size = 32      # Context length
    emb_dim = 256 if is_gpu_available else 64         # Embedding dimension
    num_heads = 4        # Number of attention heads
    num_layers = 4 if is_gpu_available else 2         # Number of Transformer blocks

    # ---------------------------------------------------------
    # 3. Instantiate model and load weights
    # ---------------------------------------------------------
    print("Initializing model and loading weights...")
    model = GPTModel(vocab_size, block_size, emb_dim, num_heads, num_layers)
    model.load_weights(weights_path)
    print("Model loaded successfully!")

    # ---------------------------------------------------------
    # 4. Generate Text
    # ---------------------------------------------------------
    max_tokens = 400
    temperature = 0.8

    print(f"\nGenerating {max_tokens} characters with temperature={temperature}...\n")
    print("=" * 60)

    # Seed generation with a newline character
    start_idx = tokenizer.encode('\n')
    start_context = xp.array([start_idx], dtype=xp.int32).reshape(1, 1)

    # Autoregressive generation loop
    generated_indices = model.generate(
        start_context,
        max_new_tokens=max_tokens,
        temperature=temperature
    )

    # Decode back to text
    generated_text = tokenizer.decode(asnumpy(generated_indices)[0].tolist())
    print(generated_text)
    print("=" * 60)

# end main


if __name__ == "__main__":
    main()
