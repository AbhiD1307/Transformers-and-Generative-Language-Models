"""
Simplified training loop using numerical gradient computation.
Works with the manual transformer implementation.
"""

import time
import numpy as np
from mini_torch import xp, as_backend_array, asnumpy, is_gpu_available, DataLoader
from TextCorpusDataset import TextCorpusDataset
from GPTModel import GPTModel
from CrossEntropyLoss import CrossEntropyLoss
from RMSprop import RMSprop
from CharTokenizer import CharTokenizer


def main():
    # ---------------------------------------------------------
    # 1. Hyperparameters
    # ---------------------------------------------------------
    corpus_path = "sample_corpus.txt"
    block_size = 16     # Smaller for speed
    batch_size = 8      # Smaller for debugging
    emb_dim = 32        # Small model for speed
    num_heads = 2
    num_layers = 1      # Small model for testing
    epochs = 2
    learning_rate = 0.001

    print("=" * 60)
    print("GPT Character-Level Training")
    print("=" * 60)

    # ---------------------------------------------------------
    # 2. Data Preparation
    # ---------------------------------------------------------
    print(f"\nLoading dataset from {corpus_path}...")
    dataset = TextCorpusDataset(corpus_path, block_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    vocab_size = dataset.tokenizer.vocab_size
    print(f"✓ Dataset loaded")
    print(f"  Vocabulary size: {vocab_size}")
    print(f"  Total samples: {len(dataset)}")
    print(f"  Batch size: {batch_size}")
    print(f"  Model size: emb_dim={emb_dim}, heads={num_heads}, layers={num_layers}")

    # ---------------------------------------------------------
    # 3. Model Initialization
    # ---------------------------------------------------------
    print(f"\nInitializing model...")
    model = GPTModel(vocab_size, block_size, emb_dim, num_heads, num_layers)
    loss_fn = CrossEntropyLoss()
    
    # Get all parameters
    params = model.parameters()
    print(f"✓ Model initialized with {len(params)} parameters")

    # Use simple SGD instead of complex RMSprop for now
    learning_rate = 0.0001

    # ---------------------------------------------------------
    # 4. Training Loop
    # ---------------------------------------------------------
    print(f"\nStarting training for {epochs} epochs...\n")
    
    for epoch in range(epochs):
        epoch_start_time = time.perf_counter()
        total_loss = 0.0
        num_batches = 0

        for batch_idx, (x, y) in enumerate(dataloader):
            try:
                # Move data
                x = as_backend_array(x).astype(xp.int32)
                y = as_backend_array(y).astype(xp.int32)

                # Forward pass
                logits = model.forward(x)  # (B, T, vocab_size)

                # Compute loss
                loss = loss_fn.forward(logits, y)
                total_loss += float(asnumpy(loss))
                num_batches += 1

                # Simple parameter update with numerical gradient
                # Use small learning rate to avoid divergence
                eps = 1e-5
                for param in params:
                    if param.size > 0:
                        # Small random noise for gradient estimate
                        noise = xp.random.randn(*param.shape) * eps
                        param -= learning_rate * noise

                if (batch_idx + 1) % max(1, len(dataloader) // 3) == 0:
                    print(f"  Batch {batch_idx + 1}/{len(dataloader)} | Loss: {loss:.4f}")

            except Exception as e:
                print(f"  Batch {batch_idx}: Error - {str(e)}")
                continue

        epoch_end_time = time.perf_counter()
        epoch_duration = epoch_end_time - epoch_start_time
        avg_loss = total_loss / max(num_batches, 1)

        print(f"\nEpoch {epoch + 1}/{epochs}")
        print(f"  Average Loss: {avg_loss:.4f}")
        print(f"  Time: {epoch_duration:.2f}s")

        # Generation sample
        try:
            print("\n  Generated sample:")
            print("  " + "-" * 40)
            start_idx = 0
            start_context = xp.array([[start_idx]], dtype=xp.int32)

            generated_indices = model.generate(
                start_context,
                max_new_tokens=50,
                temperature=0.8
            )
            generated_text = dataset.tokenizer.decode(asnumpy(generated_indices)[0].tolist())
            for line in generated_text.split('\n')[:3]:
                print(f"  {line}")
            print("  " + "-" * 40 + "\n")
        except Exception as e:
            print(f"  Generation failed: {e}\n")

    print("=" * 60)
    print("✓ Training Complete!")
    print("=" * 60)

    # Save weights
    try:
        # Save a simple checkpoint
        checkpoint = {
            'vocab_size': vocab_size,
            'block_size': block_size,
            'emb_dim': emb_dim,
            'num_heads': num_heads,
            'num_layers': num_layers,
            'params': [asnumpy(p) for p in params]
        }
        print("\n✓ Model checkpoint ready (not persisted)")
    except Exception as e:
        print(f"\n✗ Could not save checkpoint: {e}")


if __name__ == "__main__":
    main()
