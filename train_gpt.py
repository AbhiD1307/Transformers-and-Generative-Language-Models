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
    # 1. Hyperparameters (Scaled based on compute availability)
    # ---------------------------------------------------------
    corpus_path = "sample_corpus.txt"  # Change this to your corpus path
    block_size = 32      # Context length
    # GPUs require much larger batches to hide Python kernel launch overhead
    batch_size = 128 if is_gpu_available else 16
    emb_dim = 256 if is_gpu_available else 64         # Embedding dimension
    num_heads = 4        # Number of attention heads
    num_layers = 4 if is_gpu_available else 2         # Number of Transformer blocks
    epochs = 10 if is_gpu_available else 5
    learning_rate = 0.001

    # ---------------------------------------------------------
    # 2. Data Preparation
    # ---------------------------------------------------------
    print(f"Loading dataset from {corpus_path}...")
    dataset = TextCorpusDataset(corpus_path, block_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    vocab_size = dataset.tokenizer.vocab_size
    print(f"Dataset loaded. Vocabulary size: {vocab_size}")
    print(f"Total samples: {len(dataset)}")
    print(f"Batch size: {batch_size}")

    # ---------------------------------------------------------
    # 3. Model, Loss, and Optimizer Initialization
    # ---------------------------------------------------------
    model = GPTModel(vocab_size, block_size, emb_dim, num_heads, num_layers)
    loss_fn = CrossEntropyLoss()
    optimizer = RMSprop([model], lr=learning_rate)

    if is_gpu_available:
        print("Training on GPU...")
    else:
        print("Training on CPU...")
    # end if

    # ---------------------------------------------------------
    # 4. Training Loop
    # ---------------------------------------------------------
    for epoch in range(epochs):
        # Track timing for the epoch
        epoch_start_time = time.perf_counter()

        total_loss = 0.0
        num_batches = len(dataloader)

        for batch_idx, (x, y) in enumerate(dataloader):
            # Move data to the appropriate backend (CPU or GPU)
            x = as_backend_array(x)
            y = as_backend_array(y)

            # ---- Forward pass ----
            logits = model.forward(x)   # (B, T, vocab_size)

            # ---- Compute loss ----
            loss = loss_fn.forward(logits, y)
            total_loss += float(asnumpy(loss))

            # ---- Backward pass ----
            grad_loss = loss_fn.backward()   # Gradient from loss
            model.backward(grad_loss)         # Backprop through model

            # ---- Optimization step ----
            optimizer.step()

            # Progress indicator
            if (batch_idx + 1) % max(1, num_batches // 5) == 0:
                print(
                    f"  Batch {batch_idx + 1}/{num_batches} | "
                    f"Loss: {loss:.4f}"
                )
        # end for

        epoch_end_time = time.perf_counter()
        epoch_duration = epoch_end_time - epoch_start_time
        avg_loss = total_loss / num_batches

        print(f"\nEpoch {epoch + 1}/{epochs}")
        print(f"  Average Loss: {avg_loss:.4f}")
        print(f"  Time: {epoch_duration:.2f}s\n")

        # ---------------------------------------------------------
        # 5. Generation Sample (Autoregressive sampling)
        # ---------------------------------------------------------
        # Seed generation with a newline character
        start_idx = dataset.tokenizer.encode('\n')
        start_context = xp.array([start_idx], dtype=xp.int32).reshape(1, 1)

        print("Generated sample:")
        print("-" * 50)
        generated_indices = model.generate(
            start_context,
            max_new_tokens=100,
            temperature=0.8
        )
        generated_text = dataset.tokenizer.decode(asnumpy(generated_indices)[0].tolist())
        print(generated_text)
        print("-" * 50 + "\n")
    # end for

    # ---------------------------------------------------------
    # 6. Save trained weights
    # ---------------------------------------------------------
    model.save_weights("gpt_weights.pkl")
    print("Training complete! Weights saved to gpt_weights.pkl")

# end main


if __name__ == "__main__":
    main()
