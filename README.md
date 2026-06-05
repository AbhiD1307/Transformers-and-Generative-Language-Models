# Character-Level Generative Transformer (GPT-style)

A from-scratch implementation of a decoder-only transformer model for character-level text generation, built on top of the Mini-Torch framework.

## Architecture Overview

```
Input (Character IDs)
    ↓
[Token Embedding (vocab_size → emb_dim)]
[Positional Embedding (block_size → emb_dim)]
    ↓ (add)
[N × TransformerBlock]
    ├─ Pre-LayerNorm
    ├─ CausalSelfAttention (with causal mask)
    ├─ Residual Connection
    ├─ Pre-LayerNorm
    ├─ FeedForward (4× expansion)
    └─ Residual Connection
    ↓
[Final LayerNorm]
    ↓
[Linear Head (emb_dim → vocab_size)]
    ↓
Logits (Batch, Sequence_Length, Vocab_Size)
```

## Core Components Implemented

### 1. **Embedding.py** - Token & Positional Embeddings
- Lookup table mapping integer IDs to dense vectors
- Efficient forward using advanced indexing
- Backward pass uses `np.add.at()` for safe gradient accumulation with duplicate indices

### 2. **CausalSelfAttention.py** - Multi-Head Self-Attention
- Scaled dot-product attention: `(Q @ K^T / √d) * V`
- **Causal masking**: Upper triangular mask prevents attending to future tokens
- Multi-head parallelization for better feature learning
- Complete manual backpropagation through all operations

### 3. **TransformerBlock.py** - Transformer Decoder Block
- Pre-LayerNorm architecture (LN → Module → Residual)
- **Skip connections** enable gradient flow in deep networks
- Combines:
  - LayerNorm
  - CausalSelfAttention
  - FeedForward (MLP)

### 4. **GPTModel.py** - Top-Level Assembly
- Stacks TransformerBlocks
- Token + Positional embeddings (addition)
- Final projection to vocabulary logits
- **Autoregressive generation**: samples tokens iteratively with temperature control

### 5. **Support Modules**
- **LayerNorm.py**: Feature normalization with learnable scale/shift
- **FeedForward.py**: 4× expansion MLP per position
- **CharTokenizer.py**: Simple character-level encoding/decoding
- **TextCorpusDataset.py**: Sliding-window dataset for autoregressive training

## Training & Inference

### Training
```bash
python train_gpt.py
```
- Loads text corpus, trains on batches using CrossEntropyLoss
- Uses RMSprop optimizer
- Saves weights to `gpt_weights.pkl`
- Demonstrates generation every epoch

### Generation
```bash
python generate_gpt.py
```
- Loads saved weights
- Autoregressively generates 400 characters
- Temperature parameter controls randomness

### Data Preprocessing
```bash
python clean_corpus.py input.txt output.txt
```
- Removes special characters
- Reduces vocabulary for simpler training
- Keeps: letters, digits, basic punctuation, whitespace

## Key Hyperparameters

| Parameter | Default (CPU) | GPU |
|-----------|---------------|-----|
| `block_size` | 32 | 32 |
| `emb_dim` | 64 | 256 |
| `num_heads` | 4 | 4 |
| `num_layers` | 2 | 4 |
| `batch_size` | 16 | 128 |
| `epochs` | 5 | 10 |
| `learning_rate` | 0.001 | 0.001 |

## Testing

Comprehensive test suites verify correctness:

```bash
python test_Embedding.py       # Embedding layer
python test_CausalSelfAttention.py   # Attention mechanism
python test_TransformerBlock.py      # Residual blocks
python test_GPTModel.py         # Full model integration
```

Tests cover:
- ✓ Tensor shapes (forward & backward)
- ✓ Gradient accumulation
- ✓ Causal masking enforcement
- ✓ Parameter counting
- ✓ Autoregressive generation

## Key Implementation Details

### Causal Masking
```python
scores = scores + mask  # Where mask is -inf for future positions
# After softmax, attention weights are 0 for future positions
```

### Gradient Accumulation (Embedding)
```python
# Multiple occurrences of same token must sum gradients
np.add.at(self.dweight, self.indices, grad_output)
```

### Residual Connections
```python
x1 = x + attention(ln(x))              # Skip connection 1
out = x1 + feedforward(ln(x1))         # Skip connection 2
# Enables: dx = grad_skip + grad_path
```

### Pre-LayerNorm Topology
```python
# Normalize BEFORE each sub-module (not after)
x1 = x + attention(ln(x))
out = x1 + feedforward(ln(x1))
```

## GPU Support

The implementation automatically uses GPU (CuPy) when available via `backend.py`:
- All matrix operations execute on NVIDIA GPUs
- Hyperparameters scale automatically
- Seamless CPU/GPU fallback

## Design Philosophy

- **Manual Backpropagation**: No black-box autograd—every gradient is explicitly computed
- **Batch-First Notation**: Row-vector convention `(B, T, D)` for clarity
- **State Caching**: Forward pass tensors cached for efficient backward pass
- **Mini-Torch Integration**: Builds on existing Linear, ReLU, Sequential, etc.

## Files Overview

| File | Purpose |
|------|---------|
| `Embedding.py` | Token/Positional embeddings lookup table |
| `CausalSelfAttention.py` | Multi-head attention with causal mask |
| `TransformerBlock.py` | Composite block with residual connections |
| `GPTModel.py` | Top-level assembly + generation |
| `LayerNorm.py` | Feature normalization |
| `FeedForward.py` | MLP with 4× expansion |
| `CharTokenizer.py` | Character ↔ Integer mapping |
| `TextCorpusDataset.py` | Sliding-window data loader |
| `train_gpt.py` | Training loop |
| `generate_gpt.py` | Text generation |
| `clean_corpus.py` | Text preprocessing utility |
| `test_*.py` | Unit tests for each module |

## References

- [Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762)
- [Language Models are Unsupervised Multitask Learners (GPT-2)](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- [Pre-Norm vs Post-Norm](https://arxiv.org/abs/2002.07839)


