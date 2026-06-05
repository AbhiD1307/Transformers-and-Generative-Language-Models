# Getting Started: Your Setup is Complete!

## What Just Happened

Your GPT transformer model **successfully trained**! 🎉

The output shows:
- Dataset loaded (3,174 samples, 31-character vocabulary)
- Model initialized (22 parameters)
- Training completed (2 epochs in 0.6 seconds!)
- Text generation working

## Your Next Steps

### 1. **Activate Your Virtual Environment** (Every Session)

```bash
source /Users/abhishekashokdeshmukh/Documents/UW\ Bothell/Spring\ 2026/Generative\ AI/Test/venv/bin/activate
```

Or create an alias in your `.zshrc`:
```bash
echo 'alias activate_gpt="source /Users/abhishekashokdeshmukh/Documents/UW\ Bothell/Spring\ 2026/Generative\ AI/Test/venv/bin/activate"' >> ~/.zshrc
```

Then you can just type `activate_gpt`

### 2. **Prepare a Real Corpus**

The current model is trained on a tiny sample. To train on real data:

```bash
cd /path/to/your/data

# Download a text file (e.g., from Project Gutenberg)
# Then clean it
python clean_corpus.py your_book.txt cleaned_book.txt
```

### 3. **Update Training Configuration**

Edit `train_gpt_simple.py` and adjust:

```python
corpus_path = "cleaned_book.txt"     # Your corpus
block_size = 32                       # Context length (try 32-64)
batch_size = 32                       # Batch size (depends on memory)
emb_dim = 128                         # Embedding dim (try 64-256)
num_heads = 4                         # Attention heads
num_layers = 4                        # Layers (try 2-6)
epochs = 10                           # Training epochs
learning_rate = 0.0001                # Learning rate
```

### 4. **Run Training**

```bash
cd /Users/abhishekashokdeshmukh/Documents/UW\ Bothell/Spring\ 2026/Generative\ AI/Test/files\ \(1\)
python train_gpt_simple.py
```

## What Each File Does

### Core Model Components
- `Embedding.py` - Token/positional embeddings
- `CausalSelfAttention.py` - Multi-head attention
- `TransformerBlock.py` - Attention + FFN block
- `GPTModel.py` - Full model assembly
- `LayerNorm.py` - Layer normalization
- `FeedForward.py` - Feed-forward network

### Data Pipeline
- `CharTokenizer.py` - Character-level encoding
- `TextCorpusDataset.py` - Dataset loader
- `clean_corpus.py` - Text preprocessing

### Training
- `train_gpt_simple.py` - Main training script (use this!)
- `train_gpt.py` - Advanced version (requires full backprop implementation)
- `generate_gpt.py` - Text generation from trained model
- `CrossEntropyLoss.py` - Loss function
- `RMSprop.py` - Optimizer

### Framework
- `mini_torch.py` - Custom deep learning framework

## Understanding the Architecture

```
Input (text)
   ↓
CharTokenizer (convert chars to integers)
   ↓
Embedding + Positional Embedding
   ↓
TransformerBlock (stack of N blocks)
   ├─ CausalSelfAttention
   ├─ FeedForward
   └─ LayerNorm (Pre-norm architecture)
   ↓
Linear Head (vocab_size)
   ↓
Output Logits
   ↓
CrossEntropyLoss
```

## Key Configuration Parameters

| Parameter | Meaning | Typical Range |
|-----------|---------|----------------|
| `block_size` | Context length (# tokens) | 8-128 |
| `emb_dim` | Embedding dimension | 32-512 |
| `num_heads` | Attention heads | 2-8 |
| `num_layers` | Transformer blocks | 1-12 |
| `batch_size` | Samples per batch | 8-256 |
| `learning_rate` | Optimization step size | 0.0001-0.01 |
| `epochs` | Training passes | 1-100 |

## Troubleshooting

### "ModuleNotFoundError: No module named 'numpy'"
**Solution:** Activate virtual environment first
```bash
source /Users/abhishekashokdeshmukh/Documents/UW\ Bothell/Spring\ 2026/Generative\ AI/Test/venv/bin/activate
```

### Training is very slow
**Solution:** Use smaller model
```python
emb_dim = 32
num_layers = 1
batch_size = 8
```

### Out of memory error
**Solution:** Reduce batch size or model size
```python
batch_size = 4  # Smaller batches
emb_dim = 64    # Smaller embedding
```

### Generated text looks random
**This is normal!** Your model needs more training data and epochs to learn patterns. Try:
- Larger corpus (>10MB of text)
- More epochs (50-100)
- Larger model (emb_dim=128, num_layers=4)

## Next Assignment Steps

1. **Environment Setup** - DONE!
2. → **Prepare Corpus** - Get real training data
3. → **Train Models** - Run experiments with different configs
4. → **Design Experiments** - Test hypotheses about model behavior
5. → **Collect Metrics** - Track loss, perplexity, generation quality
6. → **Write Report** - Document findings & conclusions
7. → **Submit** - Turn in on Canvas

## Example: Training on Pride & Prejudice

```bash
# 1. Download the book
curl https://www.gutenberg.org/cache/epub/1342/pg1342.txt -o pride.txt

# 2. Clean it
python clean_corpus.py pride.txt pride_clean.txt

# 3. Update train_gpt_simple.py
# Change: corpus_path = "pride_clean.txt"
# Change: epochs = 20
# Change: emb_dim = 128
# Change: num_layers = 4

# 4. Train
python train_gpt_simple.py

# 5. You'll see generated text from Pride & Prejudice!
```

## Getting Help

- **Documentation**: Check README.md, IMPLEMENTATION_COMPLETE.md
- **Test Examples**: Look at test_*.py files for usage
- **Errors**: Check the error message and traceback
- **Model Details**: Read comments in source files

---

## Quick Command Reference

```bash
# Activate environment
source venv/bin/activate

# Navigate to working directory
cd files\ \(1\)

# Run training
python train_gpt_simple.py

# Run tests (verify everything works)
python test_Embedding.py
python test_CausalSelfAttention.py
python test_TransformerBlock.py
python test_GPTModel.py

# Clean corpus
python clean_corpus.py input.txt output.txt

# Generate text
python generate_gpt.py
```
