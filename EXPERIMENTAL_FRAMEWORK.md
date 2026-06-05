# Experimental Framework & Report Template

This guide helps you design and conduct experiments for your CSS 590 homework report.

---

## Experiment Design Framework

### Step 1: Formulate Research Questions

Choose **1-2 interesting questions** to investigate (beyond simple implementation):

**Example Questions:**
1. *How does context length (block_size) affect text generation quality?*
2. *What is the impact of increasing model depth (num_layers) vs width (emb_dim)?*
3. *How does dropout rate influence generalization on different corpora?*
4. *Can a model trained on Shakespeare successfully generate different genres?*
5. *How sensitive is the model to learning rate and batch size?*
6. *Does causal attention truly prevent future attending?* (verification)

### Step 2: Define Hypotheses

For each question, state a testable hypothesis:

**Example:**
> **Hypothesis 1:** "Increasing context length from 16 to 64 will improve text coherence, but with diminishing returns beyond 32 tokens due to the limited vocabulary size."

> **Hypothesis 2:** "Adding more layers (depth) will improve loss faster than widening embeddings, but only up to 4-6 layers before overfitting dominates."

### Step 3: Design Experiments

Create a controlled experimental matrix:

```python
# Experiment 1: Context Length Study
EXPERIMENTS_CONTEXT = {
    'block_size_16': {'block_size': 16, 'emb_dim': 64, 'num_layers': 2},
    'block_size_32': {'block_size': 32, 'emb_dim': 64, 'num_layers': 2},
    'block_size_64': {'block_size': 64, 'emb_dim': 64, 'num_layers': 2},
}

# Experiment 2: Model Depth vs Width
EXPERIMENTS_CAPACITY = {
    'shallow_wide': {'emb_dim': 128, 'num_layers': 1},
    'balanced': {'emb_dim': 64, 'num_layers': 4},
    'deep_narrow': {'emb_dim': 32, 'num_layers': 8},
}

# Keep everything else constant
FIXED_CONFIG = {
    'batch_size': 16,
    'learning_rate': 0.001,
    'epochs': 15,
    'corpus': 'Pride-and-Prejudice-Cleaned.txt',
}
```

---

## Metrics to Track

### 1. **Training Loss**
```python
# Log per-batch and per-epoch
avg_loss = total_loss / num_batches
# Plot: Loss vs Epoch (compare experiments)
```

### 2. **Validation/Test Loss**
- Split data: 80% train / 20% test
- Compute loss on held-out test set
- Indicates overfitting if gap widens

### 3. **Perplexity**
```python
# Perplexity = exp(loss)
perplexity = np.exp(avg_loss)
# Lower is better; absolute comparison across models
```

### 4. **Qualitative Text Quality**
- Save generated samples every N epochs
- Compare:
  - Coherence (do words make sense together?)
  - Grammar (correct punctuation/capitalization?)
  - Diversity (creativity vs repetition?)
  - Corpus fidelity (mimics training data style?)

### 5. **Generation Efficiency**
- Time-to-generate (milliseconds per token)
- Tokens per second on GPU/CPU

---

## Experiment Tracking Template

### Experiment Template:

```python
"""
EXPERIMENT: Context Length Impact on Text Quality
================================================

HYPOTHESIS:
    Longer context (block_size) enables the model to capture longer-range 
    dependencies, improving coherence and diversity in generated text, 
    with diminishing returns beyond 32 tokens.

METHODOLOGY:
    - Fixed: emb_dim=64, num_layers=2, epochs=20, batch_size=16
    - Variable: block_size ∈ {8, 16, 32, 64}
    - Corpus: Pride-and-Prejudice-Cleaned.txt
    - Metrics: Training loss, validation loss, perplexity, sample diversity

CONFIGURATION:
    block_size: [8, 16, 32, 64]
    emb_dim: 64
    num_heads: 4
    num_layers: 2
    batch_size: 16
    learning_rate: 0.001
    epochs: 20

RESULTS:
    
    | block_size | Final Loss | Perplexity | Avg Gen. Time | Quality |
    |-----------|-----------|-----------|--------|----------|
    | 8         | 2.45      | 11.6      | 125ms  | Poor     |
    | 16        | 2.12      | 8.3       | 185ms  | Fair     |
    | 32        | 1.98      | 7.2       | 310ms  | Good     |
    | 64        | 1.95      | 7.0       | 620ms  | Good     |
    
    Loss curves: [PLOT]
    Generated samples: [TEXT SAMPLES]

CONCLUSIONS:
    - Increasing block_size from 8→32 significantly improves loss (2.45 → 1.98)
    - Diminishing returns: 32→64 improvement minimal (1.98 → 1.95)
    - Sweet spot: block_size=32 balances quality and efficiency
    - Hypothesis CONFIRMED with caveats

LIMITATIONS:
    - Small corpus limits long-range pattern learning
    - Generation speed trade-off not fully explored
    - Could benefit from validation set testing
"""
```

---

## Report Sections Checklist

### ✅ Title & Authors
- [ ] Clear, descriptive title
- [ ] All group members listed
- [ ] Corresponding author identified (with supervision/admin role)

### ✅ Introduction (10 pts)
- [ ] Hook: Why is this interesting?
- [ ] Background: What are transformers? Why text generation?
- [ ] Literature: 2-3 relevant papers cited (Vaswani et al., GPT-2, etc.)
- [ ] Research questions clearly stated
- [ ] Hypotheses explicitly listed

### ✅ Methods (20 pts)
- [ ] Architecture description:
  - [ ] Model size variations (emb_dim, num_layers, num_heads)
  - [ ] Parameter counts
  - [ ] Comparison to baselines (if any)
- [ ] Data:
  - [ ] Corpus name and characteristics
  - [ ] Preprocessing steps
  - [ ] Vocabulary size
  - [ ] Train/val/test split
- [ ] Training:
  - [ ] Hyperparameters (LR, batch size, optimizer)
  - [ ] Loss function
  - [ ] Training duration / epochs
  - [ ] Hardware (GPU model, CPU specs)
- [ ] Evaluation:
  - [ ] Metrics used
  - [ ] Generation strategy (temperature, beam search, etc.)
- [ ] Experiments:
  - [ ] Exact configurations for each variant
  - [ ] Why these configurations?

### ✅ Results (30 pts)
- [ ] Loss curves (training + validation)
  - [ ] Formatted matplotlib plots, PDFs
  - [ ] Labeled axes with units
  - [ ] Legend identifying runs
- [ ] Perplexity scores
  - [ ] Table format with precision
  - [ ] Ranked by performance
- [ ] Generated text samples
  - [ ] Epoch 1 sample (shows random start)
  - [ ] Epoch 5 sample
  - [ ] Epoch 20 sample (final)
  - [ ] Samples from different configurations
  - [ ] Format: Block quote or indented code
- [ ] Beyond-standard metrics:
  - [ ] Generation diversity scores (if computed)
  - [ ] Attention weight visualizations (bonus)
  - [ ] Token frequency analysis
  - [ ] Failure case analysis

### ✅ Conclusion (10 pts)
- [ ] Interpret results in light of hypotheses
- [ ] Did hypotheses hold?
- [ ] Explain deviations from expectations
- [ ] Key insights learned
- [ ] Future work directions

### ✅ Limitations (2-3 pts)
- [ ] Corpus size/quality limitations
- [ ] Model size constraints
- [ ] Hardware limitations
- [ ] Tokenization simplifications
- [ ] What this model is NOT good at

### ✅ Ethics & Bias (2-3 pts)
- [ ] Training data bias (does corpus have inherent bias?)
- [ ] Generated text bias (does model perpetuate it?)
- [ ] Environmental impact (GPU compute cost)
- [ ] Responsible use considerations
- [ ] Transparency about limitations

### ✅ CRediT Statement (Required!)
```
## CRediT Author Statement

**Corresponding Author:** Alice Johnson (alice@uw.edu)

**Author Contributions:**
- **Alice Johnson**: Conceptualization, Methodology (attention mechanism), 
  Writing – original draft, Supervision
- **Bob Smith**: Implementation (embeddings, transformer blocks), 
  Software, Testing
- **Carol White**: Experiments, Data curation, Visualization
- **Dave Brown**: Writing – review & editing, Project administration

**Role Descriptions:**
- Supervision: Organized team meetings, ensured quality
- Project administration: Coordinated timeline, managed GitHub
- Conceptualization: Designed research questions
- Methodology: Defined technical approach
- Implementation: Wrote core code
- Software: Testing & validation
- Data curation: Corpus cleaning and preprocessing
- Visualization: Generated plots and figures
- Writing: Draft, review, and final editing
```

---

## Sample Experiment Workflow

```python
# ============================================================================
# experiment_context_length.py - Example: Vary block_size
# ============================================================================

import numpy as np
import matplotlib.pyplot as plt
from train_gpt import main as train_gpt

CONFIGURATIONS = [
    {'name': 'context_8',  'block_size': 8,  'emb_dim': 64, 'num_layers': 2},
    {'name': 'context_16', 'block_size': 16, 'emb_dim': 64, 'num_layers': 2},
    {'name': 'context_32', 'block_size': 32, 'emb_dim': 64, 'num_layers': 2},
    {'name': 'context_64', 'block_size': 64, 'emb_dim': 64, 'num_layers': 2},
]

FIXED = {
    'batch_size': 16,
    'learning_rate': 0.001,
    'epochs': 15,
    'corpus_path': '../../Transformers/Pride-and-Prejudice-Cleaned.txt',
}

results = {}

for config in CONFIGURATIONS:
    print(f"\n{'='*60}")
    print(f"Running: {config['name']}")
    print(f"{'='*60}")
    
    # Merge config with fixed params
    full_config = {**FIXED, **config}
    
    # Train model (modify train_gpt.py to return history)
    losses, model = train_gpt(**full_config)
    
    # Store results
    results[config['name']] = {
        'losses': losses,
        'config': config,
        'perplexity': [np.exp(l) for l in losses],
    }

# Visualization
plt.figure(figsize=(12, 6))

for name, result in results.items():
    plt.plot(result['losses'], label=name, linewidth=2)

plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Training Loss', fontsize=12)
plt.title('Impact of Context Length on Training Loss', fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('experiment_context_length.pdf')  # Save as PDF for report
plt.show()

# Print summary table
print("\n" + "="*60)
print("RESULTS SUMMARY")
print("="*60)
print(f"{'Config':<15} {'Final Loss':<12} {'Perplexity':<12}")
print("-"*60)
for name, result in results.items():
    final_loss = result['losses'][-1]
    final_ppl = result['perplexity'][-1]
    print(f"{name:<15} {final_loss:<12.4f} {final_ppl:<12.2f}")
```

---

## Visualization Best Practices

### ✅ DO:
- Use consistent colors across related plots
- Include grid lines for easy reading
- Label axes with units (Loss, Epoch, etc.)
- Use high-contrast colors (not red/green for colorblind)
- Save plots as PDF (scalable, embedded cleanly)
- Add figure captions below plots

### ✅ DON'T:
- Use default matplotlib colors (too pale)
- Plot without legends
- Have overlapping lines (use distinct line styles)
- Mix log and linear scales without clarity
- Save as PNG/JPG (lossy, pixelated)
- Use decorative 3D plots (hard to read)

### Example:
```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(epochs, train_loss, 'o-', label='Train Loss', linewidth=2, markersize=4)
ax.plot(epochs, val_loss, 's--', label='Val Loss', linewidth=2, markersize=4)

ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('Loss', fontsize=12)
ax.set_title('Training Progress', fontsize=14, fontweight='bold')
ax.legend(fontsize=11, loc='upper right')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_loss.pdf', dpi=300)  # PDF for embedding
```

---

## Text Sample Presentation

**DO:**
```
### Generated Samples Over Training

**Epoch 1:**
```
qkxpqvzwzlxpzwqlxqplxqzwzlxpzwq
```

**Epoch 10:**
```
the love of all the world is the heart
of a man that is not a man.
```

**Epoch 20:**
```
O Lord, if there be beauty in this world,
it is in the hearts of those who love.
```
```

**DON'T:**
- Paste raw text without formatting
- Show 50 lines of garbage from epoch 1
- No labels on which epoch/config generated text

---

## Common Mistakes to Avoid

❌ **Not tracking hyperparameters** → Can't reproduce results  
✅ Fix: Save config dict for each experiment

❌ **Training only 1 epoch** → No loss curve to show  
✅ Fix: Train 10-20 epochs minimum

❌ **No validation set** → Can't detect overfitting  
✅ Fix: Split corpus 80/20 or 90/10

❌ **Poor text samples** → Doesn't support claims  
✅ Fix: Save samples at regular intervals

❌ **Missing axis labels** → Unreadable plots  
✅ Fix: Use `set_xlabel()`, `set_ylabel()`, `set_title()`

❌ **No CRediT statement** → Report won't be graded!  
✅ Fix: Include required statement at end

---

## Submission Checklist

- [ ] All implementations complete and tested
- [ ] Report written in LaTeX or professional markdown
- [ ] All figures embedded as PDFs (not screenshots)
- [ ] All tables properly formatted with captions
- [ ] Experiments documented and reproducible
- [ ] Results section includes loss curves + samples
- [ ] Limitations section present
- [ ] Ethics/bias section present
- [ ] CRediT statement included with corresponding author
- [ ] Code uploaded to private repository (if large)
- [ ] References formatted consistently

---

**You're ready to produce an exemplary report!** 📊📄
