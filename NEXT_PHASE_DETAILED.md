# 🎯 Next Phase: Training, Experiments & Reporting

This guide walks you through the remaining steps to complete your CSS 590 assignment.

---

## Phase 1: Prepare Your Corpus (15 min)

### Step 1.1: Choose a Corpus

**Option A: Tiny Shakespeare (Recommended for quick testing)**
- Size: ~1 MB
- Vocabulary: ~65 characters
- Training time: 30 min - 2 hours
- Quality: Good for learning

**Option B: Pride and Prejudice (Better for experimentation)**
- Size: ~700 KB
- Vocabulary: ~80 characters  
- Training time: 1-3 hours
- Quality: More coherent output

**Option C: Your Own Text**
- Use any `.txt` file
- Works best if 500KB - 2MB
- Remove metadata first

### Step 1.2: Download & Clean

```bash
# If using Project Gutenberg:
# 1. Download from https://www.gutenberg.org/
# 2. Save as raw.txt

# Clean the corpus
python clean_corpus.py raw.txt cleaned.txt

# Output will show:
# Original vocabulary size: 250
# Cleaned vocabulary size: 65
# Removed characters: [special chars listed]
```

### Step 1.3: Update Paths

Edit `train_gpt.py` and `generate_gpt.py`:

```python
# In train_gpt.py, line ~18:
corpus_path = "PATH_TO_YOUR_CLEANED_CORPUS.txt"  # ← Update this

# In generate_gpt.py, line ~30:
corpus_path = "PATH_TO_YOUR_CLEANED_CORPUS.txt"  # ← Update this
```

---

## Phase 2: Run Tests (10 min)

Before training, verify everything works:

```bash
python test_Embedding.py
python test_CausalSelfAttention.py
python test_TransformerBlock.py
python test_GPTModel.py
```

**Expected output:**
```
✓ test_embedding_forward passed
✓ test_embedding_backward_accumulation passed
✓ test_embedding_parameters_grads passed

✓ All Embedding tests passed!
```

If any test fails, check `IMPLEMENTATION_COMPLETE.md` → Troubleshooting section.

---

## Phase 3: Train Your Model (2-10 hours)

### Step 3.1: Start Training

```bash
python train_gpt.py
```

### Step 3.2: Monitor Training

The output will show:

```
Loading dataset from cleaned.txt...
Dataset loaded. Vocabulary size: 65
Total samples: 45234
Batch size: 16
Training on CPU...

Epoch 1/5
  Batch 100/200 | Loss: 3.4521
  Batch 200/200 | Loss: 2.8934

Epoch 1/5
  Average Loss: 2.8934
  Time: 45.23s

Generated sample:
--------------------------------------------------
qkxpqvzwzlxpzwqlxqplxqzwzlxpzwq
--------------------------------------------------
```

### Step 3.3: Save Important Data

**During training, save:**
1. Loss values from each epoch
2. Sample outputs from each epoch
3. Final weights (`gpt_weights.pkl`)

**Create a folder for results:**
```bash
mkdir -p experiments/baseline
# Copy weights here
cp gpt_weights.pkl experiments/baseline/
```

### Step 3.4: Understanding the Output

- **Loss < 2.0**: Good progress
- **Loss < 1.5**: Excellent convergence
- **Loss not decreasing**: Check learning rate or data

If loss isn't decreasing:
- Try lower learning rate (0.0003 instead of 0.001)
- Try larger batch size (32 instead of 16)
- Check corpus isn't corrupted

---

## Phase 4: Design Your Experiments (1 hour)

### Step 4.1: Choose Research Questions

Pick **1-2 interesting questions** to investigate:

**Example Questions:**
1. "How does context length affect text coherence?"
2. "What's the impact of model depth vs width?"
3. "How sensitive is training to learning rate?"
4. "Can the model capture different writing styles?"

### Step 4.2: Create Experiment Matrix

Edit `train_gpt.py` to create multiple runs:

**Example: Context Length Study**

```python
# Add at top of train_gpt.py:
EXPERIMENTS = {
    'context_16': {
        'block_size': 16,
        'emb_dim': 64,
        'num_layers': 2,
        'epochs': 10,
    },
    'context_32': {
        'block_size': 32,
        'emb_dim': 64,
        'num_layers': 2,
        'epochs': 10,
    },
    'context_64': {
        'block_size': 64,
        'emb_dim': 64,
        'num_layers': 2,
        'epochs': 10,
    },
}

# Then modify main() to iterate:
for exp_name, config in EXPERIMENTS.items():
    print(f"\nRunning experiment: {exp_name}")
    # Merge config with fixed params
    # Train model
    # Save results to experiments/{exp_name}/
```

### Step 4.3: Create Directory Structure

```bash
mkdir -p experiments/{baseline,context_16,context_32,context_64}
mkdir -p results/{plots,samples,metrics}
```

### Step 4.4: Document Your Experiments

Create `experiments.log`:

```
EXPERIMENT 1: Context Length Study
====================================
Date: June 5, 2026

Hypothesis: Longer context (block_size) enables better long-range 
            pattern learning, improving text coherence.

Configs:
  - context_16:  block_size=16
  - context_32:  block_size=32  
  - context_64:  block_size=64
  
Fixed: emb_dim=64, num_layers=2, epochs=10, batch_size=16

Status: Starting training...
```

---

## Phase 5: Collect Metrics (During Training)

### Step 5.1: Modify Training to Save Metrics

Add to `train_gpt.py` in the training loop:

```python
import json

# At top of main():
metrics = {
    'losses': [],
    'epochs': [],
    'samples': {}
}

# In epoch loop:
for epoch in range(epochs):
    total_loss = 0.0
    
    for batch_idx, (x, y) in enumerate(dataloader):
        # ... training code ...
        total_loss += float(asnumpy(loss))
    
    avg_loss = total_loss / num_batches
    metrics['losses'].append(avg_loss)
    metrics['epochs'].append(epoch + 1)
    
    # Save sample
    metrics['samples'][f'epoch_{epoch+1}'] = generated_text
    
    # Print
    print(f"Epoch {epoch + 1}/{epochs} | Loss: {avg_loss:.4f}")

# At end of main():
with open('training_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)
```

### Step 5.2: Record Metrics

For each experiment, track:

```python
# Create results_summary.csv:
experiment,final_loss,perplexity,avg_gen_time,text_quality
baseline,1.89,6.8,250ms,Good
context_16,2.12,8.3,185ms,Fair
context_32,1.98,7.2,310ms,Good
context_64,1.95,7.0,620ms,Good
```

---

## Phase 6: Generate Samples (During & After Training)

### Step 6.1: Save Samples at Each Epoch

Modify `train_gpt.py` to save text:

```python
import os

# Create samples directory
os.makedirs('samples', exist_ok=True)

# In epoch loop, after generation:
with open(f'samples/epoch_{epoch+1}.txt', 'w') as f:
    f.write(generated_text)
    f.write(f"\n\n--- Generated at Epoch {epoch+1} ---\n")
```

### Step 6.2: Generate After Training

```bash
# Generate 500 characters from final model
python generate_gpt.py
```

### Step 6.3: Collect Best Samples

Save samples from:
- Epoch 1 (random garbage)
- Epoch 5 (words forming)
- Epoch 10 (sentences forming)
- Epoch 20 (coherent output)

For your report, you'll compare these!

---

## Phase 7: Create Visualizations (1 hour)

### Step 7.1: Plot Loss Curves

Create `plot_results.py`:

```python
import json
import matplotlib.pyplot as plt
import numpy as np

# Load metrics
with open('training_metrics.json', 'r') as f:
    metrics = json.load(f)

# Create plot
plt.figure(figsize=(12, 6))
plt.plot(metrics['epochs'], metrics['losses'], 'o-', linewidth=2, markersize=6)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Training Loss', fontsize=12)
plt.title('Training Progress: Baseline Model', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()

# SAVE AS PDF (not PNG!)
plt.savefig('results/plots/training_loss.pdf', dpi=300)
print("✓ Saved: results/plots/training_loss.pdf")

plt.show()
```

Run it:
```bash
python plot_results.py
```

### Step 7.2: Compare Experiments

```python
import matplotlib.pyplot as plt

# Plot all experiments
fig, ax = plt.subplots(figsize=(12, 6))

experiments = ['baseline', 'context_16', 'context_32', 'context_64']
colors = ['blue', 'orange', 'green', 'red']

for exp, color in zip(experiments, colors):
    with open(f'experiments/{exp}/training_metrics.json') as f:
        metrics = json.load(f)
    ax.plot(metrics['epochs'], metrics['losses'], 
            label=exp, color=color, linewidth=2, marker='o')

ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('Training Loss', fontsize=12)
ax.set_title('Comparison: Impact of Context Length', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('results/plots/comparison.pdf', dpi=300)
print("✓ Saved: results/plots/comparison.pdf")
```

### Step 7.3: Quality Checklist for Plots

- ✅ Clear axes labels with units
- ✅ Meaningful title
- ✅ Legend identifying lines
- ✅ Grid for easy reading
- ✅ High-quality (300 dpi)
- ✅ Saved as PDF (scalable)
- ✅ Readable in black & white

---

## Phase 8: Write Your Report (4-6 hours)

### Step 8.1: Report Structure

Create `report.md` following this structure:

```markdown
# Character-Level Generative Transformer: [Your Research Question]

## 1. Title and Authors
[Your names, UW affiliations]

### CRediT Author Statement
[See template below]

## 2. Introduction (2-3 pages)
- What is this project about?
- Why is it interesting?
- Your research questions/hypotheses
- Related literature (cite 2-3 papers)

## 3. Methods (3-4 pages)
- Model architecture (diagrams helpful!)
- Hyperparameters (create a table)
- Dataset and preprocessing
- Experiment design
- Metrics used

## 4. Results (3-4 pages)
- Loss curves (PDF plots)
- Perplexity scores (table)
- Generated text samples (from epochs 1, 5, 10, 20)
- Comparison of experiments

## 5. Conclusion (2-3 pages)
- What did you learn?
- Did your hypotheses hold?
- Unexpected findings?
- Future work

## 6. Limitations (1 page)
- What this model can't do
- Corpus size limitations
- Hardware constraints
- Simplifications made

## 7. Ethical Implications (1 page)
- Bias in training data
- Generated content bias
- Environmental impact (GPU)
- Responsible use

## 8. References
[Properly formatted citations]
```

### Step 8.2: Write Introduction Section

**Template:**
```markdown
## 1. Introduction

### Problem Formulation

[Your research question, clearly stated]

### Motivation

[Why is this interesting? What gap does it address?]

### Hypotheses

1. [Hypothesis 1: specific and testable]
2. [Hypothesis 2: specific and testable]

### Related Work

[Cite 2-3 relevant papers:
- Vaswani et al. (2017): Attention mechanism
- Radford et al. (2019): GPT-2 and language modeling
- [Your chosen paper]
]
```

### Step 8.3: Write Methods Section

**Template:**
```markdown
## 2. Methods

### 2.1 Model Architecture

Our implementation consists of:
- Token embeddings: vocab_size × emb_dim
- Positional embeddings: block_size × emb_dim
- [Describe your configuration]

**Table 1: Model Hyperparameters**
| Parameter | Value |
|-----------|-------|
| emb_dim | 64 |
| num_heads | 4 |
| num_layers | 2 |
| block_size | 32 |
| batch_size | 16 |
| learning_rate | 0.001 |

### 2.2 Training Configuration

- Optimizer: RMSprop
- Loss: CrossEntropyLoss
- Epochs: 10
- Hardware: CPU/GPU [specify]

### 2.3 Dataset

- Corpus: [Pride and Prejudice / Tiny Shakespeare]
- Size: [X KB]
- Vocabulary: [Y unique characters]
- Train/test split: 80/20

### 2.4 Experiments

We investigate [your research question] by varying:
- [Experiment 1 design]
- [Experiment 2 design]
```

### Step 8.4: Write Results Section

**Template:**
```markdown
## 3. Results

### 3.1 Training Progress

**Figure 1: Training Loss Over Epochs**
[Include PDF plot here]

*Caption: Training loss decreases from X to Y, showing 
steady convergence over 10 epochs.*

### 3.2 Perplexity Analysis

**Table 2: Final Metrics by Configuration**
| Config | Loss | Perplexity | Time |
|--------|------|-----------|------|
| baseline | 1.89 | 6.8 | 45s/epoch |
| context_16 | 2.12 | 8.3 | 30s/epoch |

### 3.3 Generated Samples

**Epoch 1 (Random initialization):**
```
qkxpqvzwzlxpzwqlxqplxqzwzlxpzwq
```

**Epoch 5 (Learning patterns):**
```
the love and the heart of the world
is the spirit of a man that is not
```

**Epoch 10 (Coherent sentences):**
```
O Lord, if there be beauty in this world,
it is in the hearts of those who truly love
and seek to understand the depths...
```

### 3.4 Experiment Comparison

[Discuss what you found]
```

### Step 8.5: Write Conclusion Section

**Template:**
```markdown
## 5. Conclusion

### Key Findings

[Summarize your main findings]

### Hypothesis Validation

- Hypothesis 1: [CONFIRMED / REFUTED] because...
- Hypothesis 2: [CONFIRMED / REFUTED] because...

### Unexpected Observations

[Did anything surprise you?]

### Future Work

- Larger model (emb_dim=256, num_layers=6)
- Longer sequences (block_size=128)
- Multi-corpus training
- Attention visualization
```

---

## Phase 9: Add CRediT Statement (15 min)

Add to end of your report:

```markdown
## CRediT Author Statement

**Corresponding Author:** [Your Name] ([your.email@uw.edu])

### Author Contributions

**[Your Name]:**
- Conceptualization: Problem formulation, research design
- Methodology: Experiment planning
- Software: Core implementation
- Writing – original draft
- Project administration (corresponding author)

**[Teammate 1 Name] (if applicable):**
- Implementation: Attention mechanism
- Software: Testing and validation
- Data curation: Corpus preparation

**[Teammate 2 Name] (if applicable):**
- Experiments: Running trials, data collection
- Visualization: Creating plots and figures
- Writing – review & editing

### Role Descriptions

- **Conceptualization**: Formulated research questions
- **Methodology**: Designed experimental approach
- **Software**: Wrote and debugged code
- **Testing**: Created and ran test suite
- **Data curation**: Prepared and cleaned corpus
- **Visualization**: Generated plots and figures
- **Writing**: Drafted and revised manuscript
- **Project administration**: Organized timeline, coordinated team
- **Supervision**: Provided oversight and mentorship (corresponding author role)
```

---

## Phase 10: Final Touches (1 hour)

### Checklist Before Submission

**Content:**
- [ ] All research questions answered
- [ ] All hypotheses addressed
- [ ] All metrics reported
- [ ] All samples included
- [ ] Limitations discussed
- [ ] Ethics addressed
- [ ] CRediT statement present
- [ ] Corresponding author identified

**Formatting:**
- [ ] All figures have captions below
- [ ] All tables have captions above
- [ ] All figures/tables numbered (Fig. 1, Table 1, etc.)
- [ ] All axes labeled with units
- [ ] All plots saved as PDFs
- [ ] All code formatted consistently
- [ ] No screenshots (use PDFs!)
- [ ] Professional appearance

**Writing:**
- [ ] Spell-checked
- [ ] Grammar checked
- [ ] Sentences complete
- [ ] Paragraphs coherent
- [ ] Citations formatted consistently
- [ ] References complete

---

## Quick Reference: Commands You'll Need

```bash
# Clean corpus
python clean_corpus.py input.txt cleaned.txt

# Train model
python train_gpt.py

# Generate text
python generate_gpt.py

# Run tests
python test_Embedding.py
python test_CausalSelfAttention.py
python test_TransformerBlock.py
python test_GPTModel.py

# Create plots
python plot_results.py

# LaTeX compile (if using Overleaf)
# See https://overleaf.com/
```

---

## Timeline Suggestion

**Week 1:**
- [ ] Mon: Prepare corpus, run tests (2 hours)
- [ ] Tue-Wed: Train baseline model (6+ hours)
- [ ] Thu: Design experiments, start second run
- [ ] Fri: Finish experimental runs

**Week 2:**
- [ ] Mon: Collect all metrics and samples
- [ ] Tue: Create visualizations (plots, tables)
- [ ] Wed: Write Introduction + Methods (3 hours)
- [ ] Thu: Write Results (2 hours)
- [ ] Fri: Write Conclusion + Ethics + Limitations

**Week 3:**
- [ ] Mon: Final writing polish
- [ ] Tue: Format all figures/tables
- [ ] Wed: Add CRediT statement, references
- [ ] Thu: Proofread and spell-check
- [ ] Fri: Final submission!

---

## Success Metrics

Your report is excellent when:

✨ Research questions are interesting and specific  
✨ Hypotheses are clearly testable  
✨ Methods are reproducible in detail  
✨ Results show clear trends and patterns  
✨ Conclusions interpret findings meaningfully  
✨ Ethics section is thoughtful and detailed  
✨ All figures are high-quality PDFs  
✨ All code is appropriately formatted  
✨ CRediT statement is complete  

---

## Still Need Help?

- **Questions about training?** → See `train_gpt.py` comments
- **Questions about experiments?** → See `EXPERIMENTAL_FRAMEWORK.md`
- **Questions about report structure?** → See section templates above
- **Questions about the model?** → See `README.md`
- **Still stuck?** → Check `IMPLEMENTATION_COMPLETE.md` troubleshooting

---

**You've got this! Follow these steps and you'll have an exemplary report! 🚀✨**
