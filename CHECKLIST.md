# Complete Step-by-Step Checklist

Use this checklist to track your progress through the assignment.

---

## Phase 1: Setup (Day 1)

### Prepare Corpus
- [ ] Choose a corpus (Tiny Shakespeare, Pride & Prejudice, or your own)
- [ ] Download the text file
- [ ] Clean the corpus: `python clean_corpus.py input.txt cleaned.txt`
- [ ] Verify cleaned file is readable
- [ ] Note vocabulary size (should be ~50-100 characters)

### Verify Installation
- [ ] Run all tests: `python test_*.py`
- [ ] All tests show ✓ marks
- [ ] No import errors

### Configure Paths
- [ ] Edit `train_gpt.py` - set `corpus_path`
- [ ] Edit `generate_gpt.py` - set `corpus_path`
- [ ] Create directories:
  ```bash
  mkdir -p experiments/{baseline,exp1,exp2}
  mkdir -p results/{plots,samples,metrics}
  ```

**Status:** [ ] Phase 1 Complete

---

## Phase 2: Baseline Training (Days 2-3)

### Run Baseline Training
- [ ] Start: `python train_gpt.py`
- [ ] Monitor output for convergence
- [ ] Training loss should decrease over time
- [ ] Save output log to file: `python train_gpt.py 2>&1 | tee training.log`

### Collect Results
- [ ] Weights saved: `gpt_weights.pkl` exists
- [ ] Copy weights: `cp gpt_weights.pkl experiments/baseline/`
- [ ] Save metrics: Create `experiments/baseline/metrics.json` with:
  - `epochs`: [1, 2, 3, ...]
  - `losses`: [initial_loss, ..., final_loss]
  - `samples`: {epoch_1: "...", epoch_5: "...", ...}
- [ ] Save samples to files:
  - `experiments/baseline/epoch_1_sample.txt`
  - `experiments/baseline/epoch_5_sample.txt`
  - `experiments/baseline/epoch_10_sample.txt`

### Test Generation
- [ ] Run: `python generate_gpt.py`
- [ ] Read output text
- [ ] Is it better than random? (Yes = good!)

**Status:** [ ] Phase 2 Complete

---

## Phase 3: Experiment Design (Day 4)

### Choose Research Question
- [ ] Question 1: [Write it down]
- [ ] Question 2 (optional): [Write it down]
- [ ] Both questions are specific and testable

### Design Experiments
- [ ] **Experiment 1 config:**
  - block_size: __
  - emb_dim: __
  - num_heads: __
  - num_layers: __
  - epochs: __

- [ ] **Experiment 2 config:**
  - block_size: __
  - emb_dim: __
  - num_heads: __
  - num_layers: __
  - epochs: __

### Create Tracking Files
- [ ] `experiments.log` created with experiment descriptions
- [ ] For each experiment, create: `experiments/{name}/config.json`

**Status:** [ ] Phase 3 Complete

---

## Phase 4: Run Experiments (Days 5-6)

### Experiment 1
- [ ] Modify `train_gpt.py` config for Experiment 1
- [ ] Run: `python train_gpt.py 2>&1 | tee experiments/exp1/training.log`
- [ ] Weights saved to: `experiments/exp1/gpt_weights.pkl`
- [ ] Metrics saved to: `experiments/exp1/metrics.json`
- [ ] Samples saved to: `experiments/exp1/epoch_*.txt`

### Experiment 2 (if applicable)
- [ ] Modify `train_gpt.py` config for Experiment 2
- [ ] Run: `python train_gpt.py 2>&1 | tee experiments/exp2/training.log`
- [ ] Weights saved to: `experiments/exp2/gpt_weights.pkl`
- [ ] Metrics saved to: `experiments/exp2/metrics.json`
- [ ] Samples saved to: `experiments/exp2/epoch_*.txt`

### Organize Results
- [ ] All metrics collected in JSON format
- [ ] All samples saved as text files
- [ ] Create spreadsheet with results:

| Experiment | Final Loss | Perplexity | Best Sample |
|---|---|---|---|
| baseline | | | |
| exp1 | | | |
| exp2 | | | |

**Status:** [ ] Phase 4 Complete

---

## Phase 5: Create Visualizations (Day 7)

### Generate Plots
- [ ] Install matplotlib: `pip install matplotlib`
- [ ] Create single experiment plots:
  ```python
  from plotting_helper import plot_single_experiment
  plot_single_experiment('baseline')
  plot_single_experiment('exp1')
  ```

- [ ] Create comparison plot:
  ```python
  from plotting_helper import plot_comparison
  plot_comparison(['baseline', 'exp1', 'exp2'])
  ```

### Verify Plot Quality
- [ ] All plots saved as PDF (not PNG!)
- [ ] All plots have:
  - [ ] Clear title
  - [ ] Labeled axes with units
  - [ ] Legend (if multiple lines)
  - [ ] Grid for readability
  - [ ] High resolution (300 dpi)

### Create Results Table
- [ ] Generate markdown table:
  ```python
  from plotting_helper import create_table_for_report
  create_table_for_report(['baseline', 'exp1', 'exp2'])
  ```

- [ ] Copy table to `results/metrics_summary.csv`

**Status:** [ ] Phase 5 Complete

---

## Phase 6: Write Report (Days 8-10)

### Setup
- [ ] Choose format: LaTeX (Overleaf) or Markdown
- [ ] Create document structure
- [ ] Create `report.tex` or `report.md`

### 1. Title & Authors Section
- [ ] Title (clear, descriptive)
- [ ] Author names
- [ ] Author affiliations
- [ ] **CRediT statement** with:
  - [ ] All authors listed
  - [ ] Roles clearly described
  - [ ] Corresponding author identified

### 2. Introduction (500-750 words)
- [ ] Hook: Why is this interesting?
- [ ] Background on transformers (2-3 sentences)
- [ ] Your research questions (clearly stated)
- [ ] Hypotheses (testable, specific)
- [ ] Literature review (cite 2-3 papers):
  - [ ] Vaswani et al. (2017) - Transformers
  - [ ] Radford et al. (2019) - GPT-2
  - [ ] [Your chosen paper]

### 3. Methods (750-1000 words)
- [ ] Model architecture description
- [ ] **Table 1: Hyperparameters**
  - [ ] emb_dim, num_heads, num_layers, block_size
  - [ ] batch_size, learning_rate, epochs
  - [ ] Hardware (CPU/GPU)
  
- [ ] Dataset description
  - [ ] Corpus name and source
  - [ ] Size (KB)
  - [ ] Vocabulary size
  
- [ ] Experiment design
  - [ ] What varied between experiments
  - [ ] Why those choices
  - [ ] How results were evaluated

### 4. Results (750-1000 words)
- [ ] **Figure 1: Training Loss Curves**
  - [ ] PDF plot included
  - [ ] Caption below figure
  - [ ] Reference in text: "As shown in Figure 1..."
  
- [ ] **Table 2: Metrics Summary**
  - [ ] Experiment name, final loss, perplexity
  - [ ] Caption above table
  - [ ] Reference in text: "Table 2 shows..."
  
- [ ] **Generated Text Samples**
  - [ ] Epoch 1 sample (random)
  - [ ] Epoch 5 sample (learning)
  - [ ] Epoch 10 sample (coherent)
  - [ ] Comparison across experiments
  
- [ ] Discussion of results
  - [ ] Key findings
  - [ ] Trends observed
  - [ ] Unexpected results

### 5. Conclusion (500-750 words)
- [ ] Summary of findings
- [ ] Hypothesis validation:
  - [ ] Hypothesis 1: Confirmed/Refuted/Partially
  - [ ] Hypothesis 2: Confirmed/Refuted/Partially
  
- [ ] Interpretation of results
- [ ] Connection to literature
- [ ] Surprising observations (if any)
- [ ] Future work suggestions

### 6. Limitations (250-500 words)
- [ ] Corpus size limitations
- [ ] Model capacity constraints
- [ ] Hardware limitations
- [ ] Simplifications made
- [ ] What this model can't do

### 7. Ethical Implications (250-500 words)
- [ ] Bias in training data
- [ ] Generated content bias
- [ ] Environmental impact (GPU use)
- [ ] Responsible AI considerations
- [ ] Limitations of character-level models

### 8. References
- [ ] All citations included
- [ ] Proper formatting (APA, Chicago, or IEEE)
- [ ] Complete author names
- [ ] Publication years

**Status:** [ ] Phase 6 Complete

---

## Phase 7: Format & Polish (Days 11-12)

### Figure Quality
- [ ] All figures saved as PDF
- [ ] All figures have captions below
- [ ] All figures numbered (Figure 1, Figure 2, ...)
- [ ] All figures referenced in text
- [ ] Figure captions are descriptive (2-3 sentences)

### Table Quality
- [ ] All tables have captions above
- [ ] All tables numbered (Table 1, Table 2, ...)
- [ ] All tables referenced in text
- [ ] Table captions are descriptive
- [ ] Clear column headers
- [ ] Consistent formatting

### Code Formatting
- [ ] Code snippets properly formatted
- [ ] Code has light gray background
- [ ] Code is indented consistently
- [ ] Short code (<1 page) inline
- [ ] Long code in appendix

### Writing Quality
- [ ] Spell-checked (no red squiggles)
- [ ] Grammar checked (no underlines)
- [ ] All sentences complete
- [ ] Paragraphs flow logically
- [ ] Active voice where possible
- [ ] Past tense for methods & results

### Organization
- [ ] Clear section headings
- [ ] Consistent heading levels
- [ ] Page numbers (if applicable)
- [ ] Table of contents (if applicable)
- [ ] No orphaned headings

**Status:** [ ] Phase 7 Complete

---

## Phase 8: Final Review (Day 13)

### Content Verification
- [ ] All research questions answered
- [ ] All hypotheses addressed
- [ ] All experiments described
- [ ] All metrics reported
- [ ] All samples included
- [ ] All limitations discussed
- [ ] All ethics considered
- [ ] All contributions attributed (CRediT)

### Format Verification
- [ ] No screenshots (use PDFs!)
- [ ] All figures in PDF format
- [ ] All axes labeled
- [ ] All units specified
- [ ] Consistent font choices
- [ ] Consistent color scheme
- [ ] Professional appearance

### Content Review
- [ ] Read through once for flow
- [ ] Read through again for errors
- [ ] Have someone else read it
- [ ] Incorporate feedback
- [ ] Final proofread

### CRediT Verification
- [ ] All authors included
- [ ] All roles clearly stated
- [ ] Corresponding author identified
- [ ] Roles match actual contributions

**Status:** [ ] Phase 8 Complete

---

## Phase 9: Submit! (Day 14)

### Pre-Submission
- [ ] Save all files
- [ ] Create final PDF
- [ ] Test PDF opens correctly
- [ ] All figures visible
- [ ] All text readable

### Submission
- [ ] Note submission deadline
- [ ] Upload to Canvas/system
- [ ] Confirm receipt
- [ ] Save confirmation email

### Archive
- [ ] Save copy of submitted report
- [ ] Save all experiment data
- [ ] Create summary of experiments
- [ ] Document any issues encountered

**Status:** [ ] SUBMITTED! 

---

## Quick Summary by Phase

| Phase | Tasks | Days | Status |
|-------|-------|------|--------|
| 1 | Setup, corpus prep, verify tests | 1 | [ ] |
| 2 | Baseline training, collect results | 2-3 | [ ] |
| 3 | Design experiments, plan research | 1 | [ ] |
| 4 | Run experiments, collect metrics | 2 | [ ] |
| 5 | Create visualizations, tables | 1 | [ ] |
| 6 | Write all report sections | 3 | [ ] |
| 7 | Format, polish, quality check | 2 | [ ] |
| 8 | Final review before submission | 1 | [ ] |
| 9 | Submit! | 1 | [ ] |

**Total: ~14 days to completion**

---

## Emergency Checklist

If you're running low on time:

**Absolute Minimum (48 hours):**
- [ ] Run one training (baseline)
- [ ] Create one plot
- [ ] Write concise Methods & Results
- [ ] Include samples
- [ ] Add CRediT statement

**Better (1 week):**
- [ ] Run baseline + 1 experiment
- [ ] Create comparison plot
- [ ] Write full Methods & Results
- [ ] Include limitations & ethics
- [ ] Complete CRediT statement

**Excellent (2 weeks):**
- [ ] Run baseline + 2 experiments
- [ ] Multiple plots & tables
- [ ] Full report with Introduction, Conclusion
- [ ] Thoughtful limitations & ethics
- [ ] Professional formatting throughout

---

## Support Resources

**Stuck on something?**

- Model questions → `README.md`
- Implementation questions → `IMPLEMENTATION_COMPLETE.md`
- Experiment questions → `EXPERIMENTAL_FRAMEWORK.md`
- Report questions → `NEXT_PHASE_DETAILED.md` (this file!)
- Code questions → See comments in the `.py` files

---
