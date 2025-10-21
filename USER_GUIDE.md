# Complete User Guide - ML Drug Screening Pipeline

## What This Software Does

This pipeline helps you:
1. **Find active compounds** for a target protein (like drugs that bind to a receptor)
2. **Train machine learning models** to predict which molecules are active
3. **Screen new compounds** to find potential drug candidates

Think of it as an automated drug discovery assistant!

---

## Setup (One-Time)

### Step 1: Install Python

**Check if you have Python:**
```bash
python --version
```

If you see "Python 3.7" or higher, you're good! If not:
- **Windows**: Download from https://python.org/downloads
- **Mac**: Usually pre-installed, or use `brew install python3`
- **Linux**: Usually pre-installed, or `sudo apt install python3`

### Step 2: Get the Code

**Easy Way (Download ZIP):**
1. Go to https://github.com/Abbas24-AI/ChatbotAI
2. Click green "Code" button → "Download ZIP"
3. Extract to a folder like `C:\Users\YourName\ChatbotAI`
4. Open terminal/command prompt in that folder

**Advanced Way (Git):**
```bash
git clone https://github.com/Abbas24-AI/ChatbotAI.git
cd ChatbotAI
git checkout claude/ml-drug-screening-pipeline-011CULAEpHnvCcUF77BiSCnh
```

### Step 3: Test It Works

Run the quick test (5-10 minutes):
```bash
python test_pipeline.py
```

If you see "TEST COMPLETED SUCCESSFULLY", you're ready!

---

## Three Ways to Use This Pipeline

### Option 1: Interactive Mode (Easiest - Recommended for Beginners)

Just run:
```bash
python ml_drug_screening.py
```

Then answer the questions:
```
Enter the target protein name: EGFR
Select target (1-3): 1
Do you want to perform virtual screening? no
```

**What happens:**
- Downloads data for your target protein
- Trains 6 ML models
- Shows you which model is best
- Saves everything for later use

**Time:** 30-60 minutes depending on data size

**Where are results?**
- `ml_screening_results/` - All data and summaries
- `trained_models/` - Your trained AI models
- `figures/` - Charts and graphs (open with any image viewer)

---

### Option 2: Quick Test Mode (Fastest - 5-10 Minutes)

For testing or when you're in a hurry:
```bash
python test_pipeline.py
```

Uses faster settings, good for:
- Testing the software works
- Quick experiments
- Learning how it works

Results go to `test_results/`, `test_models/`, `test_figures/`

---

### Option 3: Screen Your Compounds (After Training)

**First time:** Run Option 1 to train a model

**Then:** Screen your own molecules:

1. Edit `screen_my_compounds.py`
2. Add your SMILES on line 22:
   ```python
   your_smiles = [
       'YOUR_SMILES_HERE',
       'ANOTHER_SMILES_HERE',
   ]
   ```
3. Run:
   ```bash
   python screen_my_compounds.py
   ```

**Output:** Shows which compounds might be active!

---

## Understanding SMILES (Molecule Notation)

SMILES is a text way to write molecular structures:

- `CC(=O)O` = Acetic acid
- `c1ccccc1` = Benzene
- `CC(C)Cc1ccc(cc1)C(C)C(O)=O` = Ibuprofen

**Where to get SMILES:**
- PubChem: https://pubchem.ncbi.nlm.nih.gov/
- ChemSpider: http://www.chemspider.com/
- Draw molecule online and convert: https://cactus.nci.nih.gov/translate/

---

## Popular Targets to Try

Copy-paste these into the pipeline:

| Target | Disease Area | Good for Beginners? |
|--------|--------------|---------------------|
| `Acetylcholinesterase` | Alzheimer's | ✅ Yes - Good data |
| `EGFR` | Cancer | ✅ Yes - Lots of data |
| `COX-2` | Inflammation | ✅ Yes - Common target |
| `Dopamine D2 receptor` | Neurological | ✅ Yes - Well-studied |
| `BACE1` | Alzheimer's | ⚠️ Medium data |
| `HIV-1 protease` | Antiviral | ✅ Yes - Good data |

---

## What Each File Does

### Scripts You Run:

| File | What It Does | When to Use |
|------|-------------|-------------|
| `ml_drug_screening.py` | Main pipeline | First run, full analysis |
| `test_pipeline.py` | Quick test | Test installation |
| `screen_my_compounds.py` | Screen your molecules | After training |
| `example_usage.py` | 7 different examples | Learn advanced features |

### Results:

| Folder/File | What's Inside |
|-------------|---------------|
| `ml_screening_results/` | All your analysis results |
| `trained_models/` | AI models (can reuse later) |
| `figures/` | Charts and graphs (PNG images) |
| `training_summary.txt` | Model performance report |

---

## Reading Your Results

### 1. Check Model Performance

Open `ml_screening_results/training_summary.txt`

Look for:
```
RandomForest
----------------------------------------
  roc_auc_mean: 0.8542
  accuracy_mean: 0.7831
```

**What this means:**
- `roc_auc_mean` > 0.8 = Good model ✅
- `roc_auc_mean` > 0.9 = Excellent model 🌟
- `roc_auc_mean` < 0.7 = Poor model ❌

### 2. Look at the Figures

Open files in `figures/` folder:

**roc_curves.png:**
- Higher curve = better model
- Perfect model = curve in top-left corner
- Random guessing = diagonal line

**confusion_matrices.png:**
- Top-left + Bottom-right (green) = Correct predictions ✅
- Top-right + Bottom-left (red) = Wrong predictions ❌

**shap_summary_*.png:**
- Shows which molecular features are important
- Red = high feature value
- Blue = low feature value
- Left = decreases activity
- Right = increases activity

### 3. Check Screening Results

If you ran virtual screening, open `screening_results.csv`:

| Column | Meaning |
|--------|---------|
| `compound_id` | Your compound name |
| `smiles` | Molecular structure |
| `predicted_probability` | 0.0 to 1.0 (higher = more likely active) |
| `predicted_label` | "Active" or "Inactive" |

**Hits:** Compounds with `predicted_probability > 0.7`

---

## Customizing Settings

Edit the settings by creating a custom configuration:

```python
from ml_drug_screening import PipelineConfig

# For SPEED (faster, less accurate):
config = PipelineConfig(
    n_folds=3,                      # Fewer folds
    models_to_train=['rf', 'xgb'],  # Only 2 models
    hyperparam_search='none',       # Skip optimization
)

# For ACCURACY (slower, more accurate):
config = PipelineConfig(
    n_folds=10,                     # More folds
    hyperparam_iterations=100,      # More optimization
    ensemble_top_n=5,               # More models in ensemble
)

# For DIFFERENT THRESHOLD (what counts as "active"):
config = PipelineConfig(
    activity_threshold_nm=100,      # 100 nM instead of 1000 nM
    screening_threshold=0.8,        # Higher confidence for hits
)
```

---

## Troubleshooting

### "It's taking forever!"

**Normal!** First run takes 30-60 minutes because:
1. Downloading data (2-5 min)
2. Calculating features (5-10 min)
3. Training models (10-30 min)
4. Optimization (10-20 min)

**Speed it up:**
- Use `test_pipeline.py` for quick tests
- Or reduce `n_folds=3` and `models_to_train=['rf']`

### "No module named 'rdkit'"

**Solution:**
```bash
pip install rdkit-pypi
```

Or with conda:
```bash
conda install -c conda-forge rdkit
```

### "ChEMBL connection timeout"

**Reasons:**
1. Internet connection issue
2. ChEMBL server is busy
3. Firewall blocking access

**Solution:**
- Wait a minute and try again
- Check internet connection
- Try a different network (not corporate firewall)

### "Memory Error"

Your computer ran out of RAM.

**Solutions:**
1. Close other programs
2. Use fewer descriptors: `use_rdkit_descriptors=False`
3. Smaller fingerprints: `morgan_bits=1024`
4. Process in batches: `batch_size=5000`

### "Python not found"

**Windows:**
```bash
# Try python3 instead
python3 ml_drug_screening.py

# Or add python to PATH
```

**Mac/Linux:**
```bash
python3 ml_drug_screening.py
```

---

## Example Workflow (Complete Beginner)

### Day 1: Initial Training

1. Open terminal in ChatbotAI folder
2. Run: `python test_pipeline.py`
3. Wait 5-10 minutes
4. If successful, run: `python ml_drug_screening.py`
5. Enter: `Acetylcholinesterase`
6. Select option: `1`
7. Virtual screening: `no`
8. Wait 30-60 minutes
9. Check `figures/` for results!

### Day 2: Screen Your Compounds

1. Get SMILES for your molecules (from PubChem)
2. Edit `screen_my_compounds.py`
3. Add your SMILES to the list
4. Run: `python screen_my_compounds.py`
5. Check `my_screening_results.csv`
6. Look at `predicted_probability` > 0.7 for hits!

---

## Getting SMILES for Your Molecules

### Method 1: PubChem (Easiest)

1. Go to https://pubchem.ncbi.nlm.nih.gov/
2. Search for your compound name (e.g., "aspirin")
3. Click on the result
4. Find "Canonical SMILES" section
5. Copy the SMILES string

### Method 2: Draw It

1. Go to https://cactus.nci.nih.gov/translate/
2. Draw your molecule
3. Select "SMILES" as output
4. Click "Translate"
5. Copy the SMILES

### Method 3: From ChemDraw

If you have ChemDraw:
1. Draw your molecule
2. Edit → Copy As → SMILES
3. Paste into your script

---

## Tips for Best Results

### ✅ DO:
- Start with well-studied targets (EGFR, Acetylcholinesterase)
- Use the test script first to verify installation
- Save your results after each run
- Look at multiple metrics, not just accuracy
- Check the figures to understand your models

### ❌ DON'T:
- Skip the test run
- Use very obscure targets with little data
- Trust predictions with probability < 0.5
- Expect 100% accuracy (drug discovery is complex!)
- Run on a computer with < 4 GB RAM

---

## What If I Want to...

**...use a different protein?**
→ Just enter a different name when prompted

**...screen 10,000 compounds?**
→ Use a CSV file with `screen_my_compounds.py`

**...make it faster?**
→ Use `test_pipeline.py` or reduce `n_folds=3`

**...make it more accurate?**
→ Increase `n_folds=10`, `hyperparam_iterations=100`

**...use different models?**
→ Set `models_to_train=['rf', 'xgb', 'lgb', 'svm']`

**...share my results?**
→ Send the `ml_screening_results/` folder

**...use the model later?**
→ It's saved in `trained_models/ensemble_model.pkl`

**...understand what features matter?**
→ Look at `shap_summary_*.png` figures

---

## Need Help?

1. **Check documentation:**
   - This guide (USER_GUIDE.md)
   - QUICKSTART.md
   - ML_DRUG_SCREENING_README.md

2. **Try the examples:**
   ```bash
   python example_usage.py
   ```

3. **Check common errors above**

4. **Still stuck?** Open an issue on GitHub

---

## Summary Cheat Sheet

```bash
# First time setup
python test_pipeline.py              # Test (5-10 min)
python ml_drug_screening.py          # Full run (30-60 min)

# After training
python screen_my_compounds.py        # Screen your molecules

# Check results
ls ml_screening_results/             # Your results here
ls figures/                          # Your charts here
open figures/roc_curves.png          # View performance
```

**That's it!** You're now ready to use AI for drug discovery! 🧪🔬

---

*Last updated: 2025*
