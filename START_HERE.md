# 🚀 START HERE - How to Use on Your Computer

## ⚡ Quick Start (3 Steps)

### Step 1: Get the Code on Your Computer

**Option A - Download (Easiest):**
1. Go to: https://github.com/Abbas24-AI/ChatbotAI
2. Click the green **"Code"** button
3. Click **"Download ZIP"**
4. Extract the ZIP file to a folder (like `Documents/ChatbotAI`)

**Option B - Git Clone:**
```bash
git clone https://github.com/Abbas24-AI/ChatbotAI.git
cd ChatbotAI
```

### Step 2: Open Terminal/Command Prompt in That Folder

**Windows:**
- Open the folder in File Explorer
- Click in the address bar, type `cmd`, press Enter

**Mac:**
- Open the folder in Finder
- Right-click folder → Services → New Terminal at Folder

**Linux:**
- Right-click in folder → Open Terminal Here

### Step 3: Run the Test

Type this command and press Enter:
```bash
python test_pipeline.py
```

**If that doesn't work, try:**
```bash
python3 test_pipeline.py
```

**What you'll see:**
```
================================================================================
QUICK TEST OF ML DRUG SCREENING PIPELINE
================================================================================

This test will take approximately 5-10 minutes.
Initializing pipeline...
Retrieving bioactivity data for Acetylcholinesterase...
(This may take 2-3 minutes to download from ChEMBL)

✓ Retrieved 1234 compounds
✓ Calculated 1234 descriptors
✓ Trained 2 models
✓ Ensemble model created
✓ Screened 3 compounds, found 1 hits
✓ Model saved

✓✓✓ TEST COMPLETED SUCCESSFULLY! ✓✓✓
```

**If successful:** You're ready to use the pipeline! 🎉

**If failed:** See troubleshooting below ⬇️

---

## 📋 What to Do Next

After the test works, choose what you want to do:

### A. Train a Model for Your Target Protein

```bash
python ml_drug_screening.py
```

Then type when asked:
```
Target name: EGFR
```
(or any protein name you want)

**Time:** 30-60 minutes
**Output:** Trained AI model + results in `ml_screening_results/`

### B. Screen Your Own Molecules

1. First, make sure you have a trained model (run option A above)
2. Edit `screen_my_compounds.py`:
   - Open with any text editor
   - Find line ~22: `your_smiles = [`
   - Add your SMILES strings
3. Run:
```bash
python screen_my_compounds.py
```

**Output:** `my_screening_results.csv` with predictions

### C. See More Examples

```bash
python example_usage.py
```

Choose from 7 different examples to learn advanced features.

---

## 🔧 Troubleshooting

### Error: "python: command not found"

**Fix:** You need to install Python first

**Windows:** Download from https://python.org/downloads
- ✅ Check "Add Python to PATH" during installation

**Mac:** Usually pre-installed, try `python3` instead

**Linux:** Install with:
```bash
sudo apt install python3 python3-pip  # Ubuntu/Debian
```

### Error: "No module named 'rdkit'"

**Fix:** Install RDKit (chemistry library)

**Method 1 - Automatic (usually works):**
Just run the script, it auto-installs dependencies

**Method 2 - Manual:**
```bash
pip install rdkit-pypi
```

**Method 3 - Conda (if above fails):**
```bash
conda install -c conda-forge rdkit
```

### Error: "ChEMBL connection timeout"

**Cause:** Internet connection or ChEMBL server busy

**Fix:**
1. Check your internet connection
2. Wait 1-2 minutes and try again
3. If on corporate network, try at home (firewall might block)

### The script is very slow / stuck

**This is normal!** The pipeline:
- Downloads data from internet (2-5 min)
- Calculates molecular features (5-10 min)
- Trains AI models (10-30 min)
- Optimizes models (10-20 min)

**Total time:** 30-60 minutes for full pipeline

**Want it faster?**
- Use `test_pipeline.py` (only 5-10 min)
- Or reduce settings in the code

### Error: "Permission denied"

**Windows:** Run command prompt as Administrator

**Mac/Linux:**
```bash
chmod +x ml_drug_screening.py
./ml_drug_screening.py
```

---

## 📁 Where Are My Results?

After running, look in these folders:

```
ChatbotAI/
├── ml_screening_results/     ← All your results here!
│   ├── bioactivity_data.csv       (data from ChEMBL)
│   ├── training_summary.txt       (model performance)
│   └── pipeline.pkl               (saved AI model)
│
├── trained_models/            ← AI models here!
│   └── ensemble_model.pkl         (best model - use this!)
│
└── figures/                   ← Charts here!
    ├── roc_curves.png             (model performance)
    ├── confusion_matrices.png     (accuracy visualization)
    └── shap_summary_*.png         (feature importance)
```

**How to view:**
- `.csv` files → Open with Excel or Google Sheets
- `.png` files → Open with any image viewer
- `.txt` files → Open with Notepad or TextEdit
- `.pkl` files → These are for the program (don't open)

---

## 💡 Tips for Success

### ✅ DO:
1. **Start with the test:** Run `test_pipeline.py` first
2. **Be patient:** First run takes 30-60 minutes
3. **Use popular targets:** Try "EGFR" or "Acetylcholinesterase" first
4. **Keep results:** Don't delete `ml_screening_results/` folder
5. **Read USER_GUIDE.md:** For detailed instructions

### ❌ DON'T:
1. **Don't close terminal** while running (looks stuck but it's working!)
2. **Don't run on slow internet** (downloads data from internet)
3. **Don't use < 4GB RAM computer** (will run out of memory)
4. **Don't expect instant results** (AI training takes time)
5. **Don't skip the test** (verify it works first!)

---

## 📚 Documentation

| File | What It Is | Read If... |
|------|-----------|-----------|
| **START_HERE.md** (this file) | Quick start | You're just starting |
| **USER_GUIDE.md** | Complete beginner guide | You want step-by-step help |
| **QUICKSTART.md** | Quick reference | You know Python already |
| **ML_DRUG_SCREENING_README.md** | Full documentation | You want all details |
| **example_usage.py** | Code examples | You want to see code |

---

## 🎯 Example: Complete First Run

Here's exactly what to type for your first successful run:

```bash
# 1. Test it works (5-10 minutes)
python test_pipeline.py

# 2. If test passes, run full pipeline (30-60 minutes)
python ml_drug_screening.py

# When asked "Enter the target protein name:", type:
Acetylcholinesterase

# When asked "Select target (1-X):", type:
1

# When asked "Do you want to perform virtual screening?", type:
no

# 3. Wait for it to finish (grab a coffee ☕)

# 4. Check your results:
ls ml_screening_results/
open figures/roc_curves.png

# 5. Screen your own molecules:
python screen_my_compounds.py
```

---

## ❓ Still Need Help?

1. **Read USER_GUIDE.md** - Complete guide with troubleshooting
2. **Check examples** - Run `python example_usage.py`
3. **Review documentation** - See ML_DRUG_SCREENING_README.md
4. **Ask for help** - Open issue on GitHub

---

## 🎓 What This Pipeline Does (Simple Explanation)

```
Your Target Protein (e.g., "EGFR")
         ⬇️
   Download Data (from ChEMBL database)
         ⬇️
   Analyze Molecules (calculate features)
         ⬇️
   Train AI Models (teach computer to recognize active molecules)
         ⬇️
   Test & Optimize (make models better)
         ⬇️
   Save Best Model (your AI drug screener!)
         ⬇️
   Screen New Molecules (find potential drugs!)
```

---

## 🏁 Ready?

**First time:** Run the test
```bash
python test_pipeline.py
```

**Then:** Run the full pipeline
```bash
python ml_drug_screening.py
```

**Good luck! 🧪🔬**

---

*Need detailed help? Read USER_GUIDE.md*
*Have Python experience? Read QUICKSTART.md*
*Want all details? Read ML_DRUG_SCREENING_README.md*
