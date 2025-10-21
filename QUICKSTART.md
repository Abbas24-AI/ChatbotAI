# Quick Start Guide - ML Drug Screening Pipeline

Get started with the ML-based virtual drug screening pipeline in 5 minutes!

## Installation

### Option 1: Automatic (Recommended)

Just run the script - it will install dependencies automatically:

```bash
python ml_drug_screening.py
```

### Option 2: Manual

Install dependencies first:

```bash
pip install -r requirements.txt
```

## Basic Usage

### 1. Run the Complete Pipeline

```bash
python ml_drug_screening.py
```

Then follow the interactive prompts:

1. **Enter target name**: e.g., "EGFR", "Acetylcholinesterase", "COX-2"
2. **Select target**: Choose from the search results
3. **Optional screening**: Enter 'yes' to screen new compounds

### 2. Run Example Scripts

Try the example usage scripts:

```bash
python example_usage.py
```

Choose from 7 different examples demonstrating various use cases.

## Output

After running, you'll find:

```
ml_screening_results/          # All results
├── bioactivity_data.csv       # Retrieved ChEMBL data
├── molecular_descriptors.csv  # Calculated descriptors
├── training_summary.txt       # Model performance
└── pipeline.pkl               # Saved pipeline

trained_models/                # Trained models
├── ensemble_model.pkl         # Final ensemble
├── RandomForest_optimized.pkl
├── XGBoost_optimized.pkl
└── ...

figures/                       # Publication-quality plots
├── roc_curves.png
├── precision_recall_curves.png
├── confusion_matrices.png
├── shap_summary_*.png
└── ...
```

## What the Pipeline Does

1. **Retrieves data** from ChEMBL for your target protein
2. **Processes** IC50 values and converts to active/inactive labels
3. **Calculates** 2000+ molecular descriptors and fingerprints
4. **Trains** 6 different ML models with 10-fold cross-validation
5. **Optimizes** hyperparameters for top 3 models
6. **Builds** stacking ensemble from best models
7. **Generates** SHAP plots for interpretability
8. **Screens** new compounds (optional)
9. **Saves** all models and results

## Quick Examples

### Example 1: Screen EGFR Inhibitors

```python
from ml_drug_screening import MLDrugScreeningPipeline, PipelineConfig

# Initialize
pipeline = MLDrugScreeningPipeline()

# Run pipeline
pipeline.retrieve_target_data("EGFR", auto_select=True)
pipeline.calculate_descriptors()
pipeline.train_models()
optimized = pipeline.optimize_top_models()
ensemble = pipeline.build_ensemble(optimized)

# Screen compounds
results, hits = pipeline.perform_virtual_screening([
    'CCN(CC)CCNC(=O)c1c(C)[nH]c(/C=C2\C(=O)Nc3ccc(Br)cc32)c1C'
])

print(f"Found {len(hits)} hits!")
```

### Example 2: Load Saved Model

```python
from ml_drug_screening import MLDrugScreeningPipeline

# Load
pipeline = MLDrugScreeningPipeline.load_pipeline('ml_screening_results/pipeline.pkl')

# Screen
new_smiles = ['CC(C)Cc1ccc(cc1)C(C)C(O)=O']
results, hits = pipeline.perform_virtual_screening(new_smiles)
```

### Example 3: Custom Configuration

```python
from ml_drug_screening import PipelineConfig, MLDrugScreeningPipeline

# Configure
config = PipelineConfig(
    activity_threshold_nm=100,      # More stringent
    n_folds=5,                       # Faster
    models_to_train=['rf', 'xgb'],   # Only 2 models
    screening_threshold=0.8          # Higher confidence
)

# Run
pipeline = MLDrugScreeningPipeline(config)
# ... continue with pipeline steps
```

## Common Targets to Try

Here are some popular drug targets with good ChEMBL data:

- **EGFR** - Epidermal Growth Factor Receptor (cancer)
- **Acetylcholinesterase** - Alzheimer's disease
- **COX-2** - Cyclooxygenase-2 (inflammation)
- **Dopamine D2 receptor** - Neurological disorders
- **BACE1** - Beta-secretase 1 (Alzheimer's)
- **HIV-1 protease** - Antiviral target
- **PARP1** - Poly ADP-ribose polymerase (cancer)
- **JAK2** - Janus kinase 2 (immunology)

## Performance Tips

### For Speed

```python
config = PipelineConfig(
    n_folds=5,                      # Instead of 10
    models_to_train=['rf', 'xgb'],  # Only 2 models
    hyperparam_search='none',       # Skip tuning
    use_rdkit_descriptors=False     # Skip slow descriptors
)
```

### For Accuracy

```python
config = PipelineConfig(
    n_folds=10,                     # More folds
    hyperparam_iterations=100,      # More tuning
    ensemble_top_n=5,               # More models in ensemble
    handle_imbalance=True           # Use SMOTE
)
```

## Troubleshooting

### Issue: "ChEMBL connection timeout"
**Solution**: Wait a moment and retry. ChEMBL API has rate limits.

### Issue: "RDKit not found"
**Solution**: Install via conda: `conda install -c conda-forge rdkit`

### Issue: "Out of memory"
**Solution**: Reduce `morgan_bits=1024` or disable `use_rdkit_descriptors=False`

### Issue: "Training too slow"
**Solution**: Reduce `n_folds=5` and `hyperparam_iterations=20`

## Next Steps

1. **Read the full documentation**: See `ML_DRUG_SCREENING_README.md`
2. **Try example scripts**: Run `python example_usage.py`
3. **Customize configuration**: Adjust parameters for your needs
4. **Screen your compounds**: Use trained models for virtual screening

## Key Features

✅ Automatic dependency installation
✅ Interactive target selection
✅ 2000+ molecular descriptors
✅ 6 ML algorithms
✅ 10-fold cross-validation
✅ Hyperparameter optimization
✅ Ensemble modeling
✅ SHAP interpretability
✅ Publication-quality figures (300 DPI)
✅ Complete reproducibility
✅ Model persistence

## Typical Runtime

- **Small dataset** (100-500 compounds): ~5-15 minutes
- **Medium dataset** (500-2000 compounds): ~15-45 minutes
- **Large dataset** (2000-10000 compounds): ~1-3 hours

*Times vary based on hardware and configuration*

## Minimum Requirements

- Python 3.7+
- 4 GB RAM (8 GB recommended)
- 2 GB disk space
- Internet connection (for ChEMBL)

## Support

For issues or questions:
- Check the full README: `ML_DRUG_SCREENING_README.md`
- Review examples: `example_usage.py`
- Open an issue on GitHub

## Citation

If you use this pipeline in research, please cite appropriately.

---

**Ready to start?** Just run:

```bash
python ml_drug_screening.py
```

Happy screening! 🧪🔬
