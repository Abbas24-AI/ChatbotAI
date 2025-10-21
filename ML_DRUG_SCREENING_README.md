# ML-Based Virtual Drug Screening Pipeline

A production-ready Python pipeline for machine learning-based virtual drug screening using ChEMBL bioactivity data, comprehensive molecular descriptors, ensemble modeling, and interpretability analysis.

## Features

### Core Capabilities
- **ChEMBL Integration**: Automated retrieval of bioactivity data with interactive target selection
- **Comprehensive Preprocessing**: Missing value handling, IC50 standardization, and activity classification
- **Rich Molecular Descriptors**: Morgan fingerprints, MACCS keys, RDKit descriptors, and Lipinski properties
- **Multiple ML Models**: Random Forest, XGBoost, LightGBM, Gradient Boosting, SVM, and Neural Networks
- **10-Fold Cross-Validation**: Robust model evaluation with stratified sampling
- **Hyperparameter Optimization**: RandomizedSearchCV and GridSearchCV support
- **Ensemble Modeling**: Stacking classifier from top 3 performers
- **Model Interpretability**: SHAP analysis and feature importance plots
- **Virtual Screening**: Automated screening of compound libraries
- **Publication-Quality Figures**: High-resolution plots suitable for journals (300 DPI)
- **Complete Reproducibility**: Fixed random seeds throughout pipeline

### Advanced Features
- **Class Imbalance Handling**: SMOTE and undersampling strategies
- **Memory-Efficient Processing**: Batch processing for large datasets
- **Comprehensive Metrics**: Accuracy, precision, recall, F1, ROC-AUC, PR-AUC, MCC
- **Automatic Dependency Installation**: Self-installing required packages
- **Error Handling**: Robust error handling with detailed logging
- **Model Persistence**: Save and load trained models and pipelines
- **Configurable Parameters**: JSON-based configuration management

## Installation

### Prerequisites
- Python 3.7+
- pip package manager

### Quick Start

1. **Clone the repository:**
```bash
git clone <repository-url>
cd ChatbotAI
```

2. **Run the script (dependencies auto-install):**
```bash
python ml_drug_screening.py
```

The script will automatically install all required dependencies on first run.

### Manual Installation

If you prefer to install dependencies manually:

```bash
pip install chembl_webresource_client rdkit-pypi numpy pandas scikit-learn \
    xgboost lightgbm matplotlib seaborn shap imbalanced-learn scipy joblib requests
```

## Usage

### Basic Usage

Run the main pipeline:

```bash
python ml_drug_screening.py
```

Follow the interactive prompts to:
1. Enter target protein name (e.g., "EGFR", "Acetylcholinesterase")
2. Select target from search results
3. Optionally perform virtual screening on new compounds

### Programmatic Usage

```python
from ml_drug_screening import MLDrugScreeningPipeline, PipelineConfig

# Create custom configuration
config = PipelineConfig(
    random_seed=42,
    activity_threshold_nm=1000.0,
    n_folds=10,
    handle_imbalance=True,
    imbalance_strategy='smote',
    hyperparam_search='random',
    hyperparam_iterations=30,
    ensemble_top_n=3,
    figure_dpi=300
)

# Initialize pipeline
pipeline = MLDrugScreeningPipeline(config)

# Step 1: Retrieve target data
bioactivity_df = pipeline.retrieve_target_data("EGFR", auto_select=True)

# Step 2: Calculate molecular descriptors
descriptors_df = pipeline.calculate_descriptors()

# Step 3: Train models
training_results = pipeline.train_models()

# Step 4: Optimize top models
optimized_models = pipeline.optimize_top_models(top_n=3)

# Step 5: Build ensemble
ensemble = pipeline.build_ensemble(optimized_models)

# Step 6: Generate interpretability plots
pipeline.generate_interpretability_plots(optimized_models)

# Step 7: Virtual screening (optional)
screening_smiles = ['CC(C)Cc1ccc(cc1)C(C)C(O)=O', ...]
results, hits = pipeline.perform_virtual_screening(screening_smiles)

# Save pipeline
pipeline.save_pipeline('my_pipeline.pkl')
```

### Loading Saved Pipeline

```python
from ml_drug_screening import MLDrugScreeningPipeline

# Load previously saved pipeline
pipeline = MLDrugScreeningPipeline.load_pipeline('ml_screening_results/pipeline.pkl')

# Perform virtual screening with loaded model
results, hits = pipeline.perform_virtual_screening(new_smiles)
```

## Configuration

### Configuration Parameters

Create a custom configuration:

```python
from ml_drug_screening import PipelineConfig

config = PipelineConfig(
    # Reproducibility
    random_seed=42,

    # ChEMBL parameters
    activity_threshold_nm=1000.0,  # IC50 threshold (nM)
    min_compounds=100,              # Minimum compounds required

    # Cross-validation
    n_folds=10,                     # Number of CV folds

    # Class imbalance
    handle_imbalance=True,
    imbalance_strategy='smote',     # 'smote', 'undersample', or 'none'
    max_imbalance_ratio=3.0,

    # Molecular descriptors
    use_morgan_fingerprints=True,
    morgan_radius=2,
    morgan_bits=2048,
    use_maccs_keys=True,
    use_rdkit_descriptors=True,
    use_lipinski_descriptors=True,

    # Models to train
    models_to_train=['rf', 'xgb', 'lgb', 'svm', 'mlp', 'gb'],

    # Hyperparameter optimization
    hyperparam_search='random',     # 'grid', 'random', or 'none'
    hyperparam_iterations=50,
    hyperparam_cv=5,

    # Ensemble
    ensemble_top_n=3,

    # Virtual screening
    screening_threshold=0.7,        # Probability threshold

    # Visualization
    figure_dpi=300,
    figure_format='png',

    # Output
    output_dir='ml_screening_results',
    models_dir='trained_models',
    figures_dir='figures'
)
```

### Save/Load Configuration

```python
# Save configuration
config.save('my_config.json')

# Load configuration
config = PipelineConfig.load('my_config.json')
```

## Pipeline Workflow

### Step-by-Step Process

1. **Target Data Retrieval**
   - Search ChEMBL for target proteins
   - Interactive target selection
   - Retrieve IC50 bioactivity data
   - Preprocess and standardize data
   - Convert IC50 to binary labels (active/inactive)

2. **Molecular Descriptor Calculation**
   - Morgan (circular) fingerprints
   - MACCS keys fingerprints
   - RDKit 2D descriptors (200+ descriptors)
   - Lipinski's Rule of 5 properties
   - Error handling for invalid structures

3. **Model Training**
   - 10-fold stratified cross-validation
   - Multiple ML algorithms in parallel
   - Class imbalance handling (SMOTE)
   - Feature scaling (RobustScaler)
   - Comprehensive metrics calculation

4. **Hyperparameter Optimization**
   - Rank models by ROC-AUC
   - Select top N performers
   - RandomizedSearchCV or GridSearchCV
   - Nested cross-validation

5. **Ensemble Building**
   - Stacking classifier from top 3 models
   - Logistic regression meta-classifier
   - 5-fold CV for ensemble training

6. **Interpretability Analysis**
   - SHAP summary plots
   - Feature importance rankings
   - Publication-quality visualizations

7. **Virtual Screening**
   - Process new compound SMILES
   - Calculate descriptors
   - Apply ensemble model
   - Filter hits by probability threshold

## Output Files

### Directory Structure

```
ml_screening_results/
├── config.json                    # Pipeline configuration
├── bioactivity_data.csv          # Raw bioactivity data
├── molecular_descriptors.csv     # Calculated descriptors
├── training_summary.txt          # Model training summary
├── screening_results.csv         # Virtual screening results
├── screening_hits.csv            # Identified hits
└── pipeline.pkl                  # Complete pipeline state

trained_models/
├── RandomForest_optimized.pkl
├── XGBoost_optimized.pkl
├── LightGBM_optimized.pkl
├── ensemble_model.pkl            # Final ensemble model
├── feature_scaler.pkl            # Fitted scaler
└── feature_names.pkl             # Feature names

figures/
├── activity_distribution.png     # IC50 and class distribution
├── model_comparison.png          # Metrics comparison
├── roc_curves.png                # ROC curves for all models
├── precision_recall_curves.png   # PR curves for all models
├── confusion_matrices.png        # Confusion matrices
├── feature_importance_*.png      # Feature importance plots
└── shap_summary_*.png           # SHAP analysis plots
```

## Evaluation Metrics

The pipeline calculates comprehensive metrics:

- **Accuracy**: Overall classification accuracy
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1 Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under ROC curve
- **PR-AUC**: Area under precision-recall curve
- **MCC**: Matthews correlation coefficient
- **Confusion Matrix**: True/false positives/negatives

All metrics include mean ± standard deviation from cross-validation.

## Molecular Descriptors

### Fingerprints
- **Morgan Fingerprints**: Circular fingerprints (2048 bits, radius 2)
- **MACCS Keys**: 166-bit structural keys

### Physicochemical Properties (Lipinski)
- Molecular Weight
- LogP (lipophilicity)
- H-bond donors
- H-bond acceptors
- Topological polar surface area (TPSA)
- Rotatable bonds
- Aromatic rings
- Aliphatic rings

### RDKit Descriptors (200+)
- Topological descriptors
- Constitutional descriptors
- Electrotopological state
- Molecular orbital energies
- Surface area descriptors
- Connectivity indices
- And many more...

## Advanced Features

### Class Imbalance Handling

```python
config = PipelineConfig(
    handle_imbalance=True,
    imbalance_strategy='smote',  # or 'undersample'
    max_imbalance_ratio=3.0
)
```

### Custom Model Selection

```python
config = PipelineConfig(
    models_to_train=['rf', 'xgb', 'lgb']  # Train only specific models
)
```

### Hyperparameter Tuning Control

```python
config = PipelineConfig(
    hyperparam_search='random',
    hyperparam_iterations=100,  # More iterations = better optimization
    hyperparam_cv=5             # Cross-validation folds for tuning
)
```

### Virtual Screening Threshold

```python
config = PipelineConfig(
    screening_threshold=0.8  # Higher threshold = more stringent hits
)
```

## Troubleshooting

### Common Issues

**1. ChEMBL API timeout**
```
Solution: The script includes automatic retry logic. If persistent, reduce batch size or add delays.
```

**2. RDKit installation fails**
```
Solution: Use conda instead: conda install -c conda-forge rdkit
```

**3. Memory error with large datasets**
```
Solution: Reduce batch_size in config or process in chunks
```

**4. SHAP computation too slow**
```
Solution: SHAP automatically samples 1000 instances. Reduce sample size if needed.
```

**5. Model training very slow**
```
Solution:
- Reduce hyperparam_iterations
- Use fewer models
- Reduce n_folds
- Disable hyperparameter optimization
```

### Warnings Explained

- **Class imbalance warning**: Adjust activity_threshold_nm or enable SMOTE
- **Few compounds warning**: Try different target or lower min_compounds
- **Missing descriptors**: Some molecules fail - they're automatically skipped

## Performance Optimization

### Speed Improvements

1. **Reduce cross-validation folds**: `n_folds=5` instead of 10
2. **Limit hyperparameter search**: `hyperparam_iterations=20`
3. **Fewer models**: `models_to_train=['rf', 'xgb']`
4. **Disable SHAP**: Comment out `generate_interpretability_plots()`

### Memory Optimization

1. **Batch processing**: Set `batch_size=5000`
2. **Fewer descriptors**: Disable `use_rdkit_descriptors=False`
3. **Smaller fingerprints**: `morgan_bits=1024`

## Scientific Validation

### Best Practices Implemented

- **Stratified K-Fold CV**: Maintains class distribution in folds
- **Nested CV**: Separate CV for hyperparameter tuning
- **Fixed Random Seeds**: Complete reproducibility
- **Class Imbalance**: SMOTE for balanced training
- **Feature Scaling**: RobustScaler resistant to outliers
- **Multiple Metrics**: Avoid overfitting to single metric
- **Ensemble Methods**: Reduce overfitting, improve generalization

### Publication Guidelines

The pipeline generates publication-ready outputs:
- 300 DPI figures (configurable)
- Comprehensive metrics tables
- SHAP interpretability plots
- Confusion matrices
- ROC and PR curves

## Citation

If you use this pipeline in your research, please cite:

```
@software{ml_drug_screening,
  title={ML-Based Virtual Drug Screening Pipeline},
  author={Your Name},
  year={2025},
  url={https://github.com/your-repo}
}
```

## License

This project is licensed under the MIT License - see LICENSE file for details.

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new features
4. Submit a pull request

## Support

For issues, questions, or suggestions:
- Open an issue on GitHub
- Email: your.email@example.com

## Acknowledgments

- **ChEMBL**: For providing open bioactivity data
- **RDKit**: For cheminformatics toolkit
- **scikit-learn**: For machine learning algorithms
- **SHAP**: For model interpretability

## Changelog

### Version 1.0.0 (2025)
- Initial release
- Full pipeline implementation
- Comprehensive documentation

## Future Enhancements

Planned features:
- [ ] Deep learning models (Graph Neural Networks)
- [ ] Active learning strategies
- [ ] Multi-target screening
- [ ] 3D molecular descriptors
- [ ] Scaffold analysis and clustering
- [ ] Automated report generation
- [ ] Web interface
- [ ] Docker containerization

## Examples

### Example 1: Screen EGFR Inhibitors

```python
from ml_drug_screening import MLDrugScreeningPipeline, PipelineConfig

config = PipelineConfig(random_seed=42, activity_threshold_nm=100)
pipeline = MLDrugScreeningPipeline(config)

# Retrieve EGFR data
bioactivity_df = pipeline.retrieve_target_data("EGFR", auto_select=True)
descriptors_df = pipeline.calculate_descriptors()
results = pipeline.train_models()
optimized = pipeline.optimize_top_models()
ensemble = pipeline.build_ensemble(optimized)

# Screen new compounds
new_smiles = ["CCN(CC)CCNC(=O)c1c(C)[nH]c(/C=C2\C(=O)Nc3ccc(Br)cc32)c1C"]
results, hits = pipeline.perform_virtual_screening(new_smiles)
```

### Example 2: Load and Use Saved Model

```python
from ml_drug_screening import MLDrugScreeningPipeline

# Load saved pipeline
pipeline = MLDrugScreeningPipeline.load_pipeline('ml_screening_results/pipeline.pkl')

# Screen compounds from file
import pandas as pd
df = pd.read_csv('compounds_to_screen.csv')
results, hits = pipeline.perform_virtual_screening(df['smiles'].tolist())

print(f"Identified {len(hits)} potential hits!")
```

## Contact

For more information, visit: https://github.com/your-repo/ChatbotAI
