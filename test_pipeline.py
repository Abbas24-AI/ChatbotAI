#!/usr/bin/env python3
"""
Quick Test Script for ML Drug Screening Pipeline

This script runs a minimal test to verify the pipeline works on your computer.
It uses fewer models and smaller settings for faster execution (~5-10 minutes).
"""

from ml_drug_screening import MLDrugScreeningPipeline, PipelineConfig

print("\n" + "="*80)
print("QUICK TEST OF ML DRUG SCREENING PIPELINE")
print("="*80)
print("\nThis test will take approximately 5-10 minutes.")
print("It will use a small subset of features for faster testing.\n")

# Create a minimal configuration for testing
config = PipelineConfig(
    random_seed=42,
    activity_threshold_nm=1000.0,

    # Reduce for speed
    n_folds=3,  # Instead of 10
    models_to_train=['rf', 'xgb'],  # Only 2 models
    hyperparam_search='none',  # Skip hyperparameter tuning

    # Use minimal descriptors
    use_morgan_fingerprints=True,
    morgan_bits=1024,  # Smaller fingerprint
    use_maccs_keys=True,
    use_rdkit_descriptors=False,  # Skip for speed
    use_lipinski_descriptors=True,

    # Output
    output_dir='test_results',
    models_dir='test_models',
    figures_dir='test_figures'
)

try:
    # Initialize pipeline
    print("Initializing pipeline...")
    pipeline = MLDrugScreeningPipeline(config)

    # Use Acetylcholinesterase as test target (usually has good data)
    print("\nRetrieving bioactivity data for Acetylcholinesterase...")
    print("(This may take 2-3 minutes to download from ChEMBL)")
    bioactivity_df = pipeline.retrieve_target_data(
        "Acetylcholinesterase",
        auto_select=True  # Automatically select first match
    )

    print(f"\n✓ Retrieved {len(bioactivity_df)} compounds")

    # Calculate descriptors
    print("\nCalculating molecular descriptors...")
    descriptors_df = pipeline.calculate_descriptors()
    print(f"✓ Calculated {len(descriptors_df.columns)} descriptors")

    # Train models
    print("\nTraining ML models (this takes a few minutes)...")
    training_results = pipeline.train_models()
    print(f"✓ Trained {len(training_results)} models")

    # Build simple ensemble (skip optimization for speed)
    print("\nBuilding ensemble model...")
    # Get top 2 models
    model_scores = {
        name: result['metrics']['roc_auc_mean']
        for name, result in training_results.items()
    }
    sorted_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)
    top_models = [(name, training_results[name]['model']) for name, _ in sorted_models[:2]]

    # Prepare data
    feature_cols = [col for col in descriptors_df.columns
                   if col not in ['smiles', 'canonical_smiles', 'activity_class',
                                 'activity_label', 'pIC50']]
    X = descriptors_df[feature_cols]
    y = descriptors_df['activity_class']
    X_processed, y_processed = pipeline.model_trainer.preprocess_features(X, y, fit_scaler=True)

    ensemble = pipeline.model_trainer.build_ensemble_model(top_models, X_processed, y_processed)
    print("✓ Ensemble model created")

    # Test virtual screening with demo compounds
    print("\nTesting virtual screening on demo compounds...")
    demo_smiles = [
        'CC(C)Cc1ccc(cc1)C(C)C(O)=O',  # Ibuprofen
        'CC(=O)Oc1ccccc1C(O)=O',        # Aspirin
        'CN1C=NC2=C1C(=O)N(C(=O)N2C)C', # Caffeine
    ]
    results, hits = pipeline.virtual_screener.screen_compounds(
        ensemble,
        pipeline.model_trainer.scaler,
        pipeline.model_trainer.feature_names,
        demo_smiles,
        ['Ibuprofen', 'Aspirin', 'Caffeine']
    )
    print(f"✓ Screened {len(results)} compounds, found {len(hits)} hits")

    # Save the model
    import joblib
    import os
    os.makedirs('test_models', exist_ok=True)
    joblib.dump(ensemble, 'test_models/test_ensemble.pkl')
    print("✓ Model saved")

    print("\n" + "="*80)
    print("✓✓✓ TEST COMPLETED SUCCESSFULLY! ✓✓✓")
    print("="*80)
    print("\nYour installation is working correctly!")
    print("\nNext steps:")
    print("1. Check the 'test_results/' folder for outputs")
    print("2. Check the 'test_figures/' folder for plots")
    print("3. Run the full pipeline: python ml_drug_screening.py")
    print("4. Or try examples: python example_usage.py")
    print("\nResults summary:")
    print(f"  - Downloaded {len(bioactivity_df)} bioactivity records")
    print(f"  - Trained {len(training_results)} ML models")
    print(f"  - Best model ROC-AUC: {max(model_scores.values()):.3f}")
    print(f"  - Virtual screening identified {len(hits)} potential hits")

except Exception as e:
    print("\n" + "="*80)
    print("❌ TEST FAILED")
    print("="*80)
    print(f"\nError: {e}")
    print("\nTroubleshooting:")
    print("1. Make sure you have Python 3.7+ installed")
    print("2. Check your internet connection")
    print("3. Try running: pip install --upgrade pip")
    print("4. Try installing dependencies manually: pip install -r requirements.txt")
    print("5. If using Windows, you may need to install Visual C++ Build Tools")
    print("\nFor RDKit installation issues:")
    print("   Try: conda install -c conda-forge rdkit")
    print("   (if you have conda/anaconda installed)")

    import traceback
    print("\nFull error details:")
    traceback.print_exc()

if __name__ == "__main__":
    pass
