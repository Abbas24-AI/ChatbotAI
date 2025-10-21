#!/usr/bin/env python3
"""
Example Usage Scripts for ML Drug Screening Pipeline

This file demonstrates various ways to use the ML drug screening pipeline
for different scenarios and use cases.
"""

import pandas as pd
from ml_drug_screening import MLDrugScreeningPipeline, PipelineConfig


def example_1_basic_pipeline():
    """
    Example 1: Basic Pipeline Execution

    This example shows the complete pipeline workflow from target selection
    to virtual screening.
    """
    print("\n" + "="*80)
    print("EXAMPLE 1: Basic Pipeline Execution")
    print("="*80 + "\n")

    # Create configuration with default settings
    config = PipelineConfig(
        random_seed=42,
        activity_threshold_nm=1000.0,
        n_folds=10
    )

    # Initialize pipeline
    pipeline = MLDrugScreeningPipeline(config)

    # Step 1: Retrieve target data (auto-select first target)
    print("Retrieving bioactivity data for Acetylcholinesterase...")
    bioactivity_df = pipeline.retrieve_target_data(
        "Acetylcholinesterase",
        auto_select=True
    )

    # Step 2: Calculate molecular descriptors
    print("\nCalculating molecular descriptors...")
    descriptors_df = pipeline.calculate_descriptors()

    # Step 3: Train models
    print("\nTraining machine learning models...")
    training_results = pipeline.train_models()

    # Step 4: Optimize top models
    print("\nOptimizing hyperparameters for top 3 models...")
    optimized_models = pipeline.optimize_top_models(top_n=3)

    # Step 5: Build ensemble
    print("\nBuilding ensemble model...")
    ensemble = pipeline.build_ensemble(optimized_models)

    # Step 6: Generate interpretability plots
    print("\nGenerating interpretability plots...")
    pipeline.generate_interpretability_plots(optimized_models)

    # Step 7: Virtual screening on demo compounds
    print("\nPerforming virtual screening on demo compounds...")
    demo_smiles = [
        'CC(C)Cc1ccc(cc1)C(C)C(O)=O',  # Ibuprofen
        'CC(=O)Oc1ccccc1C(O)=O',        # Aspirin
        'CN1C=NC2=C1C(=O)N(C(=O)N2C)C', # Caffeine
    ]
    results, hits = pipeline.perform_virtual_screening(demo_smiles)

    # Save pipeline
    pipeline.save_pipeline('example_1_pipeline.pkl')

    print("\n✓ Pipeline execution complete!")
    print(f"  Total compounds: {len(bioactivity_df)}")
    print(f"  Models trained: {len(training_results)}")
    print(f"  Virtual screening hits: {len(hits)}")


def example_2_custom_configuration():
    """
    Example 2: Custom Configuration

    This example shows how to customize pipeline parameters for specific needs.
    """
    print("\n" + "="*80)
    print("EXAMPLE 2: Custom Configuration")
    print("="*80 + "\n")

    # Create custom configuration for high-performance screening
    config = PipelineConfig(
        # Reproducibility
        random_seed=123,

        # More stringent activity threshold
        activity_threshold_nm=100.0,  # 100 nM instead of 1000 nM

        # Reduce CV folds for faster training
        n_folds=5,

        # Enable class imbalance handling
        handle_imbalance=True,
        imbalance_strategy='smote',

        # Use only best-performing models
        models_to_train=['rf', 'xgb', 'lgb'],

        # Aggressive hyperparameter tuning
        hyperparam_search='random',
        hyperparam_iterations=100,

        # Ensemble from top 3
        ensemble_top_n=3,

        # Higher screening threshold for hits
        screening_threshold=0.8,

        # High-quality figures
        figure_dpi=300,

        # Custom output directory
        output_dir='custom_screening_results'
    )

    # Save configuration for reproducibility
    config.save('custom_config.json')

    # Initialize and run pipeline
    pipeline = MLDrugScreeningPipeline(config)

    print("✓ Custom configuration created and saved!")
    print(f"  Activity threshold: {config.activity_threshold_nm} nM")
    print(f"  Models to train: {config.models_to_train}")
    print(f"  Screening threshold: {config.screening_threshold}")


def example_3_minimal_descriptors():
    """
    Example 3: Minimal Descriptor Set for Faster Processing

    This example shows how to use only essential descriptors for faster
    processing of large datasets.
    """
    print("\n" + "="*80)
    print("EXAMPLE 3: Minimal Descriptor Set")
    print("="*80 + "\n")

    # Configure for speed with minimal descriptors
    config = PipelineConfig(
        random_seed=42,

        # Use only fingerprints (faster than RDKit descriptors)
        use_morgan_fingerprints=True,
        morgan_bits=1024,  # Smaller fingerprint
        use_maccs_keys=True,
        use_rdkit_descriptors=False,  # Disable slow RDKit descriptors
        use_lipinski_descriptors=True,  # Keep for drug-likeness

        # Fast training
        n_folds=5,
        models_to_train=['rf', 'xgb'],  # Only 2 fast models
        hyperparam_search='none',  # Skip hyperparameter tuning

        output_dir='fast_screening_results'
    )

    pipeline = MLDrugScreeningPipeline(config)

    print("✓ Minimal descriptor configuration created!")
    print("  This configuration prioritizes speed over descriptor richness")


def example_4_load_and_screen():
    """
    Example 4: Load Saved Model and Screen New Compounds

    This example shows how to load a previously trained model and use it
    for virtual screening without retraining.
    """
    print("\n" + "="*80)
    print("EXAMPLE 4: Load Saved Model and Screen")
    print("="*80 + "\n")

    try:
        # Load previously saved pipeline
        pipeline = MLDrugScreeningPipeline.load_pipeline(
            'ml_screening_results/pipeline.pkl'
        )
        print("✓ Pipeline loaded successfully!")

        # Screen new compounds
        new_compounds = [
            'CC(C)Cc1ccc(cc1)C(C)C(O)=O',
            'CC(=O)Oc1ccccc1C(O)=O',
            'CN1C=NC2=C1C(=O)N(C(=O)N2C)C',
            'CC(C)NCC(COc1ccccc1)O',
            'COc1ccc2nc(sc2c1)S(=O)(=O)N',
        ]

        results, hits = pipeline.perform_virtual_screening(new_compounds)

        print(f"\n✓ Screening complete!")
        print(f"  Compounds screened: {len(results)}")
        print(f"  Hits identified: {len(hits)}")

        # Display top hits
        if len(hits) > 0:
            print("\nTop hits:")
            for idx, row in hits.head(3).iterrows():
                print(f"  - {row['smiles']}: {row['predicted_probability']:.3f}")

    except FileNotFoundError:
        print("❌ No saved pipeline found. Run example_1_basic_pipeline() first.")


def example_5_screen_from_file():
    """
    Example 5: Virtual Screening from CSV File

    This example shows how to screen compounds from a CSV file.
    """
    print("\n" + "="*80)
    print("EXAMPLE 5: Screen Compounds from CSV")
    print("="*80 + "\n")

    # Create example CSV file
    example_data = pd.DataFrame({
        'compound_id': ['COMP_001', 'COMP_002', 'COMP_003', 'COMP_004', 'COMP_005'],
        'smiles': [
            'CC(C)Cc1ccc(cc1)C(C)C(O)=O',  # Ibuprofen
            'CC(=O)Oc1ccccc1C(O)=O',        # Aspirin
            'CN1C=NC2=C1C(=O)N(C(=O)N2C)C', # Caffeine
            'CC(C)NCC(COc1ccccc1)O',        # Propranolol
            'COc1ccc2nc(sc2c1)S(=O)(=O)N',  # Sulfamethoxazole
        ],
        'source': ['DrugBank', 'DrugBank', 'DrugBank', 'DrugBank', 'DrugBank']
    })

    # Save to CSV
    example_data.to_csv('compounds_to_screen.csv', index=False)
    print("✓ Created example CSV file: compounds_to_screen.csv")

    try:
        # Load pipeline
        pipeline = MLDrugScreeningPipeline.load_pipeline(
            'ml_screening_results/pipeline.pkl'
        )

        # Read compounds from file
        compounds_df = pd.read_csv('compounds_to_screen.csv')

        # Perform screening
        results, hits = pipeline.perform_virtual_screening(
            compounds_df['smiles'].tolist(),
            compounds_df['compound_id'].tolist()
        )

        # Merge with original data
        final_results = pd.merge(
            results,
            compounds_df[['compound_id', 'source']],
            on='compound_id',
            how='left'
        )

        # Save results
        final_results.to_csv('screening_results_with_metadata.csv', index=False)

        print(f"\n✓ Screening complete!")
        print(f"  Results saved to: screening_results_with_metadata.csv")
        print(f"  Hits identified: {len(hits)}")

    except FileNotFoundError:
        print("❌ No saved pipeline found. Run example_1_basic_pipeline() first.")


def example_6_compare_thresholds():
    """
    Example 6: Compare Different Activity Thresholds

    This example shows how different activity thresholds affect model performance.
    """
    print("\n" + "="*80)
    print("EXAMPLE 6: Compare Activity Thresholds")
    print("="*80 + "\n")

    thresholds = [100, 500, 1000, 5000]  # nM

    for threshold in thresholds:
        print(f"\n{'='*60}")
        print(f"Testing threshold: {threshold} nM")
        print('='*60)

        config = PipelineConfig(
            random_seed=42,
            activity_threshold_nm=threshold,
            n_folds=5,  # Faster for comparison
            models_to_train=['rf', 'xgb'],  # Only 2 models for speed
            hyperparam_search='none',  # Skip tuning for speed
            output_dir=f'threshold_{threshold}_results'
        )

        pipeline = MLDrugScreeningPipeline(config)

        # Quick training (you would add full pipeline here)
        print(f"  Configuration created for {threshold} nM threshold")

    print("\n✓ Threshold comparison configurations created!")
    print("  You can now train models with each threshold and compare performance")


def example_7_ensemble_only():
    """
    Example 7: Build Ensemble from Existing Models

    This example shows how to build an ensemble when you already have
    trained models.
    """
    print("\n" + "="*80)
    print("EXAMPLE 7: Build Ensemble from Existing Models")
    print("="*80 + "\n")

    try:
        # Load pipeline with trained models
        pipeline = MLDrugScreeningPipeline.load_pipeline(
            'ml_screening_results/pipeline.pkl'
        )

        # Rebuild ensemble with different top N
        if pipeline.training_results:
            # Get feature matrix
            feature_cols = [col for col in pipeline.descriptors_data.columns
                           if col not in ['smiles', 'canonical_smiles', 'activity_class',
                                         'activity_label', 'pIC50']]
            X = pipeline.descriptors_data[feature_cols]
            y = pipeline.descriptors_data['activity_class']
            X_processed, y_processed = pipeline.model_trainer.preprocess_features(
                X, y, fit_scaler=True
            )

            # Rank models
            model_scores = {
                name: result['metrics']['roc_auc_mean']
                for name, result in pipeline.training_results.items()
            }
            sorted_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)

            # Build ensemble with top 5 instead of top 3
            top_n = min(5, len(sorted_models))
            top_models = [(name, pipeline.training_results[name]['model'])
                         for name, _ in sorted_models[:top_n]]

            ensemble = pipeline.model_trainer.build_ensemble_model(
                top_models, X_processed, y_processed
            )

            print(f"✓ Built ensemble from top {top_n} models!")

    except FileNotFoundError:
        print("❌ No saved pipeline found. Run example_1_basic_pipeline() first.")


def main():
    """Run all examples or selected examples."""
    print("\n" + "="*80)
    print("ML DRUG SCREENING PIPELINE - EXAMPLE USAGE")
    print("="*80)

    examples = {
        '1': ('Basic Pipeline Execution', example_1_basic_pipeline),
        '2': ('Custom Configuration', example_2_custom_configuration),
        '3': ('Minimal Descriptor Set', example_3_minimal_descriptors),
        '4': ('Load and Screen', example_4_load_and_screen),
        '5': ('Screen from CSV File', example_5_screen_from_file),
        '6': ('Compare Thresholds', example_6_compare_thresholds),
        '7': ('Ensemble Only', example_7_ensemble_only),
    }

    print("\nAvailable examples:")
    for key, (name, _) in examples.items():
        print(f"  {key}. {name}")

    print("\nOptions:")
    print("  - Enter example number (1-7) to run specific example")
    print("  - Enter 'all' to run all examples")
    print("  - Enter 'q' to quit")

    choice = input("\nYour choice: ").strip().lower()

    if choice == 'q':
        print("Exiting...")
        return

    if choice == 'all':
        for key, (name, func) in examples.items():
            try:
                func()
            except Exception as e:
                print(f"❌ Example {key} failed: {e}")
    elif choice in examples:
        try:
            examples[choice][1]()
        except Exception as e:
            print(f"❌ Example failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("Invalid choice!")


if __name__ == "__main__":
    main()
