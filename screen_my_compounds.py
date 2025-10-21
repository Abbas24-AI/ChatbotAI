#!/usr/bin/env python3
"""
Screen Your Own Compounds

This script shows you how to use a trained model to screen your own compounds.
"""

from ml_drug_screening import MLDrugScreeningPipeline
import pandas as pd

print("\n" + "="*80)
print("SCREEN YOUR OWN COMPOUNDS")
print("="*80)

# Step 1: Load your previously trained model
print("\n1. Loading trained model...")
try:
    pipeline = MLDrugScreeningPipeline.load_pipeline('ml_screening_results/pipeline.pkl')
    print("   ✓ Model loaded successfully!")
except FileNotFoundError:
    print("   ❌ No trained model found!")
    print("   Please run the main pipeline first: python ml_drug_screening.py")
    exit(1)

# Step 2: Prepare your compounds
print("\n2. Preparing compounds to screen...")

# METHOD A: List your SMILES directly in the code
your_smiles = [
    'CC(C)Cc1ccc(cc1)C(C)C(O)=O',  # Ibuprofen
    'CC(=O)Oc1ccccc1C(O)=O',        # Aspirin
    'CN1C=NC2=C1C(=O)N(C(=O)N2C)C', # Caffeine
    # Add your own SMILES here:
    # 'YOUR_SMILES_HERE',
    # 'ANOTHER_SMILES_HERE',
]

compound_ids = ['COMPOUND_1', 'COMPOUND_2', 'COMPOUND_3']  # Optional IDs

# METHOD B: Or read from a CSV file
# Uncomment these lines to read from a file:
# df = pd.read_csv('my_compounds.csv')  # CSV should have 'smiles' column
# your_smiles = df['smiles'].tolist()
# compound_ids = df['compound_id'].tolist() if 'compound_id' in df.columns else None

print(f"   ✓ Prepared {len(your_smiles)} compounds for screening")

# Step 3: Screen the compounds
print("\n3. Screening compounds...")
print("   (This may take a minute depending on the number of compounds)")

results, hits = pipeline.perform_virtual_screening(
    your_smiles,
    compound_ids
)

print(f"   ✓ Screening complete!")

# Step 4: Display results
print("\n4. Results:")
print("="*80)

print(f"\nTotal compounds screened: {len(results)}")
print(f"Potential hits identified: {len(hits)}")

if len(hits) > 0:
    print("\n🎯 HITS (Active predictions):")
    print("-" * 80)
    for idx, row in hits.iterrows():
        print(f"\nCompound: {row['compound_id']}")
        print(f"  SMILES: {row['smiles']}")
        print(f"  Predicted: {row['predicted_label']}")
        print(f"  Probability: {row['predicted_probability']:.3f}")
        print(f"  Confidence: {'High' if row['predicted_probability'] > 0.8 else 'Medium'}")
else:
    print("\n⚠️  No hits identified (no compounds predicted as active)")
    print("    Try adjusting the screening threshold in the configuration")

print("\n📊 All Results:")
print("-" * 80)
print(results.to_string(index=False))

# Step 5: Save results
results_file = 'my_screening_results.csv'
results.to_csv(results_file, index=False)
print(f"\n✓ Full results saved to: {results_file}")

if len(hits) > 0:
    hits_file = 'my_screening_hits.csv'
    hits.to_csv(hits_file, index=False)
    print(f"✓ Hits saved to: {hits_file}")

print("\n" + "="*80)
print("Screening complete! Check the CSV files for detailed results.")
print("="*80)
