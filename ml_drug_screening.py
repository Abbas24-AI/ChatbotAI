#!/usr/bin/env python3
"""
Production-Ready Machine Learning-Based Virtual Drug Screening Pipeline

This script provides an end-to-end automated pipeline for ligand-based virtual screening
using ChEMBL bioactivity data, machine learning classification, and ensemble modeling.

Author: ML Drug Discovery Pipeline
Date: 2025
Version: 1.0.0
"""

import sys
import os
import logging
import warnings
import pickle
import json
import time
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
import traceback

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Check and install dependencies
def install_dependencies():
    """Install all required dependencies if not already installed."""
    required_packages = [
        'chembl_webresource_client',
        'rdkit',
        'numpy',
        'pandas',
        'scikit-learn',
        'xgboost',
        'lightgbm',
        'matplotlib',
        'seaborn',
        'shap',
        'imbalanced-learn',
        'scipy',
        'joblib',
        'requests'
    ]

    print("Checking and installing required dependencies...")
    import subprocess

    for package in required_packages:
        try:
            if package == 'rdkit':
                __import__('rdkit')
            elif package == 'scikit-learn':
                __import__('sklearn')
            elif package == 'chembl_webresource_client':
                __import__('chembl_webresource_client')
            elif package == 'imbalanced-learn':
                __import__('imblearn')
            else:
                __import__(package)
        except ImportError:
            print(f"Installing {package}...")
            if package == 'rdkit':
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'rdkit-pypi'])
            else:
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])

    print("All dependencies installed successfully!\n")

# Install dependencies first
try:
    install_dependencies()
except Exception as e:
    print(f"Warning: Some dependencies may not be installed correctly: {e}")
    print("Continuing with available packages...\n")

# Import all required libraries
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import (
    StratifiedKFold, cross_validate, GridSearchCV, RandomizedSearchCV
)
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, precision_recall_curve, auc, matthews_corrcoef,
    confusion_matrix, classification_report, roc_curve, make_scorer
)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import joblib

try:
    from chembl_webresource_client.new_client import new_client
except ImportError:
    print("Warning: ChEMBL client not available. Install with: pip install chembl_webresource_client")
    new_client = None

try:
    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem, Descriptors, Lipinski, Crippen, MolFromSmiles, MolToSmiles
    from rdkit.Chem import MACCSkeys, rdMolDescriptors
    from rdkit.ML.Descriptors import MoleculeDescriptors
except ImportError:
    print("Warning: RDKit not available. Install with: pip install rdkit-pypi")
    Chem = None

try:
    import xgboost as xgb
except ImportError:
    print("Warning: XGBoost not available. Install with: pip install xgboost")
    xgb = None

try:
    import lightgbm as lgb
except ImportError:
    print("Warning: LightGBM not available. Install with: pip install lightgbm")
    lgb = None

try:
    import shap
except ImportError:
    print("Warning: SHAP not available. Install with: pip install shap")
    shap = None

try:
    from imblearn.over_sampling import SMOTE
    from imblearn.under_sampling import RandomUnderSampler
except ImportError:
    print("Warning: imbalanced-learn not available. Install with: pip install imbalanced-learn")
    SMOTE = None


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'ml_drug_screening_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration class for pipeline parameters."""

    # Random seed for reproducibility
    random_seed: int = 42

    # ChEMBL parameters
    activity_threshold_nm: float = 1000.0  # IC50 threshold for active/inactive (nM)
    min_compounds: int = 100  # Minimum number of compounds for target

    # Data preprocessing
    test_size: float = 0.2
    n_folds: int = 10
    batch_size: int = 10000  # For memory-efficient processing

    # Class imbalance handling
    handle_imbalance: bool = True
    imbalance_strategy: str = 'smote'  # 'smote', 'undersample', or 'none'
    max_imbalance_ratio: float = 3.0

    # Molecular descriptors
    use_morgan_fingerprints: bool = True
    morgan_radius: int = 2
    morgan_bits: int = 2048
    use_maccs_keys: bool = True
    use_rdkit_descriptors: bool = True
    use_lipinski_descriptors: bool = True

    # Model training
    models_to_train: List[str] = None
    cv_scoring: str = 'roc_auc'
    n_jobs: int = -1  # Use all available cores

    # Hyperparameter optimization
    hyperparam_search: str = 'random'  # 'grid', 'random', or 'none'
    hyperparam_iterations: int = 50
    hyperparam_cv: int = 5

    # Ensemble
    ensemble_top_n: int = 3

    # Virtual screening
    screening_threshold: float = 0.7  # Probability threshold for hits

    # Visualization
    figure_dpi: int = 300
    figure_format: str = 'png'

    # Output directories
    output_dir: str = 'ml_screening_results'
    models_dir: str = 'trained_models'
    figures_dir: str = 'figures'

    def __post_init__(self):
        """Set default values and create directories."""
        if self.models_to_train is None:
            self.models_to_train = ['rf', 'xgb', 'lgb', 'svm', 'mlp', 'gb']

        # Create output directories
        for dir_path in [self.output_dir, self.models_dir, self.figures_dir]:
            os.makedirs(dir_path, exist_ok=True)

    def save(self, filepath: str):
        """Save configuration to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(asdict(self), f, indent=4)

    @classmethod
    def load(cls, filepath: str):
        """Load configuration from JSON file."""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        return cls(**config_dict)


class ChEMBLDataRetriever:
    """Handles retrieval and processing of ChEMBL bioactivity data."""

    def __init__(self, config: PipelineConfig):
        """Initialize ChEMBL data retriever."""
        self.config = config
        self.target_client = new_client.target
        self.activity_client = new_client.activity
        np.random.seed(config.random_seed)
        logger.info("ChEMBL Data Retriever initialized")

    def search_targets(self, target_name: str, organism: str = 'Homo sapiens') -> pd.DataFrame:
        """
        Search for targets by name in ChEMBL.

        Args:
            target_name: Name or partial name of the target protein
            organism: Organism to filter targets (default: Homo sapiens)

        Returns:
            DataFrame containing matching targets
        """
        logger.info(f"Searching for targets matching '{target_name}' in {organism}...")

        try:
            targets = self.target_client.filter(
                target_synonym__icontains=target_name,
                organism=organism
            )

            targets_list = []
            for target in targets:
                targets_list.append({
                    'chembl_id': target['target_chembl_id'],
                    'pref_name': target['pref_name'],
                    'target_type': target['target_type'],
                    'organism': target['organism']
                })

            df = pd.DataFrame(targets_list)
            logger.info(f"Found {len(df)} matching targets")
            return df

        except Exception as e:
            logger.error(f"Error searching targets: {e}")
            raise

    def select_target_interactive(self, targets_df: pd.DataFrame) -> str:
        """
        Interactive target selection from search results.

        Args:
            targets_df: DataFrame of target search results

        Returns:
            Selected target ChEMBL ID
        """
        if len(targets_df) == 0:
            raise ValueError("No targets found. Please try a different search term.")

        print("\n" + "="*80)
        print("AVAILABLE TARGETS")
        print("="*80)
        for idx, row in targets_df.iterrows():
            print(f"{idx + 1}. {row['chembl_id']} - {row['pref_name']}")
            print(f"   Type: {row['target_type']}, Organism: {row['organism']}")
            print()

        while True:
            try:
                choice = input(f"Select target (1-{len(targets_df)}) or 'q' to quit: ").strip()
                if choice.lower() == 'q':
                    raise KeyboardInterrupt("User cancelled target selection")

                choice_idx = int(choice) - 1
                if 0 <= choice_idx < len(targets_df):
                    selected_target = targets_df.iloc[choice_idx]['chembl_id']
                    logger.info(f"Selected target: {selected_target}")
                    return selected_target
                else:
                    print(f"Please enter a number between 1 and {len(targets_df)}")
            except ValueError:
                print("Invalid input. Please enter a number.")
            except KeyboardInterrupt:
                raise

    def retrieve_bioactivity_data(
        self,
        target_chembl_id: str,
        bioactivity_type: str = 'IC50'
    ) -> pd.DataFrame:
        """
        Retrieve bioactivity data for a target from ChEMBL.

        Args:
            target_chembl_id: ChEMBL ID of the target
            bioactivity_type: Type of bioactivity (default: IC50)

        Returns:
            DataFrame containing bioactivity data
        """
        logger.info(f"Retrieving {bioactivity_type} data for target {target_chembl_id}...")

        try:
            activities = self.activity_client.filter(
                target_chembl_id=target_chembl_id,
                standard_type=bioactivity_type,
                standard_relation='=',
                assay_type='B'  # Binding assay
            ).only([
                'molecule_chembl_id',
                'canonical_smiles',
                'standard_value',
                'standard_units',
                'standard_type',
                'assay_chembl_id',
                'assay_description',
                'target_pref_name'
            ])

            # Convert to list with progress indication
            activities_list = []
            count = 0
            for activity in activities:
                activities_list.append(activity)
                count += 1
                if count % 100 == 0:
                    print(f"Retrieved {count} activities...", end='\r')

            print(f"Retrieved {count} activities total.          ")

            df = pd.DataFrame(activities_list)
            logger.info(f"Retrieved {len(df)} bioactivity records")

            return df

        except Exception as e:
            logger.error(f"Error retrieving bioactivity data: {e}")
            raise

    def preprocess_bioactivity_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess bioactivity data: handle missing values, duplicates, and convert units.

        Args:
            df: Raw bioactivity DataFrame

        Returns:
            Preprocessed DataFrame
        """
        logger.info("Preprocessing bioactivity data...")

        initial_count = len(df)

        # Remove rows with missing SMILES or bioactivity values
        df = df.dropna(subset=['canonical_smiles', 'standard_value'])
        logger.info(f"Removed {initial_count - len(df)} rows with missing SMILES or values")

        # Convert standard_value to numeric
        df['standard_value'] = pd.to_numeric(df['standard_value'], errors='coerce')
        df = df.dropna(subset=['standard_value'])

        # Remove invalid or extremely high/low values (likely errors)
        df = df[df['standard_value'] > 0]
        df = df[df['standard_value'] < 1e10]

        # Convert units to nM if needed
        if 'standard_units' in df.columns:
            # Handle different unit types
            df.loc[df['standard_units'] == 'uM', 'standard_value'] *= 1000  # uM to nM
            df.loc[df['standard_units'] == 'mM', 'standard_value'] *= 1e6  # mM to nM
            df.loc[df['standard_units'] == 'pM', 'standard_value'] /= 1000  # pM to nM

        # Remove duplicates (keep the one with lower IC50 = more active)
        df = df.sort_values('standard_value')
        df = df.drop_duplicates(subset='canonical_smiles', keep='first')

        # Reset index
        df = df.reset_index(drop=True)

        logger.info(f"Preprocessing complete. {len(df)} compounds remaining.")

        return df

    def assign_activity_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert IC50 values to binary activity labels.

        Args:
            df: DataFrame with IC50 values

        Returns:
            DataFrame with activity labels
        """
        logger.info(f"Assigning activity labels (threshold: {self.config.activity_threshold_nm} nM)...")

        # Active if IC50 <= threshold, Inactive otherwise
        df['activity_class'] = (df['standard_value'] <= self.config.activity_threshold_nm).astype(int)
        df['activity_label'] = df['activity_class'].map({1: 'Active', 0: 'Inactive'})

        # Calculate -log10(IC50 in M) for reference
        df['pIC50'] = -np.log10(df['standard_value'] * 1e-9)

        # Report class distribution
        class_counts = df['activity_label'].value_counts()
        logger.info(f"Activity distribution:\n{class_counts}")

        # Check class imbalance
        if len(class_counts) == 2:
            ratio = max(class_counts) / min(class_counts)
            logger.info(f"Class imbalance ratio: {ratio:.2f}")

            if ratio > self.config.max_imbalance_ratio:
                logger.warning(
                    f"Class imbalance ratio ({ratio:.2f}) exceeds threshold "
                    f"({self.config.max_imbalance_ratio}). Consider using SMOTE or adjusting threshold."
                )

        return df


class MolecularDescriptorCalculator:
    """Calculates molecular descriptors and fingerprints."""

    def __init__(self, config: PipelineConfig):
        """Initialize descriptor calculator."""
        self.config = config
        np.random.seed(config.random_seed)
        logger.info("Molecular Descriptor Calculator initialized")

    def smiles_to_mol(self, smiles: str) -> Optional[Chem.Mol]:
        """
        Convert SMILES to RDKit molecule object with error handling.

        Args:
            smiles: SMILES string

        Returns:
            RDKit Mol object or None if conversion fails
        """
        try:
            mol = MolFromSmiles(smiles)
            if mol is not None:
                return mol
        except:
            pass
        return None

    def calculate_lipinski_descriptors(self, mol: Chem.Mol) -> Dict[str, float]:
        """Calculate Lipinski's Rule of 5 descriptors."""
        try:
            return {
                'MolWt': Descriptors.MolWt(mol),
                'LogP': Descriptors.MolLogP(mol),
                'NumHDonors': Descriptors.NumHDonors(mol),
                'NumHAcceptors': Descriptors.NumHAcceptors(mol),
                'TPSA': Descriptors.TPSA(mol),
                'NumRotatableBonds': Descriptors.NumRotatableBonds(mol),
                'NumAromaticRings': Descriptors.NumAromaticRings(mol),
                'NumAliphaticRings': Descriptors.NumAliphaticRings(mol),
            }
        except Exception as e:
            logger.debug(f"Error calculating Lipinski descriptors: {e}")
            return {}

    def calculate_morgan_fingerprint(
        self,
        mol: Chem.Mol,
        radius: int = 2,
        n_bits: int = 2048
    ) -> np.ndarray:
        """Calculate Morgan (circular) fingerprint."""
        try:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
            arr = np.zeros((n_bits,), dtype=int)
            DataStructs.ConvertToNumpyArray(fp, arr)
            return arr
        except Exception as e:
            logger.debug(f"Error calculating Morgan fingerprint: {e}")
            return np.zeros(n_bits, dtype=int)

    def calculate_maccs_keys(self, mol: Chem.Mol) -> np.ndarray:
        """Calculate MACCS keys fingerprint."""
        try:
            fp = MACCSkeys.GenMACCSKeys(mol)
            arr = np.zeros((167,), dtype=int)  # MACCS keys have 167 bits
            DataStructs.ConvertToNumpyArray(fp, arr)
            return arr
        except Exception as e:
            logger.debug(f"Error calculating MACCS keys: {e}")
            return np.zeros(167, dtype=int)

    def calculate_rdkit_descriptors(self, mol: Chem.Mol) -> Dict[str, float]:
        """Calculate comprehensive RDKit molecular descriptors."""
        try:
            # Get all available descriptors
            descriptor_names = [desc[0] for desc in Descriptors._descList]
            calc = MoleculeDescriptors.MolecularDescriptorCalculator(descriptor_names)
            descriptors = calc.CalcDescriptors(mol)

            desc_dict = {}
            for name, value in zip(descriptor_names, descriptors):
                # Handle NaN and inf values
                if np.isfinite(value):
                    desc_dict[f'RDKit_{name}'] = value
                else:
                    desc_dict[f'RDKit_{name}'] = 0.0

            return desc_dict
        except Exception as e:
            logger.debug(f"Error calculating RDKit descriptors: {e}")
            return {}

    def calculate_all_descriptors(self, smiles_list: List[str]) -> pd.DataFrame:
        """
        Calculate all molecular descriptors for a list of SMILES.

        Args:
            smiles_list: List of SMILES strings

        Returns:
            DataFrame containing all calculated descriptors
        """
        logger.info(f"Calculating molecular descriptors for {len(smiles_list)} compounds...")

        descriptors_list = []
        failed_count = 0

        for idx, smiles in enumerate(smiles_list):
            if (idx + 1) % 100 == 0:
                print(f"Processing compound {idx + 1}/{len(smiles_list)}...", end='\r')

            mol = self.smiles_to_mol(smiles)

            if mol is None:
                logger.debug(f"Failed to parse SMILES: {smiles}")
                failed_count += 1
                descriptors_list.append({})
                continue

            desc_dict = {'smiles': smiles}

            # Lipinski descriptors
            if self.config.use_lipinski_descriptors:
                desc_dict.update(self.calculate_lipinski_descriptors(mol))

            # Morgan fingerprints
            if self.config.use_morgan_fingerprints:
                morgan_fp = self.calculate_morgan_fingerprint(
                    mol,
                    radius=self.config.morgan_radius,
                    n_bits=self.config.morgan_bits
                )
                for i, bit in enumerate(morgan_fp):
                    desc_dict[f'Morgan_{i}'] = bit

            # MACCS keys
            if self.config.use_maccs_keys:
                maccs_fp = self.calculate_maccs_keys(mol)
                for i, bit in enumerate(maccs_fp):
                    desc_dict[f'MACCS_{i}'] = bit

            # RDKit descriptors (comprehensive)
            if self.config.use_rdkit_descriptors:
                desc_dict.update(self.calculate_rdkit_descriptors(mol))

            descriptors_list.append(desc_dict)

        print(f"Processed all {len(smiles_list)} compounds.          ")

        if failed_count > 0:
            logger.warning(f"Failed to calculate descriptors for {failed_count} compounds")

        # Convert to DataFrame
        df = pd.DataFrame(descriptors_list)

        # Remove rows with no descriptors (failed molecules)
        df = df.dropna(subset=['smiles'])

        logger.info(f"Calculated {len(df.columns) - 1} descriptors for {len(df)} compounds")

        return df


class ModelTrainer:
    """Handles training, evaluation, and optimization of ML models."""

    def __init__(self, config: PipelineConfig):
        """Initialize model trainer."""
        self.config = config
        np.random.seed(config.random_seed)
        self.scaler = None
        self.feature_names = None
        logger.info("Model Trainer initialized")

    def get_base_models(self) -> Dict[str, Any]:
        """Get dictionary of base models to train."""
        models = {}

        if 'rf' in self.config.models_to_train:
            models['RandomForest'] = RandomForestClassifier(
                n_estimators=100,
                random_state=self.config.random_seed,
                n_jobs=self.config.n_jobs,
                class_weight='balanced'
            )

        if 'xgb' in self.config.models_to_train and xgb is not None:
            models['XGBoost'] = xgb.XGBClassifier(
                n_estimators=100,
                random_state=self.config.random_seed,
                n_jobs=self.config.n_jobs,
                eval_metric='logloss'
            )

        if 'lgb' in self.config.models_to_train and lgb is not None:
            models['LightGBM'] = lgb.LGBMClassifier(
                n_estimators=100,
                random_state=self.config.random_seed,
                n_jobs=self.config.n_jobs,
                verbose=-1
            )

        if 'gb' in self.config.models_to_train:
            models['GradientBoosting'] = GradientBoostingClassifier(
                n_estimators=100,
                random_state=self.config.random_seed
            )

        if 'svm' in self.config.models_to_train:
            models['SVM'] = SVC(
                kernel='rbf',
                probability=True,
                random_state=self.config.random_seed,
                class_weight='balanced'
            )

        if 'mlp' in self.config.models_to_train:
            models['NeuralNetwork'] = MLPClassifier(
                hidden_layer_sizes=(128, 64, 32),
                random_state=self.config.random_seed,
                max_iter=500,
                early_stopping=True
            )

        logger.info(f"Initialized {len(models)} base models: {list(models.keys())}")
        return models

    def get_hyperparameter_grids(self) -> Dict[str, Dict]:
        """Get hyperparameter grids for each model type."""
        grids = {
            'RandomForest': {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, 30, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2']
            },
            'XGBoost': {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 5, 7, 10],
                'learning_rate': [0.01, 0.1, 0.3],
                'subsample': [0.6, 0.8, 1.0],
                'colsample_bytree': [0.6, 0.8, 1.0]
            },
            'LightGBM': {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 5, 7, 10],
                'learning_rate': [0.01, 0.1, 0.3],
                'num_leaves': [20, 31, 50],
                'subsample': [0.6, 0.8, 1.0]
            },
            'GradientBoosting': {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.3],
                'subsample': [0.6, 0.8, 1.0]
            },
            'SVM': {
                'C': [0.1, 1, 10, 100],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
                'kernel': ['rbf', 'poly']
            },
            'NeuralNetwork': {
                'hidden_layer_sizes': [(64, 32), (128, 64), (128, 64, 32), (256, 128, 64)],
                'activation': ['relu', 'tanh'],
                'alpha': [0.0001, 0.001, 0.01],
                'learning_rate': ['constant', 'adaptive']
            }
        }
        return grids

    def preprocess_features(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        fit_scaler: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess features: handle missing values, scale, and handle class imbalance.

        Args:
            X: Feature DataFrame
            y: Target Series
            fit_scaler: Whether to fit the scaler (True for training, False for test)

        Returns:
            Preprocessed X and y as numpy arrays
        """
        logger.info("Preprocessing features...")

        # Store feature names
        if fit_scaler:
            self.feature_names = X.columns.tolist()

        # Handle missing values
        X = X.fillna(0)

        # Remove constant features
        if fit_scaler:
            variance = X.var()
            constant_features = variance[variance == 0].index
            if len(constant_features) > 0:
                logger.info(f"Removing {len(constant_features)} constant features")
                X = X.drop(columns=constant_features)
                self.feature_names = X.columns.tolist()

        # Scale features
        if fit_scaler:
            self.scaler = RobustScaler()
            X_scaled = self.scaler.fit_transform(X)
            logger.info("Fitted feature scaler")
        else:
            if self.scaler is None:
                raise ValueError("Scaler not fitted. Call with fit_scaler=True first.")
            X_scaled = self.scaler.transform(X)

        # Handle class imbalance
        if fit_scaler and self.config.handle_imbalance:
            class_counts = pd.Series(y).value_counts()
            ratio = max(class_counts) / min(class_counts)

            if ratio > self.config.max_imbalance_ratio:
                logger.info(f"Handling class imbalance (ratio: {ratio:.2f})...")

                if self.config.imbalance_strategy == 'smote' and SMOTE is not None:
                    smote = SMOTE(random_state=self.config.random_seed)
                    X_scaled, y = smote.fit_resample(X_scaled, y)
                    logger.info(f"Applied SMOTE. New shape: {X_scaled.shape}")

                elif self.config.imbalance_strategy == 'undersample':
                    rus = RandomUnderSampler(random_state=self.config.random_seed)
                    X_scaled, y = rus.fit_resample(X_scaled, y)
                    logger.info(f"Applied undersampling. New shape: {X_scaled.shape}")

        return X_scaled, np.array(y)

    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive evaluation metrics."""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='binary', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='binary', zero_division=0),
            'f1': f1_score(y_true, y_pred, average='binary', zero_division=0),
            'roc_auc': roc_auc_score(y_true, y_prob),
            'mcc': matthews_corrcoef(y_true, y_pred)
        }

        # Calculate precision-recall AUC
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        metrics['pr_auc'] = auc(recall, precision)

        return metrics

    def train_with_cross_validation(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> Dict[str, Dict[str, Any]]:
        """
        Train all base models with 10-fold cross-validation.

        Args:
            X: Feature matrix
            y: Target vector

        Returns:
            Dictionary of model results
        """
        logger.info(f"Training models with {self.config.n_folds}-fold cross-validation...")

        models = self.get_base_models()
        results = {}

        # Define scoring metrics
        scoring = {
            'accuracy': 'accuracy',
            'precision': 'precision',
            'recall': 'recall',
            'f1': 'f1',
            'roc_auc': 'roc_auc'
        }

        cv = StratifiedKFold(
            n_splits=self.config.n_folds,
            shuffle=True,
            random_state=self.config.random_seed
        )

        for name, model in models.items():
            logger.info(f"\nTraining {name}...")
            start_time = time.time()

            try:
                # Perform cross-validation
                cv_results = cross_validate(
                    model, X, y,
                    cv=cv,
                    scoring=scoring,
                    n_jobs=self.config.n_jobs,
                    return_train_score=True,
                    return_estimator=True
                )

                # Calculate mean and std of metrics
                metrics = {}
                for metric in scoring.keys():
                    metrics[f'{metric}_mean'] = cv_results[f'test_{metric}'].mean()
                    metrics[f'{metric}_std'] = cv_results[f'test_{metric}'].std()

                # Train final model on full dataset
                model.fit(X, y)

                # Get predictions for full dataset
                y_pred = model.predict(X)
                y_prob = model.predict_proba(X)[:, 1]

                # Calculate confusion matrix
                cm = confusion_matrix(y, y_pred)

                elapsed_time = time.time() - start_time

                results[name] = {
                    'model': model,
                    'cv_results': cv_results,
                    'metrics': metrics,
                    'predictions': y_pred,
                    'probabilities': y_prob,
                    'confusion_matrix': cm,
                    'training_time': elapsed_time
                }

                logger.info(f"{name} completed in {elapsed_time:.2f}s")
                logger.info(f"  ROC-AUC: {metrics['roc_auc_mean']:.4f} ± {metrics['roc_auc_std']:.4f}")
                logger.info(f"  Accuracy: {metrics['accuracy_mean']:.4f} ± {metrics['accuracy_std']:.4f}")

            except Exception as e:
                logger.error(f"Error training {name}: {e}")
                traceback.print_exc()

        return results

    def optimize_hyperparameters(
        self,
        model_name: str,
        base_model: Any,
        X: np.ndarray,
        y: np.ndarray
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Optimize hyperparameters for a model using RandomizedSearchCV or GridSearchCV.

        Args:
            model_name: Name of the model
            base_model: Base model instance
            X: Feature matrix
            y: Target vector

        Returns:
            Tuple of (best_model, best_params)
        """
        logger.info(f"Optimizing hyperparameters for {model_name}...")

        param_grids = self.get_hyperparameter_grids()

        if model_name not in param_grids:
            logger.warning(f"No hyperparameter grid defined for {model_name}")
            return base_model, {}

        param_grid = param_grids[model_name]

        cv = StratifiedKFold(
            n_splits=self.config.hyperparam_cv,
            shuffle=True,
            random_state=self.config.random_seed
        )

        try:
            if self.config.hyperparam_search == 'random':
                search = RandomizedSearchCV(
                    base_model,
                    param_grid,
                    n_iter=self.config.hyperparam_iterations,
                    cv=cv,
                    scoring=self.config.cv_scoring,
                    n_jobs=self.config.n_jobs,
                    random_state=self.config.random_seed,
                    verbose=1
                )
            elif self.config.hyperparam_search == 'grid':
                search = GridSearchCV(
                    base_model,
                    param_grid,
                    cv=cv,
                    scoring=self.config.cv_scoring,
                    n_jobs=self.config.n_jobs,
                    verbose=1
                )
            else:
                logger.warning(f"Unknown search strategy: {self.config.hyperparam_search}")
                return base_model, {}

            search.fit(X, y)

            logger.info(f"Best {self.config.cv_scoring}: {search.best_score_:.4f}")
            logger.info(f"Best parameters: {search.best_params_}")

            return search.best_estimator_, search.best_params_

        except Exception as e:
            logger.error(f"Error optimizing {model_name}: {e}")
            traceback.print_exc()
            return base_model, {}

    def build_ensemble_model(
        self,
        top_models: List[Tuple[str, Any]],
        X: np.ndarray,
        y: np.ndarray
    ) -> Any:
        """
        Build stacking ensemble from top performing models.

        Args:
            top_models: List of (name, model) tuples for top models
            X: Feature matrix
            y: Target vector

        Returns:
            Trained stacking ensemble model
        """
        logger.info(f"Building stacking ensemble from top {len(top_models)} models...")

        estimators = [(name, model) for name, model in top_models]

        # Use logistic regression as meta-classifier
        meta_classifier = LogisticRegression(
            random_state=self.config.random_seed,
            max_iter=1000
        )

        ensemble = StackingClassifier(
            estimators=estimators,
            final_estimator=meta_classifier,
            cv=5,
            n_jobs=self.config.n_jobs
        )

        ensemble.fit(X, y)

        logger.info("Ensemble model trained successfully")

        return ensemble


class VirtualScreener:
    """Handles virtual screening of compound libraries."""

    def __init__(self, config: PipelineConfig):
        """Initialize virtual screener."""
        self.config = config
        self.descriptor_calculator = MolecularDescriptorCalculator(config)
        logger.info("Virtual Screener initialized")

    def screen_compounds(
        self,
        model: Any,
        scaler: Any,
        feature_names: List[str],
        smiles_list: List[str],
        compound_ids: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Screen a list of compounds using the trained model.

        Args:
            model: Trained ML model
            scaler: Fitted feature scaler
            feature_names: List of feature names used in training
            smiles_list: List of SMILES strings to screen
            compound_ids: Optional list of compound IDs

        Returns:
            DataFrame with screening results
        """
        logger.info(f"Screening {len(smiles_list)} compounds...")

        if compound_ids is None:
            compound_ids = [f"COMPOUND_{i+1}" for i in range(len(smiles_list))]

        # Calculate descriptors
        descriptors_df = self.descriptor_calculator.calculate_all_descriptors(smiles_list)

        # Merge with compound IDs
        descriptors_df['compound_id'] = compound_ids[:len(descriptors_df)]

        # Align features with training features
        X = descriptors_df[feature_names].fillna(0)

        # Scale features
        X_scaled = scaler.transform(X)

        # Make predictions
        predictions = model.predict(X_scaled)
        probabilities = model.predict_proba(X_scaled)[:, 1]

        # Create results DataFrame
        results = pd.DataFrame({
            'compound_id': descriptors_df['compound_id'],
            'smiles': descriptors_df['smiles'],
            'predicted_class': predictions,
            'predicted_probability': probabilities,
            'predicted_label': ['Active' if p == 1 else 'Inactive' for p in predictions]
        })

        # Filter hits based on threshold
        hits = results[results['predicted_probability'] >= self.config.screening_threshold]

        logger.info(f"Identified {len(hits)} hits (threshold: {self.config.screening_threshold})")

        return results, hits


class Visualizer:
    """Generates publication-quality figures."""

    def __init__(self, config: PipelineConfig):
        """Initialize visualizer."""
        self.config = config
        self.figures_dir = config.figures_dir

        # Set publication-quality defaults
        plt.rcParams['figure.dpi'] = config.figure_dpi
        plt.rcParams['savefig.dpi'] = config.figure_dpi
        plt.rcParams['font.size'] = 10
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['axes.linewidth'] = 1.5

        sns.set_style("whitegrid")
        logger.info("Visualizer initialized")

    def plot_roc_curves(
        self,
        results: Dict[str, Dict],
        y_true: np.ndarray,
        filename: str = 'roc_curves.png'
    ):
        """Plot ROC curves for all models."""
        plt.figure(figsize=(10, 8))

        for name, result in results.items():
            y_prob = result['probabilities']
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            roc_auc = result['metrics']['roc_auc_mean']

            plt.plot(fpr, tpr, linewidth=2,
                    label=f'{name} (AUC = {roc_auc:.3f})')

        plt.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random (AUC = 0.500)')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
        plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
        plt.title('Receiver Operating Characteristic (ROC) Curves', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right", fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        filepath = os.path.join(self.figures_dir, filename)
        plt.savefig(filepath, dpi=self.config.figure_dpi, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved ROC curves to {filepath}")

    def plot_precision_recall_curves(
        self,
        results: Dict[str, Dict],
        y_true: np.ndarray,
        filename: str = 'precision_recall_curves.png'
    ):
        """Plot precision-recall curves for all models."""
        plt.figure(figsize=(10, 8))

        for name, result in results.items():
            y_prob = result['probabilities']
            precision, recall, _ = precision_recall_curve(y_true, y_prob)
            pr_auc = auc(recall, precision)

            plt.plot(recall, precision, linewidth=2,
                    label=f'{name} (AUC = {pr_auc:.3f})')

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall', fontsize=12, fontweight='bold')
        plt.ylabel('Precision', fontsize=12, fontweight='bold')
        plt.title('Precision-Recall Curves', fontsize=14, fontweight='bold')
        plt.legend(loc="best", fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        filepath = os.path.join(self.figures_dir, filename)
        plt.savefig(filepath, dpi=self.config.figure_dpi, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved precision-recall curves to {filepath}")

    def plot_confusion_matrices(
        self,
        results: Dict[str, Dict],
        filename: str = 'confusion_matrices.png'
    ):
        """Plot confusion matrices for all models."""
        n_models = len(results)
        n_cols = min(3, n_models)
        n_rows = (n_models + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        if n_models == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        for idx, (name, result) in enumerate(results.items()):
            cm = result['confusion_matrix']

            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       ax=axes[idx], cbar=False,
                       xticklabels=['Inactive', 'Active'],
                       yticklabels=['Inactive', 'Active'])

            axes[idx].set_title(name, fontsize=12, fontweight='bold')
            axes[idx].set_ylabel('True Label', fontsize=10)
            axes[idx].set_xlabel('Predicted Label', fontsize=10)

        # Hide unused subplots
        for idx in range(n_models, len(axes)):
            axes[idx].axis('off')

        plt.tight_layout()
        filepath = os.path.join(self.figures_dir, filename)
        plt.savefig(filepath, dpi=self.config.figure_dpi, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved confusion matrices to {filepath}")

    def plot_feature_importance(
        self,
        model: Any,
        feature_names: List[str],
        model_name: str = 'Model',
        top_n: int = 20,
        filename: str = 'feature_importance.png'
    ):
        """Plot feature importance for tree-based models."""
        try:
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                indices = np.argsort(importances)[::-1][:top_n]

                plt.figure(figsize=(10, 8))
                plt.barh(range(top_n), importances[indices])
                plt.yticks(range(top_n), [feature_names[i] for i in indices])
                plt.xlabel('Feature Importance', fontsize=12, fontweight='bold')
                plt.ylabel('Feature', fontsize=12, fontweight='bold')
                plt.title(f'Top {top_n} Feature Importances - {model_name}',
                         fontsize=14, fontweight='bold')
                plt.gca().invert_yaxis()
                plt.tight_layout()

                filepath = os.path.join(self.figures_dir, filename)
                plt.savefig(filepath, dpi=self.config.figure_dpi, bbox_inches='tight')
                plt.close()
                logger.info(f"Saved feature importance plot to {filepath}")
            else:
                logger.warning(f"Model {model_name} does not have feature_importances_ attribute")
        except Exception as e:
            logger.error(f"Error plotting feature importance: {e}")

    def plot_shap_summary(
        self,
        model: Any,
        X: np.ndarray,
        feature_names: List[str],
        model_name: str = 'Model',
        filename: str = 'shap_summary.png'
    ):
        """Generate SHAP summary plot."""
        if shap is None:
            logger.warning("SHAP not available. Skipping SHAP plots.")
            return

        try:
            logger.info(f"Calculating SHAP values for {model_name}...")

            # Sample data if too large
            if len(X) > 1000:
                sample_indices = np.random.choice(len(X), 1000, replace=False)
                X_sample = X[sample_indices]
            else:
                X_sample = X

            # Create SHAP explainer based on model type
            if hasattr(model, 'predict_proba'):
                if 'XGB' in model_name or 'LightGBM' in model_name or 'RandomForest' in model_name:
                    explainer = shap.TreeExplainer(model)
                else:
                    explainer = shap.KernelExplainer(model.predict_proba, X_sample[:100])
            else:
                explainer = shap.KernelExplainer(model.predict, X_sample[:100])

            shap_values = explainer.shap_values(X_sample)

            # Handle multi-output SHAP values
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # Use positive class

            plt.figure(figsize=(10, 8))
            shap.summary_plot(
                shap_values,
                X_sample,
                feature_names=feature_names,
                show=False,
                max_display=20
            )
            plt.title(f'SHAP Feature Importance - {model_name}',
                     fontsize=14, fontweight='bold', pad=20)
            plt.tight_layout()

            filepath = os.path.join(self.figures_dir, filename)
            plt.savefig(filepath, dpi=self.config.figure_dpi, bbox_inches='tight')
            plt.close()
            logger.info(f"Saved SHAP summary plot to {filepath}")

        except Exception as e:
            logger.error(f"Error generating SHAP plot for {model_name}: {e}")
            traceback.print_exc()

    def plot_model_comparison(
        self,
        results: Dict[str, Dict],
        filename: str = 'model_comparison.png'
    ):
        """Plot comprehensive model comparison."""
        metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()

        for idx, metric in enumerate(metrics_to_plot):
            model_names = []
            means = []
            stds = []

            for name, result in results.items():
                model_names.append(name)
                means.append(result['metrics'][f'{metric}_mean'])
                stds.append(result['metrics'][f'{metric}_std'])

            x = np.arange(len(model_names))
            axes[idx].bar(x, means, yerr=stds, capsize=5, alpha=0.7, color='steelblue')
            axes[idx].set_xticks(x)
            axes[idx].set_xticklabels(model_names, rotation=45, ha='right')
            axes[idx].set_ylabel(metric.upper(), fontweight='bold')
            axes[idx].set_title(f'{metric.upper()} Comparison', fontweight='bold')
            axes[idx].set_ylim([0, 1.1])
            axes[idx].grid(True, alpha=0.3, axis='y')

        # Hide the last subplot
        axes[-1].axis('off')

        plt.tight_layout()
        filepath = os.path.join(self.figures_dir, filename)
        plt.savefig(filepath, dpi=self.config.figure_dpi, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved model comparison plot to {filepath}")

    def plot_activity_distribution(
        self,
        df: pd.DataFrame,
        filename: str = 'activity_distribution.png'
    ):
        """Plot distribution of IC50 values and activity classes."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # IC50 distribution
        axes[0].hist(np.log10(df['standard_value']), bins=50, edgecolor='black', alpha=0.7)
        axes[0].set_xlabel('log10(IC50) [nM]', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Frequency', fontsize=12, fontweight='bold')
        axes[0].set_title('IC50 Value Distribution', fontsize=14, fontweight='bold')
        axes[0].axvline(np.log10(1000), color='red', linestyle='--', linewidth=2,
                       label='Activity Threshold (1000 nM)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Activity class distribution
        class_counts = df['activity_label'].value_counts()
        axes[1].bar(class_counts.index, class_counts.values, alpha=0.7,
                   color=['green', 'orange'], edgecolor='black')
        axes[1].set_xlabel('Activity Class', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Count', fontsize=12, fontweight='bold')
        axes[1].set_title('Activity Class Distribution', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3, axis='y')

        # Add counts on bars
        for i, (label, count) in enumerate(class_counts.items()):
            axes[1].text(i, count, str(count), ha='center', va='bottom', fontweight='bold')

        plt.tight_layout()
        filepath = os.path.join(self.figures_dir, filename)
        plt.savefig(filepath, dpi=self.config.figure_dpi, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved activity distribution plot to {filepath}")


class MLDrugScreeningPipeline:
    """Main pipeline orchestrator."""

    def __init__(self, config: Optional[PipelineConfig] = None):
        """Initialize the pipeline."""
        self.config = config or PipelineConfig()

        # Set random seeds for reproducibility
        np.random.seed(self.config.random_seed)

        # Initialize components
        self.chembl_retriever = ChEMBLDataRetriever(self.config)
        self.descriptor_calculator = MolecularDescriptorCalculator(self.config)
        self.model_trainer = ModelTrainer(self.config)
        self.virtual_screener = VirtualScreener(self.config)
        self.visualizer = Visualizer(self.config)

        # Storage for pipeline artifacts
        self.bioactivity_data = None
        self.descriptors_data = None
        self.training_results = None
        self.ensemble_model = None

        logger.info("ML Drug Screening Pipeline initialized")
        logger.info(f"Configuration: {self.config}")

    def retrieve_target_data(self, target_name: str, auto_select: bool = False) -> pd.DataFrame:
        """
        Retrieve bioactivity data for a target.

        Args:
            target_name: Name of the target protein
            auto_select: If True, automatically select first target; if False, interactive selection

        Returns:
            Bioactivity DataFrame
        """
        logger.info(f"\n{'='*80}\nSTEP 1: Target Data Retrieval\n{'='*80}")

        # Search for targets
        targets_df = self.chembl_retriever.search_targets(target_name)

        if len(targets_df) == 0:
            raise ValueError(f"No targets found for '{target_name}'")

        # Select target
        if auto_select:
            target_chembl_id = targets_df.iloc[0]['chembl_id']
            logger.info(f"Auto-selected target: {target_chembl_id}")
        else:
            target_chembl_id = self.chembl_retriever.select_target_interactive(targets_df)

        # Retrieve bioactivity data
        bioactivity_df = self.chembl_retriever.retrieve_bioactivity_data(target_chembl_id)

        # Preprocess data
        bioactivity_df = self.chembl_retriever.preprocess_bioactivity_data(bioactivity_df)

        # Assign activity labels
        bioactivity_df = self.chembl_retriever.assign_activity_labels(bioactivity_df)

        # Check minimum compounds
        if len(bioactivity_df) < self.config.min_compounds:
            logger.warning(
                f"Only {len(bioactivity_df)} compounds found. "
                f"Minimum recommended: {self.config.min_compounds}"
            )

        self.bioactivity_data = bioactivity_df

        # Save bioactivity data
        output_path = os.path.join(self.config.output_dir, 'bioactivity_data.csv')
        bioactivity_df.to_csv(output_path, index=False)
        logger.info(f"Saved bioactivity data to {output_path}")

        return bioactivity_df

    def calculate_descriptors(self) -> pd.DataFrame:
        """Calculate molecular descriptors for all compounds."""
        logger.info(f"\n{'='*80}\nSTEP 2: Molecular Descriptor Calculation\n{'='*80}")

        if self.bioactivity_data is None:
            raise ValueError("No bioactivity data available. Run retrieve_target_data() first.")

        # Calculate descriptors
        smiles_list = self.bioactivity_data['canonical_smiles'].tolist()
        descriptors_df = self.descriptor_calculator.calculate_all_descriptors(smiles_list)

        # Merge with activity labels
        merged_df = pd.merge(
            descriptors_df,
            self.bioactivity_data[['canonical_smiles', 'activity_class', 'activity_label', 'pIC50']],
            left_on='smiles',
            right_on='canonical_smiles',
            how='inner'
        )

        self.descriptors_data = merged_df

        # Save descriptors
        output_path = os.path.join(self.config.output_dir, 'molecular_descriptors.csv')
        merged_df.to_csv(output_path, index=False)
        logger.info(f"Saved molecular descriptors to {output_path}")

        return merged_df

    def train_models(self) -> Dict[str, Dict]:
        """Train all models with cross-validation."""
        logger.info(f"\n{'='*80}\nSTEP 3: Model Training\n{'='*80}")

        if self.descriptors_data is None:
            raise ValueError("No descriptor data available. Run calculate_descriptors() first.")

        # Prepare features and target
        feature_cols = [col for col in self.descriptors_data.columns
                       if col not in ['smiles', 'canonical_smiles', 'activity_class',
                                     'activity_label', 'pIC50']]

        X = self.descriptors_data[feature_cols]
        y = self.descriptors_data['activity_class']

        logger.info(f"Training set: {len(X)} samples, {len(feature_cols)} features")

        # Preprocess features
        X_processed, y_processed = self.model_trainer.preprocess_features(X, y, fit_scaler=True)

        # Train models with cross-validation
        results = self.model_trainer.train_with_cross_validation(X_processed, y_processed)

        self.training_results = results

        # Generate visualizations
        self.visualizer.plot_activity_distribution(self.bioactivity_data)
        self.visualizer.plot_model_comparison(results)
        self.visualizer.plot_roc_curves(results, y_processed)
        self.visualizer.plot_precision_recall_curves(results, y_processed)
        self.visualizer.plot_confusion_matrices(results)

        # Save training results summary
        self._save_training_summary(results)

        return results

    def optimize_top_models(self, top_n: int = 3) -> Dict[str, Any]:
        """
        Optimize hyperparameters for top N models.

        Args:
            top_n: Number of top models to optimize

        Returns:
            Dictionary of optimized models
        """
        logger.info(f"\n{'='*80}\nSTEP 4: Hyperparameter Optimization\n{'='*80}")

        if self.training_results is None:
            raise ValueError("No training results available. Run train_models() first.")

        # Rank models by ROC-AUC
        model_scores = {
            name: result['metrics']['roc_auc_mean']
            for name, result in self.training_results.items()
        }
        sorted_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)

        logger.info(f"\nTop {top_n} models by ROC-AUC:")
        for i, (name, score) in enumerate(sorted_models[:top_n], 1):
            logger.info(f"{i}. {name}: {score:.4f}")

        # Get feature matrix
        feature_cols = [col for col in self.descriptors_data.columns
                       if col not in ['smiles', 'canonical_smiles', 'activity_class',
                                     'activity_label', 'pIC50']]
        X = self.descriptors_data[feature_cols]
        y = self.descriptors_data['activity_class']
        X_processed, y_processed = self.model_trainer.preprocess_features(X, y, fit_scaler=True)

        # Optimize top models
        optimized_models = {}
        for name, _ in sorted_models[:top_n]:
            base_model = self.training_results[name]['model']
            optimized_model, best_params = self.model_trainer.optimize_hyperparameters(
                name, base_model, X_processed, y_processed
            )
            optimized_models[name] = {
                'model': optimized_model,
                'params': best_params
            }

        # Save optimized models
        for name, data in optimized_models.items():
            model_path = os.path.join(self.config.models_dir, f'{name}_optimized.pkl')
            joblib.dump(data['model'], model_path)
            logger.info(f"Saved optimized {name} to {model_path}")

        return optimized_models

    def build_ensemble(self, optimized_models: Dict[str, Any]) -> Any:
        """
        Build stacking ensemble from optimized models.

        Args:
            optimized_models: Dictionary of optimized models

        Returns:
            Trained ensemble model
        """
        logger.info(f"\n{'='*80}\nSTEP 5: Ensemble Model Building\n{'='*80}")

        # Prepare data
        feature_cols = [col for col in self.descriptors_data.columns
                       if col not in ['smiles', 'canonical_smiles', 'activity_class',
                                     'activity_label', 'pIC50']]
        X = self.descriptors_data[feature_cols]
        y = self.descriptors_data['activity_class']
        X_processed, y_processed = self.model_trainer.preprocess_features(X, y, fit_scaler=True)

        # Build ensemble
        top_models = [(name, data['model']) for name, data in optimized_models.items()]
        ensemble = self.model_trainer.build_ensemble_model(top_models, X_processed, y_processed)

        self.ensemble_model = ensemble

        # Evaluate ensemble
        y_pred = ensemble.predict(X_processed)
        y_prob = ensemble.predict_proba(X_processed)[:, 1]
        metrics = self.model_trainer.calculate_metrics(y_processed, y_pred, y_prob)

        logger.info("\nEnsemble Model Performance:")
        for metric, value in metrics.items():
            logger.info(f"  {metric}: {value:.4f}")

        # Save ensemble model
        ensemble_path = os.path.join(self.config.models_dir, 'ensemble_model.pkl')
        joblib.dump(ensemble, ensemble_path)
        logger.info(f"\nSaved ensemble model to {ensemble_path}")

        # Save scaler
        scaler_path = os.path.join(self.config.models_dir, 'feature_scaler.pkl')
        joblib.dump(self.model_trainer.scaler, scaler_path)
        logger.info(f"Saved feature scaler to {scaler_path}")

        # Save feature names
        features_path = os.path.join(self.config.models_dir, 'feature_names.pkl')
        joblib.dump(self.model_trainer.feature_names, features_path)
        logger.info(f"Saved feature names to {features_path}")

        return ensemble

    def generate_interpretability_plots(self, optimized_models: Dict[str, Any]):
        """Generate SHAP and feature importance plots."""
        logger.info(f"\n{'='*80}\nSTEP 6: Model Interpretability Analysis\n{'='*80}")

        # Prepare data
        feature_cols = [col for col in self.descriptors_data.columns
                       if col not in ['smiles', 'canonical_smiles', 'activity_class',
                                     'activity_label', 'pIC50']]
        X = self.descriptors_data[feature_cols]
        y = self.descriptors_data['activity_class']
        X_processed, y_processed = self.model_trainer.preprocess_features(X, y, fit_scaler=True)

        # Generate plots for each optimized model
        for name, data in optimized_models.items():
            model = data['model']

            # Feature importance
            self.visualizer.plot_feature_importance(
                model,
                self.model_trainer.feature_names,
                model_name=name,
                filename=f'feature_importance_{name}.png'
            )

            # SHAP analysis
            self.visualizer.plot_shap_summary(
                model,
                X_processed,
                self.model_trainer.feature_names,
                model_name=name,
                filename=f'shap_summary_{name}.png'
            )

    def perform_virtual_screening(
        self,
        screening_smiles: List[str],
        compound_ids: Optional[List[str]] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Perform virtual screening on new compounds.

        Args:
            screening_smiles: List of SMILES strings to screen
            compound_ids: Optional list of compound IDs

        Returns:
            Tuple of (all_results, hits)
        """
        logger.info(f"\n{'='*80}\nSTEP 7: Virtual Screening\n{'='*80}")

        if self.ensemble_model is None:
            raise ValueError("No ensemble model available. Run build_ensemble() first.")

        # Perform screening
        results, hits = self.virtual_screener.screen_compounds(
            self.ensemble_model,
            self.model_trainer.scaler,
            self.model_trainer.feature_names,
            screening_smiles,
            compound_ids
        )

        # Save results
        results_path = os.path.join(self.config.output_dir, 'screening_results.csv')
        results.to_csv(results_path, index=False)
        logger.info(f"Saved screening results to {results_path}")

        hits_path = os.path.join(self.config.output_dir, 'screening_hits.csv')
        hits.to_csv(hits_path, index=False)
        logger.info(f"Saved screening hits to {hits_path}")

        return results, hits

    def _save_training_summary(self, results: Dict[str, Dict]):
        """Save training summary to file."""
        summary_path = os.path.join(self.config.output_dir, 'training_summary.txt')

        with open(summary_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("MODEL TRAINING SUMMARY\n")
            f.write("="*80 + "\n\n")

            for name, result in results.items():
                f.write(f"\n{name}\n")
                f.write("-" * 40 + "\n")
                f.write(f"Training time: {result['training_time']:.2f}s\n")
                f.write("\nCross-Validation Metrics:\n")

                metrics = result['metrics']
                for metric, value in sorted(metrics.items()):
                    f.write(f"  {metric}: {value:.4f}\n")

                f.write("\nConfusion Matrix:\n")
                f.write(str(result['confusion_matrix']) + "\n")

        logger.info(f"Saved training summary to {summary_path}")

    def save_pipeline(self, filename: str = 'pipeline.pkl'):
        """Save the entire pipeline state."""
        pipeline_path = os.path.join(self.config.output_dir, filename)

        state = {
            'config': self.config,
            'bioactivity_data': self.bioactivity_data,
            'descriptors_data': self.descriptors_data,
            'training_results': self.training_results,
            'ensemble_model': self.ensemble_model,
            'scaler': self.model_trainer.scaler,
            'feature_names': self.model_trainer.feature_names
        }

        joblib.dump(state, pipeline_path)
        logger.info(f"Saved complete pipeline to {pipeline_path}")

    @classmethod
    def load_pipeline(cls, filename: str):
        """Load a saved pipeline state."""
        state = joblib.load(filename)

        pipeline = cls(config=state['config'])
        pipeline.bioactivity_data = state['bioactivity_data']
        pipeline.descriptors_data = state['descriptors_data']
        pipeline.training_results = state['training_results']
        pipeline.ensemble_model = state['ensemble_model']
        pipeline.model_trainer.scaler = state['scaler']
        pipeline.model_trainer.feature_names = state['feature_names']

        logger.info(f"Loaded pipeline from {filename}")
        return pipeline


def main():
    """Main execution function."""
    print("\n" + "="*80)
    print("ML-BASED VIRTUAL DRUG SCREENING PIPELINE")
    print("="*80 + "\n")

    try:
        # Create configuration
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

        # Save configuration
        config.save(os.path.join(config.output_dir, 'config.json'))

        # Initialize pipeline
        pipeline = MLDrugScreeningPipeline(config)

        # Get target name from user
        print("\nEnter the target protein name to search in ChEMBL:")
        print("Examples: 'EGFR', 'Acetylcholinesterase', 'Dopamine D2 receptor'")
        target_name = input("Target name: ").strip()

        if not target_name:
            logger.warning("No target name provided. Using default: 'Acetylcholinesterase'")
            target_name = "Acetylcholinesterase"

        # Step 1: Retrieve target data
        bioactivity_df = pipeline.retrieve_target_data(target_name, auto_select=False)

        # Step 2: Calculate molecular descriptors
        descriptors_df = pipeline.calculate_descriptors()

        # Step 3: Train models
        training_results = pipeline.train_models()

        # Step 4: Optimize top models
        optimized_models = pipeline.optimize_top_models(top_n=config.ensemble_top_n)

        # Step 5: Build ensemble
        ensemble = pipeline.build_ensemble(optimized_models)

        # Step 6: Generate interpretability plots
        pipeline.generate_interpretability_plots(optimized_models)

        # Step 7: Virtual screening (optional)
        print("\n" + "="*80)
        print("VIRTUAL SCREENING")
        print("="*80)
        print("\nDo you want to perform virtual screening on new compounds?")
        screen_choice = input("Enter 'yes' to screen or 'no' to skip: ").strip().lower()

        if screen_choice in ['yes', 'y']:
            print("\nEnter path to CSV file with SMILES (column name: 'smiles'):")
            print("Or enter 'demo' to use demo SMILES")
            file_path = input("File path: ").strip()

            if file_path.lower() == 'demo':
                # Demo SMILES
                demo_smiles = [
                    'CC(C)Cc1ccc(cc1)C(C)C(O)=O',  # Ibuprofen
                    'CC(=O)Oc1ccccc1C(O)=O',  # Aspirin
                    'CN1C=NC2=C1C(=O)N(C(=O)N2C)C',  # Caffeine
                ]
                results, hits = pipeline.perform_virtual_screening(demo_smiles)
            else:
                screening_df = pd.read_csv(file_path)
                results, hits = pipeline.perform_virtual_screening(
                    screening_df['smiles'].tolist(),
                    screening_df.get('compound_id', None)
                )

            print(f"\nScreening complete! Found {len(hits)} hits.")

        # Save complete pipeline
        pipeline.save_pipeline()

        # Final summary
        print("\n" + "="*80)
        print("PIPELINE EXECUTION COMPLETE")
        print("="*80)
        print(f"\nResults saved to: {config.output_dir}/")
        print(f"Models saved to: {config.models_dir}/")
        print(f"Figures saved to: {config.figures_dir}/")
        print("\nPipeline execution successful!")

    except KeyboardInterrupt:
        print("\n\nPipeline execution cancelled by user.")
        logger.info("Pipeline execution cancelled by user")
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
