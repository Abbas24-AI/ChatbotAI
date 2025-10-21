# ChatbotAI Repository

This repository contains two main projects:

## 1. 🤖💬 OpenAI Chatbot

A conversational chatbot built in Python using Streamlit and the OpenAI LLM model GPT 3.5.

### Demo App

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://openai-chatbot.streamlit.app/)

### Prerequisite libraries

```
streamlit
openai
```

### Get an OpenAI API key

You can get your own OpenAI API key by following the following instructions:
1. Go to https://platform.openai.com/account/api-keys.
2. Click on the `+ Create new secret key` button.
3. Next, enter an identifier name (optional) and click on the `Create secret key` button.

### Further Reading

- 🛠️ [Streamlit Documentation Tutorial on _**Build conversational apps**_](https://docs.streamlit.io/knowledge-base/tutorials/build-conversational-apps)
- 📖 [Streamlit Documentation on _**Chat elements**_](https://docs.streamlit.io/library/api-reference/chat)

---

## 2. 🧪🔬 ML-Based Virtual Drug Screening Pipeline

A production-ready Python pipeline for machine learning-based virtual drug screening using ChEMBL bioactivity data.

### Key Features

- **ChEMBL Integration**: Automated retrieval of IC50 bioactivity data
- **Comprehensive Descriptors**: 2000+ molecular features (Morgan fingerprints, MACCS keys, RDKit descriptors)
- **Multiple ML Models**: Random Forest, XGBoost, LightGBM, SVM, Neural Networks
- **10-Fold Cross-Validation**: Robust model evaluation
- **Hyperparameter Optimization**: Automatic tuning for top performers
- **Ensemble Modeling**: Stacking classifier from best models
- **SHAP Analysis**: Model interpretability and feature importance
- **Virtual Screening**: Automated compound library screening
- **Publication-Quality Figures**: 300 DPI plots for journals

### Quick Start

```bash
# Run the complete pipeline
python ml_drug_screening.py

# Or try the examples
python example_usage.py
```

### Documentation

- **Full Documentation**: See [ML_DRUG_SCREENING_README.md](ML_DRUG_SCREENING_README.md)
- **Quick Start Guide**: See [QUICKSTART.md](QUICKSTART.md)
- **Example Scripts**: See [example_usage.py](example_usage.py)

### Typical Workflow

1. Search and select target protein from ChEMBL
2. Retrieve and preprocess IC50 bioactivity data
3. Calculate molecular descriptors and fingerprints
4. Train multiple ML models with 10-fold CV
5. Optimize hyperparameters for top 3 models
6. Build ensemble stacking classifier
7. Generate SHAP interpretability plots
8. Perform virtual screening on new compounds

### Output

The pipeline generates:
- Trained ensemble model (saved as .pkl)
- Publication-quality figures (ROC, PR curves, SHAP plots)
- Comprehensive performance metrics
- Virtual screening results with hits

### Requirements

See [requirements.txt](requirements.txt) for all dependencies. The script auto-installs required packages.

---

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd ChatbotAI

# Install dependencies
pip install -r requirements.txt
```

## License

MIT License

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.
