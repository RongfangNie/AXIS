# AXIS
An Attribution-based Cross-interaction Model for Interpretable Drug Synergy Prediction and Mechanistic Insight

## 📋About

Here we develop AXIS, an interpretable deep learning framework that integrates chemical language models, self‑supervised omics encoders, and a multi‑source cross‑attention module to predict synergy and uncover underlying biological mechanisms.

![](./AXIS.png)

## 🚀 Installation

Create the environment:
```bash

conda env create -f environment.yml
conda activate AXIS
pip install -r requirements.txt
```
## Usage

The notebook `notebooks/Drug_substructure_contribution_maps.ipynb` generates SHAP contribution heatmaps for drug substructures, using the docetaxel‑tamoxifen‑NCIH838 combination as an example.
