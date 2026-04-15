# Forest Fire Detection System

**Author**: [Your Name/Organization]  
**Status**: Active Research / IEEE Compliant

## Overview

This project implements a robust machine learning pipeline for Forest Fire Detection using satellite data from **MODIS** and **VIIRS-SNPP** sensors. The system is designed to classify fire events based on active fire tabular data (CSV), leveraging advanced classifiers like Random Forest and XGBoost.

Key features include:
*   **Dual-Satellite Support**: Process and compare performance between MODIS (1km resolution) and VIIRS (375m resolution).
*   **Comprehensive Audit System**: specialized module (`ml_audit_system.py`) to rigorous check for data leakage, overfitting, and threshold sensitivity.
*   **IEEE-Style Reporting**: Generates publication-ready ROC curves, confusion matrices, and feature importance plots.
*   **Tabular Data Focus**: This project relies exclusively on physics-based thermal anomalies (brightness temperature, FRP), **NOT** computer vision or imagery.

## Directory Structure

```
forest-fire-detection/
 ┣ data/                 # Dataset storage (CSV files not included, see data/README.md)
 ┣ notebooks/            # Jupyter notebooks for interactive analysis
 ┣ results/              # Generated plots, tables, and audit reports
 ┃ ┣ audit_outputs/      # Deep-dive audit results
 ┃ ┗ outputs/            # General EDA outputs
 ┣ src/                  # Source code
 ┃ ┣ eda.py              # Exploratory Data Analysis script
 ┃ ┣ improved_pipeline.py# Main training and evaluation pipeline
 ┃ ┗ ml_audit_system.py  # rigorous ML audit and stress-testing
 ┣ requirements.txt      # Python dependencies
 ┗ README.md             # Project documentation
```

## Data

The project uses standard NASA FIRMS active fire data (CSV format).
**Note**: The raw CSV files are **not** included in this repository.
Please refer to [data/README.md](data/README.md) for instructions on how to download the required MODIS and VIIRS datasets.

## Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/yourusername/forest-fire-detection.git
    cd forest-fire-detection
    ```

2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### 1. Data Setup
Place your downloaded CSV files in the `data/` directory.
*   `data/modis_2024.csv`
*   `data/viirs_2024.csv`

### 2. Exploratory Data Analysis (EDA)
Generate histograms, correlation heatmaps, and distribution plots.
```bash
python src/eda.py --data data/modis_2024.csv
```

### 3. Run Main Pipeline
Train models, evaluate performance, and generate IEEE-style results.
```bash
# Run with default paths (../data/modis_2024.csv)
python src/improved_pipeline.py

# Or specify custom paths
python src/improved_pipeline.py --modis data/my_modis.csv --viirs data/my_viirs.csv
```
Results will be saved in `results/`.

### 4. Run ML Audit System
Execute stress tests, check for leakage, and verify model stability.
```bash
python src/ml_audit_system.py --modis data/modis_2024.csv --viirs data/viirs_2024.csv
```
Audit reports will be saved in `results/audit_outputs/`.

## Models Used

*   **Logistic Regression**: Baseline linear model.
*   **Random Forest**: Ensemble bagging method (High performance).
*   **XGBoost**: Gradient boosting method (Best performance).

## Reproducibility & Limitations

*   **Data Leakage Prevention**: The `confidence` score is strictly removed from the feature set during training to prevent target leakage, as it is a direct proxy for the label.
*   **Class Imbalance**: VIIRS data is highly imbalanced. The pipeline uses stratified sampling and appropriate metrics (F1-score, AUC) to handle this.
*   **No Imagery**: This model is based on tabular physics parameters (Brightness Temperature, FRP), not visual features.

## License

[MIT License](LICENSE)
