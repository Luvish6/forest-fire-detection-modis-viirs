import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, roc_curve, auc, 
                             precision_recall_curve, average_precision_score, classification_report)
import os
import sys
import argparse
from pathlib import Path

# Set plotting style for IEEE publications
plt.style.use('seaborn-v0_8-paper')
sns.set_context("paper", font_scale=1.2)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['figure.dpi'] = 300

class FireDetectionPipeline:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {
            'Logistic Regression': LogisticRegression(random_state=random_state, max_iter=1000),
            'Random Forest': RandomForestClassifier(random_state=random_state, n_estimators=100, n_jobs=-1),
            'XGBoost': XGBClassifier(random_state=random_state, eval_metric='logloss', n_jobs=-1)
        }
        self.results = {}
        
    def load_data(self, file_path, dataset_name):
        """
        Load and perform initial check of the dataset.
        """
        print(f"\n[{dataset_name}] Loading data from {file_path}...")
        try:
            df = pd.read_csv(file_path)
            print(f"[{dataset_name}] Loaded {df.shape[0]} rows, {df.shape[1]} columns.")
            return df
        except FileNotFoundError:
            print(f"ERROR: File not found at {file_path}")
            return None

    def preprocess(self, df, dataset_name, fire_threshold=50):
        """
        Preprocess data, separate features/targets, and remove potential leaks.
        CRITICAL: 'confidence' is used ONLY for defining the target and then removed.
        """
        print(f"[{dataset_name}] Preprocessing...")
        
        # 1. Define Candidate Features (Before Drop)
        # Note: We must ensure 'confidence' is NOT in this list for the final X
        if dataset_name == 'MODIS':
            # MODIS specific columns
            feature_cols = ['latitude', 'longitude', 'brightness', 'frp'] 
            # Potentially acq_date/acq_time could be processed, but tabular data often lacks them in float format
            # keeping it simple and strictly numeric/physics based for now.
        else: # VIIRS
            # VIIRS specific columns (I-Band 375m)
            feature_cols = ['latitude', 'longitude', 'bright_ti4', 'bright_ti5', 'frp']
            
            # Helper for label generation - VIIRS confidence is often 'l', 'n', 'h'
            if df['confidence'].dtype == 'O':
                 conf_map = {'l': 75, 'n': 50, 'h': 100} # VIIRS mapping standard
                 df['confidence'] = df['confidence'].map(conf_map)
        
        # 2. Generate Target
        # The project defines 'Fire' as high confidence.
        # Currently, the dataset HAS NO "True Non-Fire" points naturally.
        # It only has "Detected Thermal Anomalies". 
        # So the task is effectively "Classifying High Confidence Fire vs Low Confidence Fire".
        # This is a key limitation to be reported.
        
        if dataset_name == 'VIIRS':
            # VIIRS has confidence 'n'(50), 'l'(75), 'h'(100).
            # Threshold 50 includes ALL data (single class).
            # To create a binary classification, we treat 'Nominal' (n) as class 0
            # and 'Low'/'High' (l, h) as class 1.
            fire_threshold = 75
            
        df = df.dropna()
        y = (df['confidence'] >= fire_threshold).astype(int)
        
        # 3. Create Feature Matrix X
        # STRICTLY drop confidence to prevent target leakage
        X = df[feature_cols].copy()
        
        # 4. Feature Engineering (Optional/Basic)
        # Adding a simple interaction term if applicable? 
        # For strict audit, let's keep it clean to prove over-performance isn't due to magic features.
        
        print(f"[{dataset_name}] Class Distribution:")
        print(y.value_counts(normalize=True))
        
        return X, y

    def evaluate_model(self, X, y, dataset_name):
        """
        Train and evaluate models using multiple strict validation techniques.
        Now compares 80-20 and 70-30 splits.
        """
        # Check for single class
        if len(y.unique()) < 2:
            print(f"CRITICAL WARNING: Dataset {dataset_name} contains only ONE class: {y.unique()}.")
            return

        test_sizes = {
            '80-20': 0.2, 
            '70-30': 0.3
        }

        for split_name, test_size in test_sizes.items():
            print(f"\n[{dataset_name}] Running {split_name} Split (Test Size: {test_size})...")
            
            # 1. Split Data (Strict Holdout)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, stratify=y, random_state=self.random_state
            )
            
            # Scaling (Important for LR)
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            split_key = f"{dataset_name} ({split_name})"
            dataset_results = {}
            
            for name, model in self.models.items():
                # Use scaled data for LR, raw for Trees
                X_tr = X_train_scaled if name == 'Logistic Regression' else X_train
                X_te = X_test_scaled if name == 'Logistic Regression' else X_test
                
                # Cross Validation (Training Set Only)
                cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
                cv_scores = cross_val_score(model, X_tr, y_train, cv=cv, scoring='f1')
                
                # Final Fit & Predict
                model.fit(X_tr, y_train)
                y_pred = model.predict(X_te)
                y_probs = model.predict_proba(X_te)[:, 1]
                
                # Metrics
                acc = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred)
                auc_score = calc_auc(y_test, y_probs)
                
                dataset_results[name] = {
                    'model': model,
                    'y_test': y_test,
                    'y_pred': y_pred,
                    'y_probs': y_probs,
                    'accuracy': acc,
                    'f1': f1,
                    'auc': auc_score,
                    'cv_f1_mean': cv_scores.mean(),
                    'cv_f1_std': cv_scores.std()
                }
            
            self.results[split_key] = dataset_results

    def generate_report(self):
        """
        Generate IEEE-style visualizations and comparison table.
        """
        if not os.path.exists('../results'):
            os.makedirs('../results')
            
        # 1. Generate Comparison Table
        print("\n" + "="*80)
        print("COMPARISON TABLE: 80-20 vs 70-30 SPLITS")
        print("="*80)
        table_data = []
        
        for exp_name, res in self.results.items():
            dataset_name = exp_name.split(' (')[0]
            split_name = exp_name.split(' (')[1].replace(')', '')
            
            for model_name, metrics in res.items():
                table_data.append({
                    'Dataset': dataset_name,
                    'Split': split_name,
                    'Model': model_name,
                    'Test Accuracy': f"{metrics['accuracy']:.4f}",
                    'F1-Score': f"{metrics['f1']:.4f}",
                    'AUC': f"{metrics['auc']:.4f}",
                    'CV Mean F1': f"{metrics['cv_f1_mean']:.4f}"
                })
        
        df_results = pd.DataFrame(table_data)
        print(df_results.to_string(index=False))
        # df_results.to_csv('results/comparison_table_v2.csv', index=False)
        # print("\nTable saved to results/comparison_table_v2.csv")

        # 2. Visualizations (Only for the first available split to avoid clutter, e.g. 70-30)
        # Or generate for all. Let's do it for all but name files appropriately.
        for exp_name, res in self.results.items():
            safe_name = exp_name.replace(' ', '_').replace('(', '').replace(')', '')
            dataset_name = exp_name.split(' (')[0]
            
            # ROC Curves
            plt.figure(figsize=(6, 5))
            for name, metrics in res.items():
                fpr, tpr, _ = roc_curve(metrics['y_test'], metrics['y_probs'])
                plt.plot(fpr, tpr, label=f"{name} (AUC = {metrics['auc']:.2f})")
            plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'{exp_name} - ROC Curves')
            plt.legend(loc='lower right')
            plt.tight_layout()
            plt.savefig(f'../results/roc_{safe_name}.png')
            plt.close()
            
            # 2. Confusion Matrices
            fig, axes = plt.subplots(1, 3, figsize=(15, 4))
            for i, (name, metrics) in enumerate(res.items()):
                cm = confusion_matrix(metrics['y_test'], metrics['y_pred'])
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i])
                axes[i].set_title(f'{name}\nAcc: {metrics["accuracy"]:.1%}')
                axes[i].set_ylabel('True Label')
                axes[i].set_xlabel('Predicted Label')
            plt.suptitle(f'{exp_name} - Confusion Matrices', y=1.05)
            plt.tight_layout()
            plt.savefig(f'../results/cm_{safe_name}.png')
            plt.close()

            # 3. Feature Importance (RF/XGB)
            # Just do one combined plot for brevity
            tree_models = [('Random Forest', res['Random Forest']['model']), 
                           ('XGBoost', res['XGBoost']['model'])]
            
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            for i, (name, model) in enumerate(tree_models):
                if hasattr(model, 'feature_importances_'):
                    imps = model.feature_importances_
                    # Recover feature names - simple way since we have the instance
                    if hasattr(model, 'feature_names_in_'):
                         feats = model.feature_names_in_
                    else:
                        # Fallback for when feature names aren't saved (e.g. numpy input)
                        # We know the cols based on dataset
                        if dataset_name == 'MODIS':
                            feats = ['lat', 'long', 'bright', 'frp']
                        else:
                            feats = ['lat', 'long', 'temp4', 'temp5', 'frp']
                            
                    sns.barplot(x=imps, y=feats, ax=axes[i], orient='h')
                    axes[i].set_title(f'{name} Feature Importance')
            plt.suptitle(f'{exp_name} - Feature Drivers')
            plt.tight_layout()
            plt.savefig(f'../results/feat_imp_{safe_name}.png')
            plt.close()

def calc_auc(y_test, y_probs):
    try:
        return auc(*roc_curve(y_test, y_probs)[:2])
    except:
        return 0.5

def main():
    parser = argparse.ArgumentParser(description="Forest Fire Detection Pipeline")
    parser.add_argument('--modis', type=str, default='../data/modis_2024.csv', help='Path to MODIS CSV file')
    parser.add_argument('--viirs', type=str, default='../data/viirs_2024.csv', help='Path to VIIRS CSV file')
    args = parser.parse_args()

    pipeline = FireDetectionPipeline()
    
    # Configure Paths
    modis_path = args.modis
    viirs_path = args.viirs
    
    # Check if files exist before trying to load
    if not os.path.exists(modis_path):
        print(f"NOTE: MODIS file not found at {modis_path}. Skipping MODIS pipeline.")
        print(f"To run MODIS, place 'modis_2024.csv' in 'data/' or provide path via --modis")
    else:
        # 1. MODIS Pipeline
        df_modis = pipeline.load_data(modis_path, "MODIS")
        if df_modis is not None:
            X_mod, y_mod = pipeline.preprocess(df_modis, "MODIS")
            pipeline.evaluate_model(X_mod, y_mod, "MODIS")
        
    if not os.path.exists(viirs_path):
        print(f"NOTE: VIIRS file not found at {viirs_path}. Skipping VIIRS pipeline.")
        print(f"To run VIIRS, place 'viirs_2024.csv' in 'data/' or provide path via --viirs")
    else:
        # 2. VIIRS Pipeline
        df_viirs = pipeline.load_data(viirs_path, "VIIRS")
        if df_viirs is not None:
            X_vir, y_vir = pipeline.preprocess(df_viirs, "VIIRS")
            pipeline.evaluate_model(X_vir, y_vir, "VIIRS")
        
    # 3. Generate Artifacts
    if pipeline.results:
        pipeline.generate_report()
        
        import pickle

        # Safely create models folder relative to script location
        model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../models')
        os.makedirs(model_dir, exist_ok=True)

        # Find best model using AUC
        best_model = None
        best_score = 0

        for exp in pipeline.results.values():
            for model_name, metrics in exp.items():
                if metrics['auc'] > best_score:
                    best_score = metrics['auc']
                    best_model = metrics['model']

        # Save model
        if best_model is not None:
            model_path = os.path.join(model_dir, 'fire_model.pkl')
            with open(model_path, "wb") as f:
                pickle.dump(best_model, f)
            print(f"\n🔥 Model saved successfully! AUC: {best_score:.4f}")
        else:
            print("\n❌ No model found to save")
            
        print("\nSUCCESS: Pipeline completed. Results saved in 'results/' directory.")
        print("Please check the 'results' folder for ROC curves and Confusion Matrices.")
    else:
        print("\nWARNING: No results generated. Please check data availability.")

if __name__ == "__main__":
    main()
