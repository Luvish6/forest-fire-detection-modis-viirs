import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, confusion_matrix, classification_report,
                           roc_curve, auc, roc_auc_score)
import os
import argparse
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style for professional academic plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Create audit output directory
def create_audit_directory():
    """Create the audit_outputs directory"""
    if not os.path.exists('../results/audit_outputs'):
        os.makedirs('../results/audit_outputs')
        print("Created directory: ../results/audit_outputs")
    else:
        print("Directory '../results/audit_outputs' already exists")

# Load datasets
def load_data(modis_path, viirs_path):
    print("="*80)
    print("ML AUDIT: LOADING DATASETS")
    print("="*80)

    modis_df = None
    viirs_df = None

    if os.path.exists(modis_path):
        modis_df = pd.read_csv(modis_path)
        print(f"✓ MODIS dataset loaded: {modis_df.shape}")
    else:
        print(f"✗ MODIS file not found at {modis_path}")

    if os.path.exists(viirs_path):
        viirs_df = pd.read_csv(viirs_path)
        print(f"✓ VIIRS dataset loaded: {viirs_df.shape}")
    else:
        print(f"✗ VIIRS file not found at {viirs_path}")
        
    return modis_df, viirs_df

    if modis_df is not None:
        print(f"\nMODIS columns: {list(modis_df.columns)}")
    if viirs_df is not None:
        print(f"VIIRS columns: {list(viirs_df.columns)}")

# Data preprocessing with audit focus
    print("\n" + "="*80)
    print("DATA PREPROCESSING FOR AUDIT")
    print("="*80)

def preprocess_for_audit(df, dataset_name):
    """Preprocess datasets for comprehensive audit"""
    if dataset_name == 'MODIS':
        # Select all available features for audit
        features = ['latitude', 'longitude', 'brightness', 'confidence', 'frp']
        df_processed = df[features].copy()
        
        # Handle missing values
        df_processed = df_processed.dropna()
        print(f"MODIS after cleaning: {df_processed.shape}")
        
        # Create labels with different thresholds
        df_processed['fire_50'] = (df_processed['confidence'] >= 50).astype(int)
        df_processed['fire_60'] = (df_processed['confidence'] >= 60).astype(int)
        df_processed['fire_70'] = (df_processed['confidence'] >= 70).astype(int)
        
        print(f"MODIS fire rates - 50%: {df_processed['fire_50'].mean():.3f}, "
              f"60%: {df_processed['fire_60'].mean():.3f}, "
              f"70%: {df_processed['fire_70'].mean():.3f}")
        
        return df_processed
    
    elif dataset_name == 'VIIRS':
        # Handle VIIRS confidence mapping
        confidence_mapping = {'h': 100, 'l': 75, 'n': 50}
        df['confidence'] = df['confidence'].map(confidence_mapping)
        
        # Select features
        features = ['latitude', 'longitude', 'bright_ti4', 'bright_ti5', 'confidence', 'frp']
        df_processed = df[features].copy()
        
        # Handle missing values
        df_processed = df_processed.dropna()
        print(f"VIIRS after cleaning: {df_processed.shape}")
        
        # Create labels with different thresholds
        df_processed['fire_50'] = (df_processed['confidence'] >= 50).astype(int)
        df_processed['fire_60'] = (df_processed['confidence'] >= 60).astype(int)
        df_processed['fire_70'] = (df_processed['confidence'] >= 70).astype(int)
        
        print(f"VIIRS fire rates - 50%: {df_processed['fire_50'].mean():.3f}, "
              f"60%: {df_processed['fire_60'].mean():.3f}, "
              f"70%: {df_processed['fire_70'].mean():.3f}")
        
        return df_processed

# Preprocess datasets


# Comprehensive audit function
def conduct_comprehensive_audit(df, dataset_name, threshold='50'):
    """Conduct comprehensive ML audit"""
    print(f"\n{'='*60}")
    print(f"AUDITING {dataset_name} - THRESHOLD {threshold}%")
    print(f"{'='*60}")
    
    label_col = f'fire_{threshold}'
    y = df[label_col]
    
    # Define feature sets for leakage testing
    if dataset_name == 'MODIS':
        feature_sets = {
            'all_features': ['latitude', 'longitude', 'brightness', 'confidence', 'frp'],
            'without_confidence': ['latitude', 'longitude', 'brightness', 'frp'],
            'brightness_frp_only': ['brightness', 'frp'],
            'spatial_only': ['latitude', 'longitude']
        }
    else:  # VIIRS
        feature_sets = {
            'all_features': ['latitude', 'longitude', 'bright_ti4', 'bright_ti5', 'confidence', 'frp'],
            'without_confidence': ['latitude', 'longitude', 'bright_ti4', 'bright_ti5', 'frp'],
            'brightness_frp_only': ['bright_ti4', 'bright_ti5', 'frp'],
            'spatial_only': ['latitude', 'longitude']
        }
    
    audit_results = {}
    
    for feature_set_name, features in feature_sets.items():
        print(f"\n--- Testing feature set: {feature_set_name} ---")
        X = df[features].copy()
        
        # Handle any remaining missing values
        X = X.fillna(X.mean())
        
        audit_results[feature_set_name] = evaluate_feature_set(X, y, dataset_name, feature_set_name, threshold)
    
    return audit_results

def evaluate_feature_set(X, y, dataset_name, feature_set_name, threshold):
    """Evaluate a specific feature set with comprehensive tests"""
    results = {}
    
    # Test different train-test splits
    splits = [0.2, 0.3, 0.4]
    split_results = {}
    
    for test_size in splits:
        split_name = f"{int((1-test_size)*100)}{int(test_size*100)}"
        split_results[split_name] = test_train_test_split(X, y, test_size, dataset_name, feature_set_name, split_name, threshold)
    
    results['splits'] = split_results
    
    # Test cross-validation
    results['cv'] = test_cross_validation(X, y, dataset_name, feature_set_name, threshold)
    
    # Test with shuffled labels (sanity check)
    results['shuffled'] = test_shuffled_labels(X, y, dataset_name, feature_set_name, threshold)
    
    # Test constrained models
    results['constrained'] = test_constrained_models(X, y, dataset_name, feature_set_name, threshold)
    
    return results

def test_train_test_split(X, y, test_size, dataset_name, feature_set_name, split_name, threshold):
    """Test train-test split performance"""
    print(f"  Testing {split_name} split...")
    
    # Check if we have both classes
    if len(np.unique(y)) < 2:
        print(f"    ⚠️  Only one class present. Skipping {dataset_name} {feature_set_name} {split_name}")
        return None
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    # Scale features for Logistic Regression
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Initialize models
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
        'XGBoost': XGBClassifier(random_state=42, eval_metric='logloss')
    }
    
    split_results = {}
    
    for name, model in models.items():
        try:
            if name == 'Logistic Regression':
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            
            try:
                auc_score = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted')
            except:
                auc_score = 0.5  # Default if ROC calculation fails
            
            split_results[name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'auc_score': auc_score,
                'y_test': y_test,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba,
                'confusion_matrix': confusion_matrix(y_test, y_pred),
                'model': model
            }
            
            print(f"    {name}: Acc={accuracy:.4f}, F1={f1:.4f}, AUC={auc_score:.4f}")
            
        except Exception as e:
            print(f"    ⚠️  {name} failed: {str(e)}")
            split_results[name] = None
    
    # Generate confusion matrices only if we have valid results
    valid_results = {k: v for k, v in split_results.items() if v is not None}
    if valid_results:
        generate_confusion_matrices(valid_results, dataset_name, feature_set_name, f"{split_name}_threshold{threshold}")
    
    return split_results

def test_cross_validation(X, y, dataset_name, feature_set_name, threshold):
    """Test 5-fold stratified cross-validation"""
    print(f"  Testing 5-fold CV...")
    
    # Check if we have both classes
    if len(np.unique(y)) < 2:
        print(f"    ⚠️  Only one class present. Skipping CV for {dataset_name} {feature_set_name}")
        return {}
    
    # Scale features for Logistic Regression
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Initialize models
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
        'XGBoost': XGBClassifier(random_state=42, eval_metric='logloss')
    }
    
    cv_results = {}
    
    for name, model in models.items():
        try:
            if name == 'Logistic Regression':
                scores = cross_val_score(model, X_scaled, y, cv=StratifiedKFold(5, shuffle=True, random_state=42), 
                                       scoring='accuracy')
            else:
                scores = cross_val_score(model, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=42), 
                                       scoring='accuracy')
            
            cv_results[name] = {
                'mean_accuracy': scores.mean(),
                'std_accuracy': scores.std(),
                'scores': scores
            }
            
            print(f"    {name}: {scores.mean():.4f} ± {scores.std():.4f}")
            
        except Exception as e:
            print(f"    ⚠️  {name} CV failed: {str(e)}")
            cv_results[name] = {'mean_accuracy': 0, 'std_accuracy': 0, 'scores': []}
    
    return cv_results

def test_shuffled_labels(X, y, dataset_name, feature_set_name, threshold):
    """Test with shuffled labels (should be ~50%)"""
    print(f"  Testing shuffled labels...")
    
    # Check if we have both classes
    if len(np.unique(y)) < 2:
        print(f"    ⚠️  Only one class present. Skipping shuffled test for {dataset_name} {feature_set_name}")
        return {}
    
    # Shuffle labels
    y_shuffled = np.random.permutation(y)
    
    # Use 80-20 split for quick test
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_shuffled, test_size=0.2, random_state=42
        )
    except:
        print(f"    ⚠️  Cannot split shuffled data for {dataset_name} {feature_set_name}")
        return {}
    
    # Scale features for Logistic Regression
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Initialize models
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
        'XGBoost': XGBClassifier(random_state=42, eval_metric='logloss')
    }
    
    shuffled_results = {}
    
    for name, model in models.items():
        try:
            if name == 'Logistic Regression':
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
            
            accuracy = accuracy_score(y_test, y_pred)
            shuffled_results[name] = {'accuracy': accuracy}
            
            print(f"    {name}: {accuracy:.4f}")
            
        except Exception as e:
            print(f"    ⚠️  {name} shuffled test failed: {str(e)}")
            shuffled_results[name] = {'accuracy': 0}
    
    return shuffled_results

def test_constrained_models(X, y, dataset_name, feature_set_name, threshold):
    """Test with constrained model complexity"""
    print(f"  Testing constrained models...")
    
    # Check if we have both classes
    if len(np.unique(y)) < 2:
        print(f"    ⚠️  Only one class present. Skipping constrained models for {dataset_name} {feature_set_name}")
        return {}
    
    # Use 80-20 split
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
    except:
        print(f"    ⚠️  Cannot split data for constrained models {dataset_name} {feature_set_name}")
        return {}
    
    # Scale features for Logistic Regression
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Constrained models
    constrained_models = {
        'Logistic Regression (C=0.1)': LogisticRegression(random_state=42, max_iter=1000, C=0.1),
        'Random Forest (max_depth=3)': RandomForestClassifier(random_state=42, n_estimators=50, max_depth=3),
        'XGBoost (max_depth=3, reg_lambda=1)': XGBClassifier(random_state=42, eval_metric='logloss', 
                                                             max_depth=3, reg_lambda=1, n_estimators=50)
    }
    
    constrained_results = {}
    
    for name, model in constrained_models.items():
        try:
            if 'Logistic' in name:
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
            
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            
            constrained_results[name] = {'accuracy': accuracy, 'f1_score': f1}
            
            print(f"    {name}: Acc={accuracy:.4f}, F1={f1:.4f}")
            
        except Exception as e:
            print(f"    ⚠️  {name} constrained test failed: {str(e)}")
            constrained_results[name] = {'accuracy': 0, 'f1_score': 0}
    
    return constrained_results

def generate_confusion_matrices(results, dataset_name, feature_set_name, split_name):
    """Generate and save confusion matrices"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    models = ['Logistic Regression', 'Random Forest', 'XGBoost']
    
    for i, model_name in enumerate(models):
        ax = axes[i]
        cm = results[model_name]['confusion_matrix']
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', square=True, ax=ax,
                   xticklabels=['No Fire', 'Fire'], yticklabels=['No Fire', 'Fire'])
        
        ax.set_title(f'{model_name}\nAcc: {results[model_name]["accuracy"]:.4f}', 
                    fontsize=12, fontweight='bold')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
    
    plt.suptitle(f'Confusion Matrices - {dataset_name} {feature_set_name} {split_name}', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    filename = f"../results/audit_outputs/confusion_{dataset_name.lower()}_{feature_set_name}_{split_name}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    ✓ Saved: {filename}")

def generate_learning_curves(X, y, dataset_name, feature_set_name, threshold):
    """Generate learning curves for overfitting analysis"""
    print(f"  Generating learning curves...")
    
    # Check if we have both classes
    if len(np.unique(y)) < 2:
        print(f"    ⚠️  Only one class present. Skipping learning curves for {dataset_name} {feature_set_name}")
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    models = [
        ('Logistic Regression', LogisticRegression(random_state=42, max_iter=1000)),
        ('Random Forest', RandomForestClassifier(random_state=42, n_estimators=100)),
        ('XGBoost', XGBClassifier(random_state=42, eval_metric='logloss'))
    ]
    
    for i, (name, model) in enumerate(models):
        ax = axes[i]
        
        try:
            if name == 'Logistic Regression':
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                train_sizes, train_scores, val_scores = learning_curve(
                    model, X_scaled, y, cv=5, train_sizes=np.linspace(0.1, 1.0, 10), 
                    random_state=42, scoring='accuracy'
                )
            else:
                train_sizes, train_scores, val_scores = learning_curve(
                    model, X, y, cv=5, train_sizes=np.linspace(0.1, 1.0, 10), 
                    random_state=42, scoring='accuracy'
                )
            
            train_mean = np.mean(train_scores, axis=1)
            train_std = np.std(train_scores, axis=1)
            val_mean = np.mean(val_scores, axis=1)
            val_std = np.std(val_scores, axis=1)
            
            ax.plot(train_sizes, train_mean, 'o-', color='blue', label='Training accuracy')
            ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
            
            ax.plot(train_sizes, val_mean, 'o-', color='red', label='Validation accuracy')
            ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color='red')
            
            ax.set_title(f'{name}', fontsize=12, fontweight='bold')
            ax.set_xlabel('Training Set Size')
            ax.set_ylabel('Accuracy')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
        except Exception as e:
            print(f"    ⚠️  {name} learning curves failed: {str(e)}")
            ax.text(0.5, 0.5, f'{name}\nLearning curves failed', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
    
    plt.suptitle(f'Learning Curves - {dataset_name} {feature_set_name} (Threshold {threshold}%)', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    filename = f"../results/audit_outputs/learning_curves_{dataset_name.lower()}_{feature_set_name}_threshold{threshold}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    ✓ Saved: {filename}")

def generate_feature_importance_analysis(X, y, dataset_name, feature_set_name, threshold):
    """Generate feature importance analysis"""
    print(f"  Generating feature importance analysis...")
    
    # Check if we have both classes
    if len(np.unique(y)) < 2:
        print(f"    ⚠️  Only one class present. Skipping feature importance for {dataset_name} {feature_set_name}")
        return
    
    # Use 80-20 split
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
    except:
        print(f"    ⚠️  Cannot split data for feature importance {dataset_name} {feature_set_name}")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Random Forest
    try:
        rf = RandomForestClassifier(random_state=42, n_estimators=100)
        rf.fit(X_train, y_train)
        
        importances = rf.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        axes[0].bar(range(len(importances)), importances[indices], color='skyblue', alpha=0.8)
        axes[0].set_title(f'Random Forest Feature Importance\nAcc: {rf.score(X_test, y_test):.4f}', 
                         fontsize=12, fontweight='bold')
        axes[0].set_xticks(range(len(importances)))
        axes[0].set_xticklabels([X.columns[i] for i in indices], rotation=45, ha='right')
        axes[0].set_ylabel('Importance')
        axes[0].grid(True, alpha=0.3)
    except Exception as e:
        print(f"    ⚠️  Random Forest feature importance failed: {str(e)}")
        axes[0].text(0.5, 0.5, 'Random Forest\nFeature importance failed', 
                   ha='center', va='center', transform=axes[0].transAxes, fontsize=12)
    
    # XGBoost
    try:
        xgb = XGBClassifier(random_state=42, eval_metric='logloss')
        xgb.fit(X_train, y_train)
        
        importances = xgb.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        axes[1].bar(range(len(importances)), importances[indices], color='lightcoral', alpha=0.8)
        axes[1].set_title(f'XGBoost Feature Importance\nAcc: {xgb.score(X_test, y_test):.4f}', 
                         fontsize=12, fontweight='bold')
        axes[1].set_xticks(range(len(importances)))
        axes[1].set_xticklabels([X.columns[i] for i in indices], rotation=45, ha='right')
        axes[1].set_ylabel('Importance')
        axes[1].grid(True, alpha=0.3)
    except Exception as e:
        print(f"    ⚠️  XGBoost feature importance failed: {str(e)}")
        axes[1].text(0.5, 0.5, 'XGBoost\nFeature importance failed', 
                   ha='center', va='center', transform=axes[1].transAxes, fontsize=12)
    
    plt.suptitle(f'Feature Importance Analysis - {dataset_name} {feature_set_name} (Threshold {threshold}%)', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    filename = f"../results/audit_outputs/feature_importance_{dataset_name.lower()}_{feature_set_name}_threshold{threshold}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    ✓ Saved: {filename}")

# Conduct comprehensive audits


# Audit VIIRS dataset


# Generate comprehensive audit report


def generate_audit_summary_table(modis_audit_results, viirs_audit_results):
    """Generate comprehensive audit summary table"""
    plt.figure(figsize=(20, 12))
    ax = plt.gca()
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare summary data
    summary_data = []
    
    if modis_audit_results:
        # MODIS results
        for threshold in ['50', '60', '70']:
            for feature_set in ['all_features', 'without_confidence', 'brightness_frp_only', 'spatial_only']:
                results = modis_audit_results[threshold][feature_set]['splits']['8020']
                
                for model_name in ['Logistic Regression', 'Random Forest', 'XGBoost']:
                    row = [
                        f"MODIS {threshold}%",
                        feature_set.replace('_', ' ').title(),
                        model_name,
                        f"{results[model_name]['accuracy']:.4f}",
                        f"{results[model_name]['f1_score']:.4f}",
                        f"{modis_audit_results[threshold][feature_set]['cv'][model_name]['mean_accuracy']:.4f} ± {modis_audit_results[threshold][feature_set]['cv'][model_name]['std_accuracy']:.4f}",
                        f"{modis_audit_results[threshold][feature_set]['shuffled'][model_name]['accuracy']:.4f}"
                    ]
                    summary_data.append(row)
    
    if viirs_audit_results:
        # VIIRS results
        for threshold in ['50', '60', '70']:
            for feature_set in ['all_features', 'without_confidence', 'brightness_frp_only', 'spatial_only']:
                results = viirs_audit_results[threshold][feature_set]['splits']['8020']
                
                for model_name in ['Logistic Regression', 'Random Forest', 'XGBoost']:
                    row = [
                        f"VIIRS {threshold}%",
                        feature_set.replace('_', ' ').title(),
                        model_name,
                        f"{results[model_name]['accuracy']:.4f}",
                        f"{results[model_name]['f1_score']:.4f}",
                        f"{viirs_audit_results[threshold][feature_set]['cv'][model_name]['mean_accuracy']:.4f} ± {viirs_audit_results[threshold][feature_set]['cv'][model_name]['std_accuracy']:.4f}",
                        f"{viirs_audit_results[threshold][feature_set]['shuffled'][model_name]['accuracy']:.4f}"
                    ]
                    summary_data.append(row)
    
    # Create table
    columns = ['Dataset', 'Feature Set', 'Model', 'Test Accuracy', 'F1-Score', 'CV Accuracy (±std)', 'Shuffled Accuracy']
    table = ax.table(cellText=summary_data,
                    colLabels=columns,
                    cellLoc='center',
                    loc='center',
                    bbox=[0, 0, 1, 1])
    
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.5)
    
    # Style the table
    for j in range(len(columns)):
        table[(0, j)].set_facecolor('#2196F3')
        table[(0, j)].set_text_props(weight='bold', color='white')
    
    # Color code rows based on dataset
    colors = ['#E3F2FD', '#BBDEFB', '#90CAF9', '#64B5F6']
    for i in range(1, len(summary_data)):
        row_color = colors[(i-1) % 4]
        for j in range(len(columns)):
            table[(i, j)].set_facecolor(row_color)
    
    plt.title('Comprehensive ML Audit Summary', fontsize=16, fontweight='bold', pad=20)
    filename = "../results/audit_outputs/audit_summary_table.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ Saved: {filename}")



# Print audit conclusions
print("\n" + "="*80)
print("AUDIT CONCLUSIONS")
print("="*80)

def analyze_audit_results(modis_audit_results, viirs_audit_results):
    """Analyze audit results and provide conclusions"""
    
    print("\n🔍 FEATURE LEAKAGE ANALYSIS:")
    print("-" * 50)
    
    # Check confidence feature impact
    if modis_audit_results:
        modis_all_acc = modis_audit_results['50']['all_features']['splits']['8020']['Random Forest']['accuracy']
        modis_no_conf_acc = modis_audit_results['50']['without_confidence']['splits']['8020']['Random Forest']['accuracy']
        
        print(f"MODIS Random Forest Accuracy:")
        print(f"  With confidence: {modis_all_acc:.4f}")
        print(f"  Without confidence: {modis_no_conf_acc:.4f}")
        print(f"  Performance drop: {(modis_all_acc - modis_no_conf_acc):.4f}")
    else:
        modis_all_acc = 0
        modis_no_conf_acc = 0

    if viirs_audit_results:
        viirs_all_acc = viirs_audit_results['50']['all_features']['splits']['8020']['Random Forest']['accuracy']
        viirs_no_conf_acc = viirs_audit_results['50']['without_confidence']['splits']['8020']['Random Forest']['accuracy']
        
        print(f"\nVIIRS Random Forest Accuracy:")
        print(f"  With confidence: {viirs_all_acc:.4f}")
        print(f"  Without confidence: {viirs_no_conf_acc:.4f}")
        print(f"  Performance drop: {(viirs_all_acc - viirs_no_conf_acc):.4f}")
    else:
        viirs_all_acc = 0
        viirs_no_conf_acc = 0
    
    print("\n🎯 THRESHOLD SENSITIVITY ANALYSIS:")
    print("-" * 50)
    
    for dataset_name, audit_results in [('MODIS', modis_audit_results), ('VIIRS', viirs_audit_results)]:
        if audit_results:
            print(f"\n{dataset_name} Dataset:")
            for threshold in ['50', '60', '70']:
                acc = audit_results[threshold]['all_features']['splits']['8020']['Random Forest']['accuracy']
                print(f"  {threshold}% threshold: {acc:.4f}")
    
    print("\n🔄 CROSS-VALIDATION STABILITY:")
    print("-" * 50)
    
    for dataset_name, audit_results in [('MODIS', modis_audit_results), ('VIIRS', viirs_audit_results)]:
        if audit_results:
            print(f"\n{dataset_name} Dataset (50% threshold):")
            cv_mean = audit_results['50']['all_features']['cv']['Random Forest']['mean_accuracy']
            cv_std = audit_results['50']['all_features']['cv']['Random Forest']['std_accuracy']
            print(f"  CV Mean: {cv_mean:.4f} ± {cv_std:.4f}")
            print(f"  CV Range: [{cv_mean - cv_std:.4f}, {cv_mean + cv_std:.4f}]")
    
    print("\n🎲 SHUFFLED LABELS SANITY CHECK:")
    print("-" * 50)
    
    for dataset_name, audit_results in [('MODIS', modis_audit_results), ('VIIRS', viirs_audit_results)]:
        if audit_results:
            print(f"\n{dataset_name} Dataset:")
            shuffled_acc = audit_results['50']['all_features']['shuffled']['Random Forest']['accuracy']
            print(f"  Shuffled accuracy: {shuffled_acc:.4f}")
            print(f"  Expected: ~0.5000, Actual: {shuffled_acc:.4f}")
    
    print("\n⚖️ CONSTRAINED MODEL PERFORMANCE:")
    print("-" * 50)
    
    for dataset_name, audit_results in [('MODIS', modis_audit_results), ('VIIRS', viirs_audit_results)]:
        if audit_results:
            print(f"\n{dataset_name} Dataset:")
            constrained = audit_results['50']['all_features']['constrained']
            for model_name, metrics in constrained.items():
                print(f"  {model_name}: Acc={metrics['accuracy']:.4f}, F1={metrics['f1_score']:.4f}")
    
    print("\n📊 FINAL VERDICT:")
    print("-" * 50)
    
    # Determine if 100% accuracy is justified
    modis_perfect = False
    if modis_audit_results:
        modis_perfect = modis_audit_results['50']['all_features']['splits']['8020']['Random Forest']['accuracy'] == 1.0
    
    viirs_perfect = False
    if viirs_audit_results:
        viirs_perfect = viirs_audit_results['50']['all_features']['splits']['8020']['Random Forest']['accuracy'] == 1.0
    
    print(f"Perfect accuracy detected:")
    print(f"  MODIS: {'YES' if modis_perfect else 'NO'}")
    print(f"  VIIRS: {'YES' if viirs_perfect else 'NO'}")
    
    if modis_perfect or viirs_perfect:
        print(f"\n⚠️  WARNING: Perfect accuracy detected!")
        print(f"This suggests potential data leakage or overly simplistic labeling.")
        
        # Check if confidence is the culprit
        modis_conf_drop = modis_all_acc - modis_no_conf_acc
        viirs_conf_drop = viirs_all_acc - viirs_no_conf_acc
        
        if modis_conf_drop > 0.1 or viirs_conf_drop > 0.1:
            print(f"🔴 CONFIDENCE FEATURE IS LEAKING LABEL INFORMATION!")
            print(f"Removing confidence reduces performance significantly.")
        else:
            print(f"🟡 Performance remains high without confidence feature.")
            
        # Check CV stability (if available)
        cv_std = 0
        if modis_audit_results:
            cv_std = modis_audit_results['50']['all_features']['cv']['Random Forest']['std_accuracy']
        
        if cv_std < 0.01:
            print(f"🔴 EXTREMELY LOW CV VARIANCE SUGGESTS OVERFITTING!")
        else:
            print(f"🟡 CV variance is within acceptable range.")
    else:
        print(f"🟢 No perfect accuracy detected - models appear to be learning genuine patterns.")
    
    print(f"\n📋 RECOMMENDATIONS:")
    print(f"1. Remove confidence feature from training to avoid label leakage")
    print(f"2. Use more realistic fire detection thresholds (≥60%)")
    print(f"3. Implement proper feature engineering with domain knowledge")
    print(f"4. Consider ensemble methods with regularization")
    print(f"5. Validate on temporally separated test sets")

def main():
    parser = argparse.ArgumentParser(description="Forest Fire Detection ML Audit System")
    parser.add_argument('--modis', type=str, default='../data/modis_2024.csv', help='Path to MODIS CSV file')
    parser.add_argument('--viirs', type=str, default='../data/viirs_2024.csv', help='Path to VIIRS CSV file')
    args = parser.parse_args()

    create_audit_directory()
    modis_df, viirs_df = load_data(args.modis, args.viirs)
    
    modis_audit_results = {}
    if modis_df is not None:
        modis_audit = preprocess_for_audit(modis_df, 'MODIS')
        # Audit MODIS dataset
        print("\n" + "="*80)
        print("CONDUCTING COMPREHENSIVE ML AUDIT: MODIS")
        print("="*80)
        for threshold in ['50', '60', '70']:
            modis_audit_results[threshold] = conduct_comprehensive_audit(modis_audit, 'MODIS', threshold)
            
            # Generate additional diagnostics for 50% threshold
            if threshold == '50':
                for feature_set in ['all_features', 'without_confidence', 'brightness_frp_only']:
                    X = modis_audit[['latitude', 'longitude', 'brightness', 'confidence', 'frp'] if feature_set == 'all_features'
                                   else (['latitude', 'longitude', 'brightness', 'frp'] if feature_set == 'without_confidence'
                                        else ['brightness', 'frp'])]
                    y = modis_audit['fire_50']
                    
                    generate_learning_curves(X, y, 'MODIS', feature_set, threshold)
                    generate_feature_importance_analysis(X, y, 'MODIS', feature_set, threshold)
    
    viirs_audit_results = {}
    if viirs_df is not None:
        viirs_audit = preprocess_for_audit(viirs_df, 'VIIRS')
        # Audit VIIRS dataset
        print("\n" + "="*80)
        print("CONDUCTING COMPREHENSIVE ML AUDIT: VIIRS")
        print("="*80)
        for threshold in ['50', '60', '70']:
            viirs_audit_results[threshold] = conduct_comprehensive_audit(viirs_audit, 'VIIRS', threshold)
            
            # Generate additional diagnostics for 50% threshold
            if threshold == '50':
                for feature_set in ['all_features', 'without_confidence', 'brightness_frp_only']:
                    X = viirs_audit[['latitude', 'longitude', 'bright_ti4', 'bright_ti5', 'confidence', 'frp'] if feature_set == 'all_features'
                                   else (['latitude', 'longitude', 'bright_ti4', 'bright_ti5', 'frp'] if feature_set == 'without_confidence'
                                        else ['bright_ti4', 'bright_ti5', 'frp'])]
                    y = viirs_audit['fire_50']
                    
                    generate_learning_curves(X, y, 'VIIRS', feature_set, threshold)
                    generate_feature_importance_analysis(X, y, 'VIIRS', feature_set, threshold)
    
    if modis_audit_results or viirs_audit_results:
        print("\n" + "="*80)
        print("GENERATING AUDIT REPORT")
        print("="*80)
        generate_audit_summary_table(modis_audit_results, viirs_audit_results)
        
        print("\n" + "="*80)
        print("AUDIT CONCLUSIONS")
        print("="*80)
        analyze_audit_results(modis_audit_results, viirs_audit_results)
        
        print("\n" + "="*80)
        print("✅ ML AUDIT COMPLETED SUCCESSFULLY!")
        print("="*80)
        print(f"\n📁 All audit outputs saved in '../results/audit_outputs' directory")
    else:
        print("\n❌ No data processed. Please check file inputs.")

if __name__ == "__main__":
    main()
