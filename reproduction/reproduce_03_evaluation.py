#!/usr/bin/env python3
"""
Research Reproduction: Section 3 - Evaluation Metrics

This module reproduces the evaluation methodologies from the paper:
  3.1 Censoring-Adjusted Metrics - Validating BRS, IBS, AUC, and CTD computation
  3.2 Treatment Arm Comparisons - Reproducing RMST-based effect estimation
  3.3 Propensity Adjustment - Reproducing causal inference with IPW weighting
"""

import warnings
warnings.filterwarnings('ignore')
import sys
import numpy as np
import pandas as pd


def reproduce_metrics_evaluation():
    """Reproduces comprehensive survival regression metrics from paper."""
    print("\n[REPRODUCTION 13/15] Reproducing Comprehensive Survival Regression Metrics")
    print("-" * 60)
    
    try:
        from auton_survival.metrics import survival_regression_metric
        from auton_survival.models.cph import DeepCoxPH
        from auton_survival import datasets
        
        # Load dataset
        outcomes, features = datasets.load_dataset("SUPPORT")
        
        # Use subset for demonstration
        n_samples = 200
        features = features.iloc[:n_samples]
        outcomes = outcomes.iloc[:n_samples]
        
        # Split into train and test sets
        split_idx = int(0.7 * len(features))
        features_train = features.iloc[:split_idx]
        outcomes_train = outcomes.iloc[:split_idx]
        features_test = features.iloc[split_idx:]
        outcomes_test = outcomes.iloc[split_idx:]
        
        # Use numerical features
        num_feats = ['age', 'num.co', 'meanbp', 'wblc', 'hrt', 'resp', 'temp', 'pafi', 'alb', 'bili', 'crea', 'sod', 'ph', 'glucose', 'bun', 'urine', 'adlp', 'adls']
        features_train = features_train[num_feats].fillna(0)
        features_test = features_test[num_feats].fillna(0)
        
        print(f"[PASS] Dataset prepared: {len(features_train)} train samples and {len(features_test)} test samples")  # 140 train, 60 test
        
        # Train a model
        model = DeepCoxPH(layers=[100])
        model.fit(features_train, outcomes_train.time, outcomes_train.event, iters=100, learning_rate=1e-4)
        
        print(f"[PASS] Model trained successfully")
        
        # Infer event-free survival probability from model
        times = [30, 90, 180]
        predictions_test = model.predict_survival(features_test, times)
        
        print(f"[PASS] Survival predictions generated: {predictions_test.shape}")
        
        # Compute Brier Score, Integrated Brier Score, Area Under ROC Curve, and Time Dependent Concordance Index
        metrics = ['brs', 'ibs', 'auc', 'ctd']
        
        print(f"[PASS] Computing multiple evaluation metrics:")
        for metric in metrics:
            score = survival_regression_metric(metric, outcomes_test, predictions_test, times, outcomes_train=outcomes_train)  # returns numpy.ndarray, numpy.float64, numpy.ndarray, list

            # Convert score to list of floats regardless of type
            score_list = [float(s) for s in np.atleast_1d(score)]
            score_str = ', '.join([f'{s:.4f}' for s in score_list])
            print(f"   {metric.upper()}: [{score_str}]")
        
        print(f"[PASS] All metrics computed successfully")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] Comprehensive metrics evaluation failed: {e}")
        return False

def reproduce_treatment_effect_rmst():
    """Reproduces Restricted Mean Survival Time (RMST) for treatment effect from paper."""
    print("\n[REPRODUCTION 14/15] Reproducing Treatment Effect - Restricted Mean Survival Time")
    print("-" * 60)
    
    try:
        from auton_survival.metrics import treatment_effect
        from auton_survival import datasets
        
        # Load dataset
        outcomes, features = datasets.load_dataset("SUPPORT")
        
        # Use subset for demonstration
        n_samples = 200
        outcomes = outcomes.iloc[:n_samples]
        features = features.iloc[:n_samples]
        
        # Create binary treatment indicator (simulate treatment assignment)
        treatment_indicator = (features['age'] > features['age'].median()).values
        
        print(f"[PASS] Dataset prepared: {len(outcomes)} samples")
        print(f"   Treatment distribution: treated={treatment_indicator.sum()}, control={len(treatment_indicator)-treatment_indicator.sum()}")  # 100 treated, 100 control
        
        # Compute the difference in RMST between treatment and control groups: RMST(treated) - RMST(control) = treatment effect in days
        effect = treatment_effect(metric='restricted_mean', 
                                  outcomes=outcomes,
                                  treatment_indicator=treatment_indicator,
                                  weights=None, 
                                  horizons=120, 
                                  n_bootstrap=500)  # samping with replacement 500 times
        
        # Bootstrap returns a list of estimates - compute mean and CI
        effect_mean = np.mean(effect)
        effect_std = np.std(effect)
        effect_ci = np.percentile(effect, [2.5, 97.5])
        
        print(f"[PASS] RMST treatment effect computed")
        print(f"   Effect estimate: {np.mean(effect):.4f} days (std: {np.std(effect):.4f})")
        print(f"   95% CI: [{effect_ci[0]:.4f}, {effect_ci[1]:.4f}]")
        print(f"   Horizon: 120 days")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] RMST treatment effect evaluation failed: {e}")
        return False

def reproduce_treatment_effect_propensity_adjusted():
    """Reproduces propensity score adjusted treatment effects from paper."""
    print("\n[REPRODUCTION 15/15] Reproducing Propensity Adjusted Treatment Effects")
    print("-" * 60)
    
    try:
        from auton_survival.metrics import treatment_effect
        from sklearn.linear_model import LogisticRegression
        from auton_survival import datasets
        
        # Load dataset
        outcomes, features = datasets.load_dataset("SUPPORT")
        
        # Use subset for demonstration
        n_samples = 200
        features = features.iloc[:n_samples]
        outcomes = outcomes.iloc[:n_samples]
        
        # Use numerical features
        num_feats = ['age', 'num.co', 'meanbp', 'wblc', 'hrt', 'resp', 'temp', 'pafi', 'alb', 'bili', 'crea', 'sod', 'ph', 'glucose', 'bun', 'urine', 'adlp', 'adls']
        features = features[num_feats].fillna(0)
        
        # Create binary treatment indicator
        treatment_indicator = (features['age'] > features['age'].median()).values
        
        print(f"[PASS] Dataset prepared: {len(outcomes)} samples")
        print(f"   Treatment distribution: treated={treatment_indicator.sum()}, control={len(treatment_indicator)-treatment_indicator.sum()}")  # 100 treated, 100 control
        
        # Train a classification model to compute "treatment" propensity scores
        model = LogisticRegression(penalty='l2').fit(features, treatment_indicator)  # penalty='l2' is Ridge Regression
        
        # P(treatment | features) - probability of receiving treatment given patient characteristics 
        treatment_propensity = model.predict_proba(features)[:, 1]  
        
        print(f"[PASS] Treatment propensity scores computed")
        print(f"   Propensity range: [{treatment_propensity.min():.3f}, {treatment_propensity.max():.3f}]")
        
        # Solving confounding bias: patients with higher age are more likely to be treated, so IPTW upweights rare cases (low-propensity treated or high-propensity control) to balance groups
        adjusted_effect = treatment_effect(metric='hazard_ratio', 
                                           outcomes=outcomes,
                                           treatment_indicator=treatment_indicator,
                                           weights=treatment_propensity, 
                                           n_bootstrap=500)
        
        # Bootstrap returns a list of estimates - compute mean and CI
        hr_mean = np.mean(adjusted_effect)
        hr_std = np.std(adjusted_effect)
        hr_ci = np.percentile(adjusted_effect, [2.5, 97.5])
        
        print(f"[PASS] Propensity-adjusted treatment effect computed")
        print(f"   Hazard ratio: {hr_mean:.4f} (std: {hr_std:.4f})")
        print(f"   95% CI: [{hr_ci[0]:.4f}, {hr_ci[1]:.4f}]")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] Propensity adjusted treatment effect evaluation failed: {e}")
        return False

def main():
    """Run all evaluation metrics reproduction checks."""
    print("Research Reproduction: Section 3 - Evaluation Metrics")
    print("=" * 70)
    
    # Reproduction 1: Comprehensive metrics evaluation
    metrics_ok = reproduce_metrics_evaluation()
    if not metrics_ok:
        print("\n[FAIL] Comprehensive metrics evaluation reproduction failed.")
        sys.exit(1)
    
    # Reproduction 2: RMST treatment effect
    rmst_ok = reproduce_treatment_effect_rmst()
    if not rmst_ok:
        print("\n[FAIL] RMST treatment effect reproduction failed.")
        sys.exit(1)
    
    # Reproduction 3: Propensity adjusted treatment effect
    propensity_ok = reproduce_treatment_effect_propensity_adjusted()
    if not propensity_ok:
        print("\n[FAIL] Propensity adjusted treatment effect reproduction failed.")
        sys.exit(1)
    
    print("\n" + "=" * 70)
    print("[SUCCESS] All evaluation metrics reproduction checks passed!")


if __name__ == "__main__":
    main()
