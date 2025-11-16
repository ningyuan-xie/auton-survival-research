#!/usr/bin/env python3
"""
Research Reproduction: Section 1 - Time-to-event or Survival Regression

This module reproduces the survival analysis methodologies from the paper:
  1.1 Fitting Survival Estimators - Demonstrating Deep Cox PH and wrappers
  1.2 Importance Weighting - Validating domain adaptation techniques  
  1.3 Counterfactual Survival Regression - Reproducing treatment effect estimation
  1.4 Time-Varying Survival Regression - Reproducing RNN-based temporal models
"""

import warnings
warnings.filterwarnings('ignore')
import sys
import numpy as np
import pandas as pd


def reproduce_models_deep_cox_ph():
    """Reproduces auton_survival.models - Deep Cox PH methodology from paper."""
    print("\n[REPRODUCTION 1/15] Reproducing auton_survival.models - Deep Cox PH Example")
    print("-" * 60)
    
    try:
        from auton_survival import datasets, preprocessing, models 
        
        # Load the SUPPORT Dataset
        outcomes, features = datasets.load_dataset("SUPPORT")
        print(f"[PASS] Dataset loaded: {features.shape[0]} samples, {features.shape[1]} features")
        
        # Define feature types for preprocessing
        cat_feats = ['sex', 'dzgroup', 'dzclass', 'income', 'race', 'ca']
        num_feats = ['age', 'num.co', 'meanbp', 'wblc', 'hrt', 'resp', 'temp', 'pafi', 'alb', 'bili', 'crea', 'sod', 'ph', 'glucose', 'bun', 'urine', 'adlp', 'adls']
        
        # Preprocess (Impute and Scale) the features
        preprocessor = preprocessing.Preprocessor()
        features_processed = preprocessor.fit_transform(features, 
                                                        cat_feats=cat_feats, 
                                                        num_feats=num_feats)
        print(f"[PASS] Preprocessing completed: {features_processed.shape}")  # 9105 patients * 38 features
        
        # Use a subset for demonstration
        n_samples = 200
        features_subset = features_processed[:n_samples]
        outcomes_subset = outcomes.iloc[:n_samples]
        
        # Train a Deep Cox Proportional Hazards (DCPH) model
        model = models.cph.DeepCoxPH(layers=[32])  # Smaller for demonstration
        
        # Convert to numpy arrays with proper dtypes
        X = np.array(features_subset, dtype=np.float32)
        t = np.array(outcomes_subset.time, dtype=np.float32)
        e = np.array(outcomes_subset.event, dtype=np.int32)
        
        model.fit(X, t, e, iters=10, learning_rate=1e-3)  # Quick training
        print(f"[PASS] Deep Cox PH model trained successfully")
        
        # Make predictions
        predictions = model.predict_risk(X, t=[30, 90, 180])
        print(f"[PASS] Risk predictions generated: {predictions.shape}")  # 200 patients * 3 time horizons
        
        return True
        
    except Exception as e:
        print(f"[FAIL] Deep Cox PH example failed: {e}")
        return False

def reproduce_estimators_survival_model():
    """Reproduces auton_survival.estimators - SurvivalModel wrapper from paper."""
    print("\n[REPRODUCTION 2/15] Reproducing auton_survival.estimators - SurvivalModel Wrapper")
    print("-" * 60)
    
    try:
        from auton_survival import estimators
        from auton_survival import datasets
        
        # Load dataset
        outcomes, features = datasets.load_dataset(dataset='SUPPORT')
        print(f"[PASS] Dataset loaded: {features.shape[0]} samples")
        
        # Use subset for demonstration
        n_samples = 150
        features_subset = features.iloc[:n_samples]
        outcomes_subset = outcomes.iloc[:n_samples]
        
        # Use only numerical features for simplicity
        num_feats = ['age', 'num.co', 'meanbp', 'hrt', 'resp', 'glucose']
        features_selected = features_subset[num_feats].fillna(0)
        
        # Train a Deep Survival Machines model using the SurvivalModel class
        model = estimators.SurvivalModel(model='dsm')  # dsm: Deep Survival Machines model
        print(f"[PASS] SurvivalModel created with model='dsm'")
        
        # Fit the model
        model.fit(features_selected, outcomes_subset)
        print(f"[PASS] Model fitted successfully")
        
        # Predict risk at time horizons
        predictions = model.predict_risk(features_selected, times=[8, 12, 16])
        print(f"[PASS] Risk predictions generated: {predictions.shape}")  # 150 patients * 3 time horizons
        
        return True
        
    except Exception as e:
        print(f"[FAIL] SurvivalModel wrapper example failed: {e}")
        return False

def reproduce_experiments_survival_regression_cv():
    """Reproduces auton_survival.experiments - Survival Regression with Cross-Validation from paper."""
    print("\n[REPRODUCTION 3/15] Reproducing auton_survival.experiments - SurvivalRegressionCV")
    print("-" * 60)
    
    try:
        from auton_survival.experiments import SurvivalRegressionCV
        from auton_survival import datasets
        
        # Load dataset
        outcomes, features = datasets.load_dataset(dataset='SUPPORT')
        print(f"[PASS] Dataset loaded: {features.shape[0]} samples")
        
        # Use subset for demonstration
        n_samples = 150
        features_subset = features.iloc[:n_samples]
        outcomes_subset = outcomes.iloc[:n_samples]
        
        # Use only numerical features for simplicity
        num_feats = ['age', 'num.co', 'meanbp', 'hrt', 'resp', 'glucose']
        features_selected = features_subset[num_feats].fillna(0)
        
        # Define the Hyperparameter grid to perform Cross Validation
        hyperparam_grid = {'n_estimators': [50, 100], 'max_depth': [3, 5], 'max_features': ['sqrt', 'log2']}  # 2 * 2 * 2 = 8 models
        print(f"[PASS] Hyperparameter grid created")
        
        # Train a RSF model with cross-validation using the SurvivalRegressionCV class
        cv_model = SurvivalRegressionCV(model='rsf', num_folds=5, hyperparam_grid=hyperparam_grid)
        print(f"[PASS] SurvivalRegressionCV initialized")
        
        # Fit the model
        horizons = [30, 90, 180]
        model = cv_model.fit(features_selected, outcomes_subset, horizons)
        print(f"[PASS] Cross-validation completed")
        
        # Make predictions with the best model from CV
        predictions = model.predict_survival(features_selected, times=horizons)
        print(f"[PASS] Survival predictions generated: {predictions.shape}")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] SurvivalRegressionCV example failed: {e}")
        return False

def reproduce_importance_weighting():
    """Reproduces importance weighting with logistic regression from paper."""
    print("\n[REPRODUCTION 4/15] Reproducing Importance Weighting with Logistic Regression")
    print("-" * 60)
    
    try:
        from sklearn.linear_model import LogisticRegression
        from auton_survival.estimators import SurvivalModel
        from auton_survival.datasets import load_dataset
        
        # Load dataset
        outcomes, features = load_dataset("SUPPORT")
        
        # Use only numerical features for simplicity
        num_feats = ['age', 'meanbp', 'hrt', 'resp', 'temp', 'alb', 'bili']
        features = features[num_feats].fillna(0)
        
        # Split into source and target domains
        n_samples = 150
        features_source = features.iloc[:n_samples]
        outcomes_source = outcomes.iloc[:n_samples]
        features_target = features.iloc[n_samples:2*n_samples]
        
        print(f"[PASS] Data prepared: {len(features_source)} source samples, {len(features_target)} target samples")
        
        # Estimate Importance Weights with Logistic Regression
        domains = np.concatenate([np.zeros(len(features_source)), np.ones(len(features_target))])
        features_combined = pd.concat([features_source, features_target])
        
        p_target = LogisticRegression().fit(features_combined, domains).predict_proba(features_source)[:, 1]
        imp_weights = p_target / (1 - p_target)  # Propensity Weighting
        
        print(f"[PASS] Importance weights computed")
        print(f"   Weight range: [{imp_weights.min():.3f}, {imp_weights.max():.3f}]")
        print(f"   Mean weight: {imp_weights.mean():.3f}")
        
        # Train the Survival Regression Model
        model = SurvivalModel("dcph", layers=[100])  # Cox PH Model with 1 Hidden Layer
        model.fit(features=features_source, outcomes=outcomes_source, weights=imp_weights)
        
        print(f"[PASS] Survival model trained with importance weights")
        
        # Make predictions on target domain
        predictions = model.predict_risk(features_target, times=[30, 90, 180])
        print(f"[PASS] Risk predictions on target domain: {predictions.shape}")
        print(f"   Average risks: 30d={predictions[:,0].mean():.3f}, "
              f"90d={predictions[:,1].mean():.3f}, 180d={predictions[:,2].mean():.3f}")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] Importance weighting example failed: {e}")
        return False

def reproduce_counterfactual_survival_regression():
    """Reproduces counterfactual survival regression with cross-validation from paper."""
    print("\n[REPRODUCTION 5/15] Reproducing Counterfactual Survival Regression with CV")
    print("-" * 60)
    
    try:
        from auton_survival.experiments import CounterfactualSurvivalRegressionCV
        from auton_survival.datasets import load_dataset
        
        # Load dataset
        outcomes, features = load_dataset(dataset='SUPPORT')
        
        # Use subset for demonstration
        n_samples = 200
        features = features.iloc[:n_samples]
        outcomes = outcomes.iloc[:n_samples]
        
        # Use only numerical features for simplicity (same as function 4)
        num_feats = ['age', 'resp', 'glucose', 'meanbp', 'hrt']
        features = features[num_feats].fillna(0)
        
        # Create binary intervention indicator (simulate treatment)
        interventions = pd.Series((features.index % 2 == 0).astype(int), index=features.index)  # even index: treated (1), odd index: control (0)
        
        print(f"[PASS] Dataset loaded: {len(features)} samples")
        print(f"   Intervention distribution: treated={interventions.sum()}, control={len(interventions)-interventions.sum()}")
        
        # Hyperparameter Grid
        grid = {'layers': [[100]], 'learning_rate': [1e-3, 1e-4]}
        print(f"[PASS] Hyperparameter grid created")
        
        # Train a counterfactual Cox model with cross-validation
        model = CounterfactualSurvivalRegressionCV('dcph', 5, hyperparam_grid=grid)
        print(f"[PASS] CounterfactualSurvivalRegressionCV initialized")
        
        # Note: CounterfactualSurvivalRegressionCV trains separate models: Treated group (interventions==1) and Control group (interventions==0), so 2 hyperparameter combinations × 2 groups = 4 training runs
        horizons = [30, 90, 180]
        cf_model = model.fit(features, outcomes, interventions, horizons, metric='ibs')  # cf_model is a wrapper containing two models
        
        print(f"[PASS] Counterfactual survival model trained successfully")
        
        # Make counterfactual predictions; can also split by calling cf_model.treated_model and cf_model.control_model
        treated_survival, control_survival = cf_model.predict_counterfactual_survival(
            features, times=horizons
        )
        
        print(f"[PASS] Counterfactual survival predictions generated")
        print(f"   Treated predictions shape: {treated_survival.shape}")  # 200 patients * 3 time horizons: for 100 treated patients is the factual outcome, for 100 control patients is the counterfactual outcome
        print(f"   Control predictions shape: {control_survival.shape}")  # 200 patients * 3 time horizons: for 100 control patients is the factual outcome, for 100 treated patients is the counterfactual outcome
        
        return True
        
    except Exception as e:
        print(f"[FAIL] Counterfactual survival regression example failed: {e}")
        return False

def reproduce_time_varying_survival_regression():
    """Reproduces time-varying survival regression with recurrent neural networks from paper."""
    print("\n[REPRODUCTION 6/15] Reproducing Time-Varying Survival Regression with RNN")
    print("-" * 60)
    
    try:
        from auton_survival import datasets
        from auton_survival.models.dsm import DeepRecurrentSurvivalMachines
        
        # Load the PBC dataset with sequential data
        features, times, events = datasets.load_dataset('PBC', sequential=True)  # sequential=True: returns a list of arrays for each individual
        
        # Features is a list of arrays for sequential data
        print(f"[PASS] PBC dataset loaded with sequential=True")
        print(f"   Features type: {type(features)}")  # list
        print(f"   Features[0] shape: {features[0].shape}")  # 2 vists * 25 features for first patient
        print(f"   Number of samples: {len(features)}")  # 312 patients
        
        # Use a subset for demonstration
        n_samples = min(100, len(features))
        features_subset = features[:n_samples]
        times_subset = times[:n_samples]
        events_subset = events[:n_samples]
        
        # dtype=object creates a 1-D NumPy array where each element can be any object, allowing inner arrays to have different shapes or lengths
        features_subset = np.array(features_subset, dtype=object)
        times_subset = np.array(times_subset, dtype=object)
        events_subset = np.array(events_subset, dtype=object)
        
        print(f"[PASS] Using subset of {n_samples} samples for demonstration")
        print(f"   Data converted to numpy arrays with dtype=object")
        
        # Create DeepRecurrentSurvivalMachines model
        # Note: hidden should be an integer (number of neurons), not a list
        model = DeepRecurrentSurvivalMachines(k=3, hidden=100, typ='RNN', layers=2)  # 3 underlying distributions, 100 neurons in the hidden layer, RNN model, 2 layers
        
        print(f"[PASS] DeepRecurrentSurvivalMachines model created")
        print(f"   Parameters: k=3, hidden=100, typ='RNN', layers=2")
        
        # Fit the model - sequential data must be numpy arrays with dtype=object
        model.fit(features_subset, times_subset, events_subset, iters=10, learning_rate=1e-3)
        print(f"[PASS] Recurrent survival model trained successfully")
        
        # Make predictions
        predictions = model.predict_risk(features_subset, t=[180, 365, 730])
        print(f"[PASS] Risk predictions generated: {predictions.shape}")
        print(f"   Average risks: 180d={predictions[:,0].mean():.3f}, "
              f"365d={predictions[:,1].mean():.3f}, 730d={predictions[:,2].mean():.3f}")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] Time-varying survival regression example failed: {e}")
        return False

def main():
    """Run all survival regression reproduction checks."""
    print("Research Reproduction: Section 1 - Time-to-event or Survival Regression")
    print("=" * 70)
    
    # Reproduction 1: Deep Cox PH
    model_ok = reproduce_models_deep_cox_ph()
    if not model_ok:
        print("\n[FAIL] Deep Cox PH reproduction failed.")
        sys.exit(1)
    
    # Reproduction 2: SurvivalModel wrapper
    estimator_ok = reproduce_estimators_survival_model()
    if not estimator_ok:
        print("\n[FAIL] SurvivalModel wrapper reproduction failed.")
        sys.exit(1)
    
    # Reproduction 3: Survival Regression CV
    cv_ok = reproduce_experiments_survival_regression_cv()
    if not cv_ok:
        print("\n[FAIL] Survival Regression CV reproduction failed.")
        sys.exit(1)
    
    # Reproduction 4: Importance weighting
    weighting_ok = reproduce_importance_weighting()
    if not weighting_ok:
        print("\n[FAIL] Importance weighting reproduction failed.")
        sys.exit(1)
    
    # Reproduction 5: Counterfactual survival regression
    counterfactual_ok = reproduce_counterfactual_survival_regression()
    if not counterfactual_ok:
        print("\n[FAIL] Counterfactual survival regression reproduction failed.")
        sys.exit(1)
    
    # Reproduction 6: Time-varying survival regression
    time_varying_ok = reproduce_time_varying_survival_regression()
    if not time_varying_ok:
        print("\n[FAIL] Time-varying survival regression reproduction failed.")
        sys.exit(1)
    
    print("\n" + "=" * 70)
    print("[SUCCESS] All survival regression reproduction checks passed!")


if __name__ == "__main__":
    main()
