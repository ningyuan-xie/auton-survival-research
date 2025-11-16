#!/usr/bin/env python3
"""
Research Reproduction Script: Reproducing results from the Auton-Survival paper.

This script systematically reproduces the key experiments and methodologies
presented in the paper to validate the reproducibility of the research findings.

Run this with: conda activate autosurv && python reproduce_paper.py

Reproduction is organized by paper sections:
  - reproduce_01_survival_regression.py (Section 1)
  - reproduce_02_phenotyping.py (Section 2)
  - reproduce_03_evaluation.py (Section 3)
"""

import sys


# Import reproductions from Section 1: Time-to-event or Survival Regression
from reproduce_01_survival_regression import (
    reproduce_models_deep_cox_ph,
    reproduce_estimators_survival_model,
    reproduce_experiments_survival_regression_cv,
    reproduce_importance_weighting,
    reproduce_counterfactual_survival_regression,
    reproduce_time_varying_survival_regression
)

# Import reproductions from Section 2: Phenotyping Survival Data
from reproduce_02_phenotyping import (
    reproduce_phenotyping_intersectional,
    reproduce_phenotyping_unsupervised,
    reproduce_phenotyping_supervised,
    reproduce_phenotype_purity,
    reproduce_virtual_twins_survival,
    reproduce_counterfactual_phenotyping_cmhe
)

# Import reproductions from Section 3: Evaluation
from reproduce_03_evaluation import (
    reproduce_metrics_evaluation,
    reproduce_treatment_effect_rmst,
    reproduce_treatment_effect_propensity_adjusted
)

def main():
    """Execute systematic reproduction of all paper experiments."""
    print("Reproducing experiments from the paper")
    print("=" * 60)
    
    # Reproductions arranged in the order they appear in the paper
    reproductions = [
        # Section 1: Time-to-event or Survival Regression
        reproduce_models_deep_cox_ph,              # 1.1 Fitting Survival Estimators (Deep Cox PH)
        reproduce_estimators_survival_model,       # 1.1 Fitting Survival Estimators (SurvivalModel)
        reproduce_experiments_survival_regression_cv,  # 1.1 Fitting Survival Estimators (SurvivalRegressionCV)
        reproduce_importance_weighting,            # 1.2 Importance Weighting
        reproduce_counterfactual_survival_regression,  # 1.3 Counterfactual Survival Regression
        reproduce_time_varying_survival_regression,    # 1.4 Time-Varying Survival Regression
        
        # Section 2: Phenotyping Survival Data
        reproduce_phenotyping_intersectional,      # 2.1 Intersectional Phenotyping
        reproduce_phenotyping_unsupervised,        # 2.2 Unsupervised Phenotyping
        reproduce_phenotyping_supervised,          # 2.3 Supervised Phenotyping
        reproduce_phenotype_purity,                # 2.3 Supervised Phenotyping (Quantitative Evaluation)
        reproduce_virtual_twins_survival,          # 2.4 Counterfactual Phenotyping (Virtual Twins)
        reproduce_counterfactual_phenotyping_cmhe, # 2.4 Counterfactual Phenotyping (CMHE)
        
        # Section 3: Evaluation
        reproduce_metrics_evaluation,              # 3.1 Censoring-Adjusted Evaluation Metrics
        reproduce_treatment_effect_rmst,           # 3.2 Comparing Treatment Arms (RMST)
        reproduce_treatment_effect_propensity_adjusted,  # 3.3 Propensity Adjusted Treatment Effects
    ]
    
    passed = 0
    failed = 0
    
    for reproduction in reproductions:
        if reproduction():
            passed += 1
        else:
            failed += 1
    
    # Print reproduction summary
    print("\n" + "=" * 60)
    print(f"[SUMMARY] Reproduction Results: {passed} successful, {failed} failed")
    
    if failed == 0:
        print("[SUCCESS] All paper methods successfully reproduced!")
        print("\n" + "=" * 60)
        print("REPRODUCED RESEARCH FINDINGS (in paper order):")
        print("=" * 60)
        print("\n1. Time-to-Event Survival Regression:")
        print("   ✓ Deep Cox Proportional Hazards models")
        print("   ✓ Survival model wrappers and estimators")
        print("   ✓ Cross-validation for hyperparameter tuning")
        print("   ✓ Importance weighting for domain adaptation")
        print("   ✓ Counterfactual survival regression")
        print("   ✓ Time-varying regression with recurrent neural networks")
        print("\n2. Phenotyping Survival Data:")
        print("   ✓ Intersectional phenotyping approaches")
        print("   ✓ Unsupervised phenotyping via clustering")
        print("   ✓ Supervised phenotyping with Cox mixtures")
        print("   ✓ Phenotype purity evaluation metrics")
        print("   ✓ Virtual Twins for treatment heterogeneity")
        print("   ✓ Cox Mixture with Heterogeneous Effects (CMHE)")
        print("\n3. Evaluation Metrics:")
        print("   ✓ Censoring-adjusted metrics (BRS, IBS, AUC, CTD)")
        print("   ✓ Restricted Mean Survival Time (RMST) comparisons")
        print("   ✓ Propensity-adjusted treatment effect estimation")
        print("\n" + "=" * 60)
    else:
        print("[WARNING] Some reproductions failed. Review error messages above.")
        print("This may indicate reproducibility issues or environment differences.")
        sys.exit(1)

if __name__ == "__main__":
    main()
