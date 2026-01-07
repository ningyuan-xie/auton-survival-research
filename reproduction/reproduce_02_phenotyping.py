#!/usr/bin/env python3
"""
Research Reproduction: Section 2 - Phenotyping Survival Data

This module reproduces the phenotyping methodologies from the paper:
  2.1 Intersectional Phenotyping - Validating subgroup identification methods
  2.2 Unsupervised Phenotyping - Reproducing clustering-based approaches
  2.3 Supervised Phenotyping - Reproducing survival-informed phenotype discovery
  2.4 Counterfactual Phenotyping - Demonstrating treatment heterogeneity analysis
"""

import warnings
warnings.filterwarnings('ignore')
import sys
import numpy as np
import pandas as pd
import os


def plot_and_save_kaplan_meier(outcomes, phenotypes, filename):
    """
    Helper function to plot and save Kaplan-Meier survival curves by phenogroup.

    Parameters
    ----------
    outcomes : pd.DataFrame
        DataFrame with 'time' and 'event' columns
    phenotypes : array-like
        Phenotype assignments for each sample
    filename : str
        Name of the output file (will be saved in the script's directory)

    Returns
    -------
    str
        Full path to the saved plot file
    """
    from auton_survival.reporting import plot_kaplanmeier
    from matplotlib import pyplot as plt

    plot_kaplanmeier(outcomes, groups=phenotypes)
    plt.title('Kaplan-Meier Survival Curves by Phenogroup', fontsize=14, fontweight='bold')
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Survival Probability', fontsize=12)
    plt.tight_layout()

    # Save the plot to a file in the same directory as this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    plot_filename = os.path.join(script_dir, filename)
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()  # Close the figure to free memory

    return plot_filename

def reproduce_phenotyping_intersectional():
    """Reproduces intersectional phenotyping from paper."""
    print("\n[REPRODUCTION 7/15] Reproducing auton_survival.phenotyping - Intersectional Phenotyping")
    print("-" * 60)

    try:
        from auton_survival.phenotyping import IntersectionalPhenotyper
        from auton_survival import datasets

        # Load dataset
        outcomes, features = datasets.load_dataset("SUPPORT")

        # Use subset for demonstration
        n_samples = 300
        features = features.iloc[:n_samples]
        outcomes = outcomes.iloc[:n_samples]

        print(f"[PASS] Dataset loaded: {len(features)} samples")

        # Group patients by combinations of 'ca' (cancer status) and 'age' (binned into two quantiles)
        phenotyper = IntersectionalPhenotyper(num_vars_quantiles=(0, .5, 1.0), 
                                              cat_vars=['ca'], 
                                              num_vars=['age'])

        print(f"[PASS] IntersectionalPhenotyper created")
        print(f"   Categorical variables: ['ca']")
        print(f"   Numerical variables: ['age'] (binned at median)")

        # Fit and predict phenotypes
        phenotypes = phenotyper.fit_predict(features)

        print(f"[PASS] Intersectional phenotypes assigned")
        print(f"   Number of unique phenotypes: {len(np.unique(phenotypes))}")

        # Plot the Kaplan-Meier survival estimate specific to each phenogroup
        plot_filename = plot_and_save_kaplan_meier(outcomes, phenotypes, 'kaplan_meier_intersectional_phenotypes.png')
        print(f"[PASS] Kaplan-Meier survival curves plotted and saved to '{plot_filename}'")

        return True

    except Exception as e:
        print(f"[FAIL] Intersectional phenotyping example failed: {e}")
        return False

def reproduce_phenotyping_unsupervised():
    """Reproduces unsupervised phenotyping with clustering from paper."""
    print("\n[REPRODUCTION 8/15] Reproducing auton_survival.phenotyping - Unsupervised Phenotyping")
    print("-" * 60)

    try:
        from auton_survival.phenotyping import ClusteringPhenotyper
        from auton_survival import datasets

        # Load dataset
        outcomes, features = datasets.load_dataset("SUPPORT")

        # Use subset for demonstration
        n_samples = 300
        features = features.iloc[:n_samples]
        outcomes = outcomes.iloc[:n_samples]

        # Use numerical features (need at least 8 for 8 PCA components)
        num_feats = ['age', 'num.co', 'meanbp', 'wblc', 'hrt', 'resp', 'temp', 'pafi', 'alb', 'bili', 'crea', 'sod', 'ph', 'glucose', 'bun', 'urine', 'adlp', 'adls']
        features = features[num_feats].fillna(0)

        print(f"[PASS] Dataset prepared: {len(features)} samples")

        # Dimensionality reduction using Principal Component Analysis (PCA) to 8 dimensions
        dim_red_method = 'pca'
        n_components = 8
        # We use a Gaussian Mixture Model (GMM) with 3 components and diagonal covariance
        clustering_method, n_clusters = 'gmm', 3

        # Initialize the phenotyper with the above hyperparameters
        phenotyper = ClusteringPhenotyper(clustering_method=clustering_method,
                                          dim_red_method=dim_red_method,
                                          n_components=n_components,
                                          n_clusters=n_clusters)

        print(f"[PASS] ClusteringPhenotyper created")

        # Fit and infer the phenogroups; unsupervised here because we don't have survival outcomes
        phenotypes = phenotyper.fit_predict(features)

        print(f"[PASS] Unsupervised phenotypes assigned via clustering")
        print(f"   Number of unique phenotypes: {len(np.unique(phenotypes))}")  # 3 clusters

        # Plot the phenogroup specific Kaplan-Meier survival estimate
        plot_filename = plot_and_save_kaplan_meier(outcomes, phenotypes, 'kaplan_meier_unsupervised_phenotypes.png')
        print(f"[PASS] Kaplan-Meier survival curves plotted and saved to '{plot_filename}'")

        return True

    except Exception as e:
        print(f"[FAIL] Unsupervised phenotyping example failed: {e}")
        return False

def reproduce_phenotyping_supervised():
    """Reproduces supervised phenotyping with survival models from paper."""
    print("\n[REPRODUCTION 9/15] Reproducing auton_survival.phenotyping - Supervised Phenotyping")
    print("-" * 60)

    try:
        from auton_survival.models.dcm import DeepCoxMixtures
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

        print(f"[PASS] Dataset prepared: {len(features)} samples")

        # Instantiate a DCM Model with 3 phenogroups and a single hidden layer with size 100
        model = DeepCoxMixtures(k=3, layers=[100])

        print(f"[PASS] DeepCoxMixtures model created")

        # Fit the model; supervised here because we use the survival outcomes
        model.fit(features, outcomes.time, outcomes.event, iters=100, learning_rate=1e-4)

        print(f"[PASS] Supervised phenotyping model trained")

        # Infer the latent Phenotypes, which are hidden subgroups (here k=3) that DCM infers during supervised training on survival data
        latent_z_prob = model.predict_latent_z(features)
        phenotypes = latent_z_prob.argmax(axis=1)  # assigns phenotype based on the column with the highest probability: [0.1, 0.8, 0.2] -> phenotype 1

        print(f"[PASS] Phenotypes inferred from survival outcomes")
        print(f"   Number of unique phenotypes: {len(np.unique(phenotypes))}")  # 3 phenotypes

        # Plot the phenogroup specific Kaplan-Meier survival estimate
        plot_filename = plot_and_save_kaplan_meier(outcomes, phenotypes, 'kaplan_meier_supervised_phenotypes.png')
        print(f"[PASS] Kaplan-Meier survival curves plotted and saved to '{plot_filename}'")

        return True

    except Exception as e:
        print(f"[FAIL] Supervised phenotyping example failed: {e}")
        return False

def reproduce_phenotype_purity():
    """Reproduces quantitative evaluation of phenotyping using phenotype purity from paper."""
    print("\n[REPRODUCTION 10/15] Reproducing Quantitative Evaluation of Phenotyping - Phenotype Purity")
    print("-" * 60)

    try:
        from auton_survival.metrics import phenotype_purity
        from auton_survival.models.dcm import DeepCoxMixtures
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

        print(f"[PASS] Dataset prepared: {len(features)} samples")

        # Train a DCM Model to get phenotypes
        model = DeepCoxMixtures(k=3, layers=[100])
        model.fit(features, outcomes.time, outcomes.event, iters=100, learning_rate=1e-4)

        # Infer the latent Phenotypes
        latent_z_prob = model.predict_latent_z(features)
        phenotypes = latent_z_prob.argmax(axis=1)

        print(f"[PASS] Phenotypes inferred: {len(np.unique(phenotypes))} groups")

        # Measure phenotype purity at event horizons of 1, 2 and 5 years; instantaneous means we compute the brier score at each time point
        purity_instant = phenotype_purity(phenotypes_train=phenotypes, 
                                          outcomes_train=outcomes,
                                          strategy='instantaneous',
                                          horizons=[365, 730, 1825])

        print(f"[PASS] Phenotype purity (instantaneous) computed")
        print(f"   At 1 year (365d): {purity_instant[0]:.4f}")
        print(f"   At 2 years (730d): {purity_instant[1]:.4f}")
        print(f"   At 5 years (1825d): {purity_instant[2]:.4f}")

        # Measure phenotype purity at an event horizon of 5 years; integrated means we compute the integrated brier score over the entire time horizon
        purity_integrated = phenotype_purity(phenotypes_train=phenotypes,
                                             outcomes_train=outcomes,
                                             strategy='integrated',
                                             horizons=1825)

        print(f"[PASS] Phenotype purity (integrated) computed")
        print(f"   At 5 years (1825d): {purity_integrated[0]:.4f}")

        return True

    except Exception as e:
        print(f"[FAIL] Phenotype purity evaluation failed: {e}")
        return False

def reproduce_virtual_twins_survival():
    """Reproduces Virtual Twins for survival regression from paper."""
    print("\n[REPRODUCTION 11/15] Reproducing Virtual Twins Survival Regression")
    print("-" * 60)

    try:
        from auton_survival.phenotyping import SurvivalVirtualTwinsPhenotyper
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

        # Create binary intervention indicator (simulate treatment)
        # Note: Intervention: "Who got treated?"; Phenotype: "Who would benefit from treatment?"
        interventions = (features['age'] > features['age'].median()).astype(int).values

        print(f"[PASS] Dataset prepared: {len(features)} samples")
        print(f"   Intervention distribution: treated={interventions.sum()}, control={len(interventions)-interventions.sum()}")  # 100 treated, 100 control

        # Instantiate the Survival Virtual Twins
        model = SurvivalVirtualTwinsPhenotyper(cf_method='dcph', 
                                               phenotyping_method='rfr',
                                               cf_hyperparams={'learning_rate': [1e-3], 'iters': [10]},
                                               random_seed=0)

        print(f"[PASS] SurvivalVirtualTwins model created")
        print(f"   Primary horizon: 365 days")

        # Fit the model
        # Note: API requires at least 2 horizons for cross-validation, so we use 180 and 365 as the horizons as shown in paper example
        model.fit(features, outcomes, interventions, horizons=[180, 365], metric='ibs')  # 'ibs': Integrated Brier Score

        print(f"[PASS] Virtual Twins model trained")

        # Infer the estimated counterfactual phenotype probability
        phi_probs = model.predict_proba(features)
        phenotypes = model.predict(features)

        print(f"[PASS] Counterfactual phenotype probabilities inferred")
        print(f"   Phi probabilities shape: {phi_probs.shape}")  # 200 samples * 2 treatment groups
        print(f"   Number of phenotypes: {len(np.unique(phenotypes))}")  # 2 phenotypes
        print(f"   Phenotype distribution: {np.bincount(phenotypes)}")  # [103, 97]: Phenotype 0 (low benefit): X control + Y treated = 103 total; Phenotype 1 (high benefit): Z control + W treated = 97 total

        return True

    except Exception as e:
        print(f"[FAIL] Virtual Twins survival regression failed: {e}")
        return False

def reproduce_counterfactual_phenotyping_cmhe():
    """Reproduces Cox Mixture with Heterogeneous Effects (CMHE) for counterfactual phenotyping from paper."""
    print("\n[REPRODUCTION 12/15] Reproducing Counterfactual Phenotyping - Cox Mixture with Heterogeneous Effects (CMHE)")
    print("-" * 60)

    try:
        from auton_survival.models import cmhe
        from auton_survival import datasets

        # Load dataset
        outcomes, features = datasets.load_dataset("SUPPORT")

        # Use only numerical features for simplicity
        num_feats = ['age', 'meanbp', 'hrt', 'resp', 'temp', 'alb', 'bili']
        features_clean = features[num_feats].fillna(0)

        # Use subset for demonstration
        n_samples = 150
        features_subset = features_clean.iloc[:n_samples]
        outcomes_subset = outcomes.iloc[:n_samples]

        # Create binary intervention indicator (simulate treatment)
        interventions = (features_subset['age'] > features_subset['age'].median()).astype(int).values

        print(f"[PASS] Dataset prepared: {features_subset.shape[0]} samples")  # 150 samples
        print(f"   Intervention distribution: treated={interventions.sum()}, control={len(interventions)-interventions.sum()}")  # 75 treated, 75 control

        # Convert to numpy arrays
        X = np.array(features_subset, dtype=np.float32)
        t = np.array(outcomes_subset.time, dtype=np.float32)
        e = np.array(outcomes_subset.event, dtype=np.int32)

        # Instantiate the Cox Mixture with Heterogeneous Effects model
        # k=1: 1 baseline survival phenotype (main difference between CMHE and Virtual Twins), g=2: 2 treatment effect phenotypes; it forces a single baseline survival pattern so heterogeneity is explained by treatment effects
        model = cmhe.DeepCoxMixturesHeterogenousEffects(k=1, g=2, layers=[100])

        print(f"[PASS] DeepCoxMixturesHeterogeneousEffects model created")
        print(f"   Parameters: k=1 (base survival phenotypes), g=2 (treatment effect phenotypes), layers=[100]")

        # Fit the model with interventions
        model.fit(X, t, e, interventions, iters=10, learning_rate=1e-3)

        print(f"[PASS] CMHE model trained successfully")

        # Infer the estimated counterfactual phenotypes
        latent_phi_probs = model.predict_latent_phi(X)

        print(f"[PASS] Counterfactual phenotype probabilities inferred")
        print(f"   Phi probabilities shape: {latent_phi_probs.shape}")  # 150 samples * 2 treatment groups; with k=1 (single phenotype), phi represents treatment effect probabilities across g=2 treatment groups

        # Convert probabilities to discrete treatment response groups
        phenotypes = latent_phi_probs.argmax(axis=1)
        print(f"   Number of phenotypes: {len(np.unique(phenotypes))}")  # 2 phenotypes
        print(f"   Phenotype distribution: {np.bincount(phenotypes)}")  # [28, 122]: Treatment effect groups (similar to Virtual Twins [103, 97]) - each group can contain patients from both control and treated groups

        return True

    except Exception as e:
        print(f"[FAIL] Counterfactual Phenotyping (CMHE) failed: {e}")
        return False

def main():
    """Run all phenotyping reproduction checks."""
    print("Research Reproduction: Section 2 - Phenotyping Survival Data")
    print("=" * 70)

    # Reproduction 1: Intersectional phenotyping
    intersectional_ok = reproduce_phenotyping_intersectional()
    if not intersectional_ok:
        print("\n[FAIL] Intersectional phenotyping reproduction failed.")
        sys.exit(1)

    # Reproduction 2: Unsupervised phenotyping
    unsupervised_ok = reproduce_phenotyping_unsupervised()
    if not unsupervised_ok:
        print("\n[FAIL] Unsupervised phenotyping reproduction failed.")
        sys.exit(1)

    # Reproduction 3: Supervised phenotyping
    supervised_ok = reproduce_phenotyping_supervised()
    if not supervised_ok:
        print("\n[FAIL] Supervised phenotyping reproduction failed.")
        sys.exit(1)

    # Reproduction 4: Phenotype purity
    purity_ok = reproduce_phenotype_purity()
    if not purity_ok:
        print("\n[FAIL] Phenotype purity reproduction failed.")
        sys.exit(1)

    # Reproduction 5: Virtual Twins
    virtual_twins_ok = reproduce_virtual_twins_survival()
    if not virtual_twins_ok:
        print("\n[FAIL] Virtual Twins reproduction failed.")
        sys.exit(1)

    # Reproduction 6: Counterfactual phenotyping CMHE
    cmhe_ok = reproduce_counterfactual_phenotyping_cmhe()
    if not cmhe_ok:
        print("\n[FAIL] Counterfactual phenotyping (CMHE) reproduction failed.")
        sys.exit(1)

    print("\n" + "=" * 70)
    print("[SUCCESS] All phenotyping reproduction checks passed!")


if __name__ == "__main__":
    main()
