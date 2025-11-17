# Deep Learning Research Reproduction: Survival Analysis

![Python](https://img.shields.io/badge/Python-3.10.19-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1-EE4C2C?logo=pytorch&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-1.26.4-013243?logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2.2.3-150458?logo=pandas&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0.2-F7931E?logo=scikit-learn&logoColor=white)

**CS598 Deep Learning for Healthcare - Final Project**  
**Author:** Ningyuan Xie  
**Instructor:** Prof. Jimeng Sun  
**Institution:** University of Illinois Urbana-Champaign

---

## 1. Project Overview

This project reproduced the core functionalities of the [auton-survival](https://github.com/autonlab/auton-survival) package, a comprehensive toolkit for survival analysis with censored time-to-event data. The reproduction validated 15 key functionalities covering survival regression, phenotyping, and evaluation metrics as described in the original paper:

> Nagpal, C., Potosnak, W., & Dubrawski, A. (2022). auton-survival: an Open-Source Package for Regression, Counterfactual Estimation, Evaluation and Phenotyping with Censored Time-to-Event Data. *arXiv preprint arXiv:2204.07276*.

### Resources
- **Final Report (PDF):** [View](https://ningyuan-xie.github.io/portfolio/assets/documents/uiuc/cs598-dlh/report/auton_survival_report.pdf)
- **Slides (PDF):** [View](https://ningyuan-xie.github.io/portfolio/assets/documents/uiuc/cs598-dlh/slides/auton_survival_slides.pdf)

---

## 2. Reproduction Summary

### Models Reproduced (6 models)

1. **Deep Cox Proportional Hazards (DeepCoxPH)** - Neural network extension of Cox regression
2. **Deep Survival Machines (DSM)** - Mixture of parametric survival distributions
3. **Random Survival Forests (RSF)** - Ensemble tree-based survival method
4. **Deep Recurrent Survival Machines (RDSM)** - RNN-based model for time-varying covariates
5. **Deep Cox Mixtures (DCM)** - Cox mixture models for supervised phenotyping
6. **Cox Mixtures with Heterogeneous Effects (CMHE)** - Counterfactual phenotyping model

### Functionalities Reproduced (15 experiments)

| # | Category | Functionality | Status |
|---|----------|--------------|--------|
| 1 | Installation | Core module imports and model instantiation | ✓ |
| 2 | Regression | Deep Cox PH survival regression | ✓ |
| 3 | Regression | SurvivalModel wrapper API | ✓ |
| 4 | Regression | Cross-validation experiments | ✓ |
| 5 | Regression | Counterfactual survival regression with CV | ✓ |
| 6 | Regression | Time-varying covariates (RDSM) | ✓ |
| 7 | Phenotyping | Intersectional phenotyping | ✓ |
| 8 | Phenotyping | Unsupervised clustering-based phenotyping | ✓ |
| 9 | Phenotyping | Supervised phenotyping (DCM) | ✓ |
| 10 | Phenotyping | Phenotype purity metrics | ✓ |
| 11 | Phenotyping | Virtual Twins phenotyping | ✓ |
| 12 | Phenotyping | Counterfactual phenotyping (CMHE) | ✓ |
| 13 | Evaluation | Censoring-adjusted metrics (BRS, IBS, AUC, CTD) | ✓ |
| 14 | Evaluation | Treatment effect estimation (RMST) | ✓ |
| 15 | Evaluation | Propensity-adjusted treatment effects | ✓ |

---

## 3. Environment Setup

### Requirements

- **Python:** 3.10.19

### Core Dependencies

```
numpy==1.26.4
pandas==2.2.3
scikit-learn==1.0.2
scikit-survival==0.17.2
torch==2.5.1
torchvision==0.20.1
lifelines==0.29.0
matplotlib==3.9.2
seaborn==0.13.2
tqdm==4.66.5
```

### Installation

1. **Clone this repository:**
   ```bash
   git clone https://github.com/ningyuan-xie/auton-survival-reproduction.git
   cd auton-survival-reproduction
   ```

2. **Create and activate conda environment:**
   ```bash
   conda env create -f autosurv.yml
   conda activate autosurv
   ```

   The `autosurv.yml` file includes all necessary dependencies including the auton-survival package.

---

## 4. Running the Reproduction

### Quick Start

Run all 15 reproduction experiments:

```bash
python reproduction/reproduce_paper.py
```

### Individual Reproduction Scripts

Run specific categories of experiments:

```bash
# Installation verification
python reproduction/reproduce_00_installation.py

# Survival regression experiments (6 tests)
python reproduction/reproduce_01_survival_regression.py

# Phenotyping experiments (6 tests)
python reproduction/reproduce_02_phenotyping.py

# Evaluation metrics experiments (3 tests)
python reproduction/reproduce_03_evaluation.py
```

### Expected Output

Each experiment produces:
- **Console output** with `[PASS]` or `[FAIL]` indicators
- **Execution time** for performance tracking
- **Visualizations** saved to `reproduction/` directory (Kaplan-Meier curves)
- **Quantitative metrics** printed to console

**Total runtime:** < 30 minutes for all experiments

### Extension Studies (Ablation & Validation)

Run additional experiments to validate the paper's hypotheses:

```bash
# Run all three studies (2 ablations + 1 validation)
python extension/run_extension_studies.py

# Or run individual studies:

# 1. Ablation Study: Mixture Component Analysis (k=1, 2, 3, 5)
python extension/extend_01_mixture_components.py

# 2. Ablation Study: Architecture Depth Analysis ([100] vs [100, 100])
python extension/extend_02_architecture.py

# 3. Validation Study: Cross-Dataset Generalization (SUPPORT → PBC)
python extension/extend_03_cross_dataset.py
```

**Studies Overview:**
1. **Ablation Study - Mixture Component Analysis:** Systematically varied mixture components (k=1, 2, 3, 5) in DSM and DCM to isolate their contribution to model performance and validate whether mixture-based models improve performance for heterogeneous populations.
2. **Ablation Study - Architecture Depth:** Systematically compared shallow [100] vs deep [100, 100] architectures across DeepCoxPH, DSM, and DCM to assess whether depth improved representation learning.
3. **Validation Study - Cross-Dataset Generalization:** Tested model robustness by training on SUPPORT and testing on PBC, evaluating transfer learning across different clinical contexts, feature spaces, and patient populations.

**Estimated runtime:** ~20-30 minutes for all three studies

---

## 5. Repository Structure

```
.
├── auton_survival/                                 # Core package source code
│   ├── datasets/                                   # Built-in datasets (SUPPORT, PBC, etc.)
│   ├── models/                                     # Survival models
│   │   ├── cph/                                    # Deep Cox PH
│   │   ├── dsm/                                    # Deep Survival Machines
│   │   ├── dcm/                                    # Deep Cox Mixtures
│   │   └── cmhe/                                   # Cox Mixtures Heterogeneous Effects
│   ├── estimators.py                               # High-level model wrappers
│   ├── experiments.py                              # Cross-validation utilities
│   ├── phenotyping.py                              # Phenotyping methods
│   ├── metrics.py                                  # Evaluation metrics
│   ├── preprocessing.py                            # Data preprocessing
│   └── reporting.py                                # Visualization utilities
│
├── reproduction/                                   # Reproduction scripts
│   ├── reproduce_paper.py                          # Main script (runs all 15 experiments)
│   ├── reproduce_00_installation.py
│   ├── reproduce_01_survival_regression.py
│   ├── reproduce_02_phenotyping.py
│   ├── reproduce_03_evaluation.py
│   ├── kaplan_meier_intersectional_phenotypes.png
│   ├── kaplan_meier_supervised_phenotypes.png
│   └── kaplan_meier_unsupervised_phenotypes.png
│
├── extension/                                      # Extension studies
│   ├── run_extension_studies.py                    # Master script for extension studies
│   ├── extend_01_mixture_components.py
│   ├── extend_02_architecture.py
│   └── extend_03_cross_dataset.py
│
├── examples/                                       # Original package demo notebooks
├── docs/                                           # API documentation
├── tests/                                          # Unit tests
├── autosurv.yml                                    # Conda environment specification
└── README.md                                       # This file
```

---

## 6. Key Results

Results are presented in the same order as the reproduction functions are executed.

### Survival Regression Results (Section 1)

**Example Results:** From Deep Cox PH experiment on SUPPORT dataset (200 samples, 100 iterations):
- Predictions were generated for 3 time horizons (30d, 90d, 180d) with shape: [200 patients × 3 time points]
- Training was completed in 2 minutes on CPU
- The model successfully learned risk stratification as evidenced by varying risk scores across patients

### Phenotyping Results (Section 2: 200 samples, 3 phenogroups identified)

| Metric | 1 year | 2 years | 5 years |
|--------|--------|---------|---------|
| Phenotype Purity (Instantaneous) | 0.2477 | 0.2133 | 0.1748 |
| Phenotype Purity (Integrated) | - | - | 0.2084 |

**Kaplan-Meier Curves:** Generated for intersectional, unsupervised, and supervised phenotyping approaches, demonstrating clear separation between survival curves for different phenogroups.

### Evaluation Metrics (Section 3: Test Set: 60 samples)

| Metric | Time Horizon | Score |
|--------|--------------|-------|
| Brier Score | 30 days | 0.2013 |
| Brier Score | 90 days | 0.2282 |
| Brier Score | 180 days | 0.2429 |
| Integrated Brier Score | - | 0.2272 |
| Time-Dependent AUC | 30 days | 0.5781 |
| Time-Dependent AUC | 90 days | 0.6068 |
| Time-Dependent AUC | 180 days | 0.6263 |
| Concordance Index | 30 days | 0.5561 |
| Concordance Index | 90 days | 0.5703 |
| Concordance Index | 180 days | 0.5808 |

### Treatment Effect Analysis (Section 3: 200 samples)

- **RMST Difference:** 6.91 days (std: 7.23), 95% CI: [-6.71, 21.47]
- **Hazard Ratio (Propensity-adjusted):** 0.999 (std: 0.179), 95% CI: [0.712, 1.404]
- **Bootstrap Iterations:** 500

### Extension Study Results

**Ablation Study 1 - Mixture Component Analysis (500 samples: 400 train, 100 test):**
- **DSM**: Achieved best concordance with k=1 (0.395), best AUC with k=5 (0.683)
- **DCM**: Showed significant improvement with k=2 (0.487, +26.77% vs k=1)
- **Finding**: DCM validated mixture hypothesis; DSM showed diminishing returns
- **Training time**: Remained constant across k values (0.6-0.8s)

**Ablation Study 2 - Architecture Depth Analysis (500 samples: 400 train, 100 test):**
- **DeepCoxPH**: Showed +6.88% improvement with [100,100] (3.59x parameters)
- **DSM**: Showed +1.36% improvement with [100,100] (3.35x parameters)  
- **DCM**: Showed -18.29% degradation with [100,100] (3.54x parameters)
- **Finding**: Shallow architectures sufficed; mixtures already captured complexity

**Validation Study - Cross-Dataset Generalization:**
- **Setup**: Trained on SUPPORT (500 samples: 400 train, 100 test, 38 features) → Tested on PBC (312 samples, 25 features)
- **Result**: All models generated "no valid horizons" status (zero overlapping features)
- **Finding**: Revealed that models are feature-dependent, not population-generalizable
- **Implication**: Domain-specific training is essential for clinical deployment

---

## 7. Reproducibility Notes

### What Worked Well

- **API Stability:** All documented APIs worked as described  
- **Model Training:** Convergence was achieved within expected iterations  
- **Metrics Computation:** All evaluation metrics were computed successfully  
- **Cross-platform:** Ran successfully on macOS with M2 chip  

### Known Limitations

- **Dataset Size:** Demonstrations used subsets (150-200 samples) for speed  
- **Random Seeds:** Results may vary slightly due to stochastic training  
- **Computational Cost:** Full-scale training (~9k samples) would take 10-20 minutes per model  

### Recommendations for Future Work

1. **Hyperparameter Tuning:** Systematic grid search for optimal configurations
2. **Extension Studies:** Component-wise analysis of model performance
3. **Extended Datasets:** Validation on additional medical datasets
4. **Comparison Studies:** Benchmarking against other survival analysis packages

---

## 8. Citations

### Original Package

```bibtex
  @article{nagpal2022auton,
    title={auton-survival: an Open-Source Package for Regression, Counterfactual Estimation, Evaluation and Phenotyping with Censored Time-to-Event Data},
    author={Nagpal, Chirag and Potosnak, Willa and Dubrawski, Artur},
    journal={arXiv preprint arXiv:2204.07276},
    year={2022}
  }
```

### Key Methodologies

- **DSM:** Nagpal et al. (2021) - *Deep Survival Machines*, IEEE JBHI
- **DCM:** Nagpal et al. (2021) - *Deep Cox Mixtures*, ML4H Conference
- **CMHE:** Nagpal et al. (2022) - *Counterfactual Phenotyping*, KDD
- **RDSM:** Nagpal et al. (2021) - *Deep Parametric Time-to-Event Regression*, AAAI

See the final report for complete bibliography.

---

## 9. Contributing

This is a reproduction study for academic purposes. For contributions to the original `auton-survival` package, please visit:

**Original Repository:** https://github.com/autonlab/auton-survival

---

## 10. License

MIT License

Copyright (c) 2022 Carnegie Mellon University, [Auton Lab](http://autonlab.org)

This reproduction study is conducted for educational purposes under the original MIT License.

---

## 11. Acknowledgments

- **Original Authors:** Chirag Nagpal, Willa Potosnak, Artur Dubrawski (CMU Auton Lab)
- **Course:** CS598 Deep Learning for Healthcare, UIUC
- **Instructor:** Prof. Jimeng Sun
- **Package Maintainers:** CMU Auton Lab for excellent documentation and code quality
