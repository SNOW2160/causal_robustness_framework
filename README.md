# Null Intervention Stability in Causal Deep Learning

This repository contains the implementation for benchmarking **Null Intervention Stability** across various causal estimators (Dragonnet, Causal Forest, Linear DML).

## Key Features
- **Placebo Sensitivity Score (PSS):** A novel metric to quantify "Causal Hallucination."
- **Triple-Dataset Benchmark:** Validated on Synthetic (IHDP), Twins, and Jobs (LaLonde).
- **PSS-Weighted Ensemble:** A safety-first ensemble method that outperforms individual neural estimators in noisy regimes.

## Installation
1. Clone the repo: `git clone https://github.com/YOUR_USERNAME/causal_robustness_paper.git`
2. Install dependencies: `pip install -r requirements.txt`
3. Run experiments: `python main.py`
