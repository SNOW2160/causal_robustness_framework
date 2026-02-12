import numpy as np
import pandas as pd
from tqdm import tqdm
from src.data_loader import DataLoader
from src.models import LinearDML, CausalForest, DragonnetWrapper
from src.metrics import calculate_pehe, calculate_pss
from src.ensemble import PSSWeightedEnsemble

# CONFIG
NOISE_LEVELS = [0, 10, 20, 50, 100]
DATASETS = ["Synthetic", "Twins", "Jobs"]
N_TRIALS = 3 

def run_experiment():
    loader = DataLoader()
    results = []

    for dataset_name in DATASETS:
        print(f"\n--- Processing {dataset_name} Dataset ---")
        
        for d_noise in tqdm(NOISE_LEVELS, desc="Noise"):
            for trial in range(N_TRIALS):
                # 1. Load Data
                if dataset_name == "Synthetic": X_clean, T, Y, true_cate = loader.get_synthetic_data()
                elif dataset_name == "Twins": X_clean, T, Y, true_cate = loader.get_twins_data()
                elif dataset_name == "Jobs": X_clean, T, Y, true_cate = loader.get_jobs_data()
                
                # 2. Add Noise
                X = np.hstack([X_clean, np.random.normal(0, 1, (len(X_clean), d_noise))]) if d_noise > 0 else X_clean
                
                # 3. Placebo Test (PSS)
                T_placebo = np.random.permutation(T)
                models = {'Linear': LinearDML(), 'Forest': CausalForest(), 'Dragonnet': DragonnetWrapper(X.shape[1])}
                pss_scores = {}
                
                for name, m in models.items():
                    m.fit(X, T_placebo, Y)
                    pss = calculate_pss(m.predict_cate(X))
                    pss_scores[name] = pss
                    results.append({'Dataset': dataset_name, 'Method': name, 'Noise': d_noise, 'Metric': 'PSS', 'Value': pss})

                # 4. Real Test (PEHE)
                models_real = {'Linear': LinearDML(), 'Forest': CausalForest(), 'Dragonnet': DragonnetWrapper(X.shape[1])}
                real_preds = {}
                
                for name, m in models_real.items():
                    m.fit(X, T, Y)
                    pred = m.predict_cate(X)
                    pehe = calculate_pehe(true_cate, pred)
                    real_preds[name] = m
                    results.append({'Dataset': dataset_name, 'Method': name, 'Noise': d_noise, 'Metric': 'PEHE', 'Value': pehe})

                # 5. Ensemble
                ens = PSSWeightedEnsemble(list(real_preds.values()), list(pss_scores.values()))
                ens_pehe = calculate_pehe(true_cate, ens.predict_cate(X))
                results.append({'Dataset': dataset_name, 'Method': 'PSS-Ensemble', 'Noise': d_noise, 'Metric': 'PEHE', 'Value': ens_pehe})

    return pd.DataFrame(results)

if __name__ == "__main__":
    df = run_experiment()
    df.to_csv("final_results.csv", index=False)
    print("Experiment Complete. Run 'python plot_results.py' to see the graphs.")