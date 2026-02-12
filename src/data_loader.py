import numpy as np
import pandas as pd
import os
import requests
import io

class DataLoader:
    def get_synthetic_data(self, n_samples=1000, n_features=20):
        # IHDP-like Data
        X = np.random.normal(0, 1, (n_samples, n_features))
        logits = np.sin(X[:, 0]) + np.cos(X[:, 1])
        T = np.random.binomial(1, 1 / (1 + np.exp(-logits)))
        cate = 1 + np.sin(X[:, 2]) * np.cos(X[:, 3])
        y0 = X[:, 3] + 0.5 * X[:, 4] + np.random.normal(0, 0.5, n_samples)
        Y = T * (y0 + cate) + (1 - T) * y0
        return X, T, Y, cate

    def get_twins_data(self):
        # Downloads Twins data (X, T, Y) from CEVAE repo
        path = "data/twins.csv"
        if not os.path.exists("data"): os.makedirs("data")
        
        if not os.path.exists(path):
            print("Downloading Twins dataset...")
            base = "https://raw.githubusercontent.com/AMLab-Amsterdam/CEVAE/master/datasets/TWINS/"
            files = {"X": "twin_pairs_X_3years_samesex.csv", "T": "twin_pairs_T_3years_samesex.csv", "Y": "twin_pairs_Y_3years_samesex.csv"}
            try:
                dfs = {}
                for k, v in files.items():
                    r = requests.get(base + v); r.raise_for_status()
                    dfs[k] = pd.read_csv(io.StringIO(r.content.decode('utf-8')))
                df = pd.concat([dfs["T"], dfs["Y"], dfs["X"]], axis=1)
                df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
                df.to_csv(path, index=False)
            except Exception as e: raise e

        df = pd.read_csv(path).dropna().sample(n=min(5000, len(pd.read_csv(path))), random_state=42)
        cols = ['mort_0', 'mort_1', 'dbirwt_0', 'dbirwt_1']
        X = df.drop(columns=[c for c in cols if c in df.columns]).values[:, :30]
        y0, y1 = df['mort_0'].values, df['mort_1'].values
        T = np.random.binomial(1, 0.5, size=len(y0))
        return X, T, T*y1 + (1-T)*y0, y1-y0

    def get_jobs_data(self):
        # Downloads LaLonde data
        path = "data/jobs.csv"
        if not os.path.exists("data"): os.makedirs("data")
        
        if not os.path.exists(path):
            print("Downloading Jobs dataset...")
            url = "https://raw.githubusercontent.com/zxhdaze/lalonde/master/lalonde_data.csv"
            r = requests.get(url); r.raise_for_status()
            with open(path, 'wb') as f: f.write(r.content)

        df = pd.read_csv(path)
        # Features: age, educ, black, hispan, married, nodegree, re74, re75
        X = df[['age', 'educ', 'black', 'hispan', 'married', 'nodegree', 're74', 're75']].values
        # Normalize continuous cols (0, 1, 6, 7)
        for i in [0, 1, 6, 7]: X[:, i] = (X[:, i] - X[:, i].mean()) / (X[:, i].std() + 1e-8)
        
        # Semi-Synthetic Outcome (Standard protocol for benchmarking)
        # True CATE favors young people (col 0) with low education (col 1)
        cate = 1.0 + (X[:, 0] < 0).astype(float) + (X[:, 1] < 0).astype(float)
        y0 = X[:, 6] + 0.5 * X[:, 7] + np.random.normal(0, 0.5, len(X)) # Rich stay rich
        logits = -0.5 * X[:, 6] # Poor are treated more
        T = np.random.binomial(1, 1 / (1 + np.exp(-logits)))
        
        return X, T, T*(y0+cate) + (1-T)*y0, cate