import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.base import BaseEstimator

# --- 1. Linear DML ---
class LinearDML(BaseEstimator):
    def __init__(self):
        pass

    def fit(self, X, T, Y):
        self.model_0 = LinearRegression().fit(X[T==0], Y[T==0])
        self.model_1 = LinearRegression().fit(X[T==1], Y[T==1])
        
    def predict_cate(self, X):
        return self.model_1.predict(X) - self.model_0.predict(X)

# --- 2. Causal Forest Wrapper ---
class CausalForest(BaseEstimator):
    def __init__(self, n_estimators=100):
        self.model_0 = RandomForestRegressor(n_estimators=n_estimators, max_depth=10, n_jobs=-1)
        self.model_1 = RandomForestRegressor(n_estimators=n_estimators, max_depth=10, n_jobs=-1)

    def fit(self, X, T, Y):
        self.model_0.fit(X[T==0], Y[T==0])
        self.model_1.fit(X[T==1], Y[T==1])

    def predict_cate(self, X):
        return self.model_1.predict(X) - self.model_0.predict(X)

# --- 3. Dragonnet (Deep Learning) ---
class Dragonnet(nn.Module):
    def __init__(self, input_dim):
        super(Dragonnet, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, 200), nn.ELU(),
            nn.Linear(200, 200), nn.ELU(),
            nn.Dropout(0.1)
        )
        self.head_0 = nn.Sequential(nn.Linear(200, 100), nn.ELU(), nn.Linear(100, 1))
        self.head_1 = nn.Sequential(nn.Linear(200, 100), nn.ELU(), nn.Linear(100, 1))
        self.head_t = nn.Sequential(nn.Linear(200, 100), nn.ELU(), nn.Linear(100, 1), nn.Sigmoid())

    def forward(self, x):
        features = self.feature_extractor(x)
        return self.head_0(features), self.head_1(features), self.head_t(features)

class DragonnetWrapper:
    def __init__(self, input_dim, epochs=50, lr=1e-3):
        self.model = Dragonnet(input_dim)
        self.epochs = epochs
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        self.bce_loss = nn.BCELoss()

    def fit(self, X, T, Y):
        X_t = torch.tensor(X, dtype=torch.float32)
        T_t = torch.tensor(T, dtype=torch.float32).unsqueeze(1)
        Y_t = torch.tensor(Y, dtype=torch.float32).unsqueeze(1)

        self.model.train()
        for _ in range(self.epochs):
            self.optimizer.zero_grad()
            y0, y1, t_pred = self.model(X_t)
            loss_y = torch.mean((1-T_t)*(y0-Y_t)**2 + T_t*(y1-Y_t)**2)
            loss_t = self.bce_loss(t_pred, T_t)
            (loss_y + 0.1 * loss_t).backward()
            self.optimizer.step()

    def predict_cate(self, X):
        self.model.eval()
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32)
            y0, y1, _ = self.model(X_t)
        return (y1 - y0).numpy().flatten()