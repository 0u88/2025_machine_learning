import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import torch.nn as nn

class MLPRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(2, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.model(x)
    
    def fit(self, X_train, y_train, X_valid, y_valid, epochs=2000, lr=0.001, verbose=True):

        # === Data Conversion from dataframe to tensor ===
        X_train = torch.tensor(X_train, dtype=torch.float32)
        X_valid = torch.tensor(X_valid, dtype=torch.float32)
        y_train = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
        y_valid = torch.tensor(y_valid.values, dtype=torch.float32).unsqueeze(1)

        # === Optimizer 與 Loss ===
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        loss_fn = nn.MSELoss()

        # === 記錄 loss 用 ===
        self.train_losses = []
        self.valid_losses = []

        for epoch in range(epochs):
            # === 訓練 ===
            self.train()
            optimizer.zero_grad()
            y_pred = self(X_train)
            train_loss = loss_fn(y_pred, y_train)
            train_loss.backward()
            optimizer.step()

            # === 驗證 ===
            self.eval()
            with torch.no_grad():
                y_valid_pred = self(X_valid)
                valid_loss = loss_fn(y_valid_pred, y_valid)

            self.train_losses.append(train_loss.item())
            self.valid_losses.append(valid_loss.item())

            # 顯示訓練進度
            if verbose and epoch % 100 == 0:
                print(f"Epoch {epoch} | Train Loss: {train_loss.item():.4f} | Valid Loss: {valid_loss.item():.4f}")
        
    def predict(self, X):
        """輸入 numpy 陣列 (N, 2)，輸出實數迴歸預測"""
        self.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32)
            y_pred = self.forward(X_tensor).numpy().squeeze()
        return y_pred