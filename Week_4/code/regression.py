import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# 讀取資料（包含 lon, lat, value）
df = pd.read_csv("regression_data.csv")
X_raw = df[['lon', 'lat']].values
y_raw = df['value'].values

# 標準化經緯度
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

# 切分資料集（train 70%, valid 15%, test 15%）
X_train_np, X_temp_np, y_train_np, y_temp_np = train_test_split(
    X_scaled, y_raw, test_size=0.3, random_state=42
)
X_valid_np, X_test_np, y_valid_np, y_test_np = train_test_split(
    X_temp_np, y_temp_np, test_size=0.5, random_state=42
)

# 轉換為 tensor
X_train = torch.tensor(X_train_np, dtype=torch.float32)
X_valid = torch.tensor(X_valid_np, dtype=torch.float32)
X_test = torch.tensor(X_test_np, dtype=torch.float32)

y_train = torch.tensor(y_train_np, dtype=torch.float32).unsqueeze(1)
y_valid = torch.tensor(y_valid_np, dtype=torch.float32).unsqueeze(1)
y_test = torch.tensor(y_test_np, dtype=torch.float32).unsqueeze(1)

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

model = MLPRegressor()
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

train_losses = []
valid_losses = []

for epoch in range(1000):
    # === 訓練階段 ===
    model.train()
    optimizer.zero_grad()
    y_pred = model(X_train)
    train_loss = loss_fn(y_pred, y_train)
    train_loss.backward()
    optimizer.step()

    # === 驗證階段 ===
    model.eval()
    with torch.no_grad():
        y_valid_pred = model(X_valid)
        valid_loss = loss_fn(y_valid_pred, y_valid)

    # === 記錄 loss ===
    train_losses.append(train_loss.item())
    valid_losses.append(valid_loss.item())

    # 顯示訓練進度
    if epoch % 100 == 0:
        print(f"Epoch {epoch} | Train Loss: {train_loss.item():.4f} | Valid Loss: {valid_loss.item():.4f}")

model.eval()
with torch.no_grad():
    y_test_pred = model(X_test)

from sklearn.metrics import mean_squared_error
import numpy as np

# 模型評估模式
model.eval()
with torch.no_grad():
    y_test_pred = model(X_test)

# 計算 RMSE
import numpy as np

# Flatten 轉為一維陣列
y_true = y_test.numpy().flatten()
y_pred = y_test_pred.numpy().flatten()

# 計算 RMSE（手動）
mse = mean_squared_error(y_true, y_pred)
print(f"✅ Final Test MSE: {mse:.3f}")


import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))
plt.plot(train_losses, label='Train Loss')
plt.plot(valid_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Training and Validation Loss Curve')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()