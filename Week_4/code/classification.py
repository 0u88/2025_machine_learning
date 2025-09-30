import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#  1. 資料處理 
df = pd.read_csv("classification_data.csv")
X_raw = df[['lon', 'lat']].values
y_raw = df['label'].values

# --- 標準化輸入 ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

# --- 切分資料集（train:valid:test = 6:2:2）---
X_train_np, X_temp_np, y_train_np, y_temp_np = train_test_split(X_scaled, y_raw, test_size=0.4, stratify=y_raw, random_state=42)
X_valid_np, X_test_np, y_valid_np, y_test_np = train_test_split(X_temp_np, y_temp_np, test_size=0.5, stratify=y_temp_np, random_state=42)

# --- 轉換成 PyTorch Tensor ---
X_train = torch.tensor(X_train_np, dtype=torch.float32)
X_valid = torch.tensor(X_valid_np, dtype=torch.float32)
X_test = torch.tensor(X_test_np, dtype=torch.float32)

y_train = torch.tensor(y_train_np, dtype=torch.float32).unsqueeze(1)
y_valid = torch.tensor(y_valid_np, dtype=torch.float32).unsqueeze(1)
y_test = torch.tensor(y_test_np, dtype=torch.float32).unsqueeze(1)

# 2. 定義模型
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

model = MLP()

# 3. Loss & Optimizer
loss_fn = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

train_losses = []
valid_losses = []

# 4. 訓練階段：加入 validation loss
for epoch in range(1000):
    # === 訓練 ===
    model.train()
    optimizer.zero_grad()
    y_train_pred = model(X_train)
    train_loss = loss_fn(y_train_pred, y_train)
    train_loss.backward()
    optimizer.step()

    # === 驗證 ===
    model.eval()
    with torch.no_grad():
        y_valid_pred = model(X_valid)
        valid_loss = loss_fn(y_valid_pred, y_valid)

    # === 記錄 loss ===
    train_losses.append(train_loss.item())
    valid_losses.append(valid_loss.item())

    # === 輸出進度 ===
    if epoch % 100 == 0:
        print(f"Epoch {epoch} | Train Loss: {train_loss.item():.4f} | Valid Loss: {valid_loss.item():.4f}")

# 5. 評估：用 test set 評估最終準確率
from sklearn.metrics import accuracy_score

model.eval()
with torch.no_grad():
    y_prob = model(X_test)
    y_pred = (y_prob >= 0.5).float()

acc = accuracy_score(y_test.numpy(), y_pred.numpy())
print(f"✅ Final Test Accuracy: {acc:.4f}")

import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))
plt.plot(train_losses, label='Train Loss')
plt.plot(valid_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('BCELoss')
plt.title('Training and Validation Loss Curve')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()