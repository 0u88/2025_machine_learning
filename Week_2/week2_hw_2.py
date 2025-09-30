import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

# --------------------------
# 1. Define Runge function
# --------------------------
def runge_function(x):
    return 1.0 / (1 + 25 * x**2)

# Training data
# 生成資料
x_dense = np.linspace(-1, 1, 400)
x_edges = np.concatenate([
    np.linspace(-1, -0.8, 100),
    np.linspace(0.8, 1, 100)
])
x_all = np.unique(np.concatenate([x_dense, x_edges])).reshape(-1, 1)
y_all = runge_function(x_all).reshape(-1, 1)
# 切分 80% 做訓練, 20% 做驗證
split = int(0.8 * len(x_all))
x_train, y_train = x_all[:split], y_all[:split]
x_val, y_val = x_all[split:], y_all[split:]

# 轉成 tensor
x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
x_val_tensor = torch.tensor(x_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32)

# --------------------------
# 2. Define Neural Network
# --------------------------
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
        nn.Linear(1, 128),
        nn.Tanh(),
        nn.Linear(128, 128),
        nn.Tanh(),
        nn.Linear(128, 64),
        nn.Tanh(),
        nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.model(x)

net = MLP()

# --------------------------
# 3. Training Setup
# --------------------------
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=0.01)
epochs = 2000

train_losses = []
val_losses = []

# --------------------------
# 4. Training Loop
# --------------------------
for epoch in range(epochs):
    # Training
    net.train()
    optimizer.zero_grad()
    outputs = net(x_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()
    
    train_losses.append(loss.item())

    # Validation
    net.eval()
    with torch.no_grad():
        val_outputs = net(x_val_tensor)
        val_loss = criterion(val_outputs, y_val_tensor).item()
        val_losses.append(val_loss)
        y_pred_test = net(x_val_tensor)
        test_loss = criterion(y_pred_test, y_val_tensor).item()
    #print(f"Final Test Loss: {test_loss:.6e}")

    if (epoch+1) % 500 == 0:
        print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {loss.item():.6f}, Val Loss: {val_loss:.6f}")


# --------------------------
# 5. Plot Results
# --------------------------
net.eval()
with torch.no_grad():
    y_pred_val = net(x_val_tensor).numpy()
    y_pred = net(x_val_tensor).numpy()
print(f"Final Test Loss: {test_loss:.6e}")

# Convert to numpy
x_val_np = x_val.reshape(-1)
y_val_np = y_val.reshape(-1)
y_pred_val = net(x_val_tensor).detach().numpy().reshape(-1)

# Sort by x
idx = np.argsort(x_val_np)
x_val_sorted = x_val_np[idx]
y_val_sorted = y_val_np[idx]
y_pred_sorted = y_pred_val[idx]

# Plot
plt.figure(figsize=(8,5))
plt.plot(x_val_sorted, y_val_sorted, label="Runge function (true)", color="blue")
plt.plot(x_val_sorted, y_pred_sorted, label="Neural Network prediction", color="red", linestyle="--")
plt.legend()
plt.title("Validation Results")
plt.show()

# Plot training/validation loss curves
plt.figure(figsize=(8,5))
plt.plot(train_losses, label="Training Loss")
plt.plot(val_losses, label="Validation Loss")
plt.legend()
plt.title("Loss Curves")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.show()

# --------------------------
# 6. Compute Errors
# --------------------------
mse = np.mean((y_pred - y_val)**2)
max_error = np.max(np.abs(y_pred - y_val))
print(f"Test MSE: {mse:.6f}, Test Max Error: {max_error:.6f}")