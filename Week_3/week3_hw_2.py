import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# --------------------------
# 1. Runge function & derivative
# --------------------------
def runge_function(x):
    return 1.0 / (1 + 25 * x**2)

def runge_derivative(x):
    return (-50 * x) / (1 + 25 * x**2)**2

# --------------------------
# 2. Data generation
# --------------------------
x_dense = np.linspace(-1, 1, 400)
x_edges = np.concatenate([
    np.linspace(-1, -0.8, 100),
    np.linspace(0.8, 1, 100)
])
x_all = np.unique(np.concatenate([x_dense, x_edges])).reshape(-1, 1).astype(np.float32)
y_all = runge_function(x_all).reshape(-1, 1).astype(np.float32)
dy_all = runge_derivative(x_all).reshape(-1, 1).astype(np.float32)

# Split train / val / test (70/15/15)
x_train, x_temp, y_train, y_temp, dy_train, dy_temp = train_test_split(
    x_all, y_all, dy_all, test_size=0.3, random_state=42
)
x_val, x_test, y_val, y_test, dy_val, dy_test = train_test_split(
    x_temp, y_temp, dy_temp, test_size=0.5, random_state=42
)

# Convert to tensor
x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
dy_train_tensor = torch.tensor(dy_train, dtype=torch.float32)

x_val_tensor = torch.tensor(x_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32)
dy_val_tensor = torch.tensor(dy_val, dtype=torch.float32)

x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
dy_test_tensor = torch.tensor(dy_test, dtype=torch.float32)

# --------------------------
# 3. Model
# --------------------------
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.net(x)

net = MLP()

# --------------------------
# 4. Loss & Optimizer
# --------------------------
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=0.005)
lambda_deriv = 1   # 導數 loss 權重
epochs = 2000

train_losses, val_losses = [], []

# --------------------------
# 5. Training loop
# --------------------------
for epoch in range(epochs):
    net.train()
    optimizer.zero_grad()

    # --- function prediction ---
    x_train_req = x_train_tensor.clone().requires_grad_(True)
    y_pred = net(x_train_req)

    # --- derivative prediction (autograd) ---
    dy_pred = torch.autograd.grad(
        outputs=y_pred, inputs=x_train_req,
        grad_outputs=torch.ones_like(y_pred),
        create_graph=True
    )[0]

    # --- loss ---
    loss_func = criterion(y_pred, y_train_tensor)
    loss_deriv = criterion(dy_pred, dy_train_tensor)
    loss = loss_func + lambda_deriv * loss_deriv
    loss.backward()
    optimizer.step()
    train_losses.append(loss.item())

    # --- validation ---
    net.eval()
    with torch.no_grad():
        y_val_pred = net(x_val_tensor)
    x_val_req = x_val_tensor.clone().requires_grad_(True)
    y_val_pred_for_grad = net(x_val_req)
    dy_val_pred = torch.autograd.grad(
        outputs=y_val_pred_for_grad, inputs=x_val_req,
        grad_outputs=torch.ones_like(y_val_pred_for_grad),
        create_graph=False
    )[0].detach()

    val_loss = criterion(y_val_pred, y_val_tensor) + lambda_deriv * criterion(dy_val_pred, dy_val_tensor)
    val_losses.append(val_loss.item())

    if (epoch+1) % 500 == 0:
        print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {loss.item():.6e}, Val Loss: {val_loss.item():.6e}")

# --------------------------
# 6. Test evaluation
# --------------------------
net.eval()
with torch.no_grad():
    y_test_pred = net(x_test_tensor)

x_test_req = x_test_tensor.clone().requires_grad_(True)
y_test_pred_for_grad = net(x_test_req)
dy_test_pred = torch.autograd.grad(
    outputs=y_test_pred_for_grad, inputs=x_test_req,
    grad_outputs=torch.ones_like(y_test_pred_for_grad),
    create_graph=False
)[0].detach()

test_mse_f = criterion(y_test_pred, y_test_tensor).item()
test_mse_fp = criterion(dy_test_pred, dy_test_tensor).item()
test_maxerr_f = torch.max(torch.abs(y_test_pred - y_test_tensor)).item()
test_maxerr_fp = torch.max(torch.abs(dy_test_pred - dy_test_tensor)).item()

print(f"\n[TEST] f :  MSE={test_mse_f:.6e}, MaxErr={test_maxerr_f:.6e}")
print(f"[TEST] f': MSE={test_mse_fp:.6e}, MaxErr={test_maxerr_fp:.6e}")

# --------------------------
# 7. Plot results
# --------------------------

def plot_function_and_derivative(x_val, y_val, y_val_pred, dy_val, dy_val_pred):
    """
    畫出 function 與 derivative 的真實曲線 vs NN 預測
    """
    # Dense grid for smooth true curves
    x_plot = np.linspace(-1, 1, 500)
    y_plot = runge_function(x_plot)
    dy_plot = runge_derivative(x_plot)

    # Sort validation predictions for nice plotting
    x_val = x_val.reshape(-1)
    idx = np.argsort(x_val)

    # --- Function plot ---
    plt.figure(figsize=(8,5))
    plt.plot(x_plot, y_plot, label="True Runge Function", color="blue")
    plt.plot(x_val[idx], y_val_pred[idx], "--", label="NN Prediction", color="red")
    plt.legend(); plt.title("Function Approximation (Validation)")
    plt.xlabel("x"); plt.ylabel("f(x)"); plt.grid(alpha=0.2)
    plt.show()

    # --- Derivative plot ---
    plt.figure(figsize=(8,5))
    plt.plot(x_plot, dy_plot, label="True Derivative", color="blue")
    plt.plot(x_val[idx], dy_val_pred[idx], "--", label="NN Prediction", color="red")
    plt.legend(); plt.title("Derivative Approximation (Validation)")
    plt.xlabel("x"); plt.ylabel("f'(x)"); plt.grid(alpha=0.2)
    plt.show()


# --------------------------
# Compute predictions on validation set
# --------------------------
with torch.no_grad():
    y_val_pred = net(x_val_tensor).cpu().numpy()

x_val_req = x_val_tensor.clone().requires_grad_(True)
y_val_pred_for_grad = net(x_val_req)
dy_val_pred = torch.autograd.grad(
    outputs=y_val_pred_for_grad, inputs=x_val_req,
    grad_outputs=torch.ones_like(y_val_pred_for_grad),
    create_graph=False
)[0].detach().cpu().numpy()

# Plot function & derivative
plot_function_and_derivative(x_val, y_val, y_val_pred, dy_val, dy_val_pred)

# --------------------------
# Loss curves
# --------------------------
plt.figure(figsize=(8,5))
plt.plot(train_losses, label="Training Loss")
plt.plot(val_losses, label="Validation Loss")
plt.legend(); plt.xlabel("Epoch"); plt.ylabel("Loss")
plt.title("Loss Curves"); plt.grid(alpha=0.2)
plt.show()