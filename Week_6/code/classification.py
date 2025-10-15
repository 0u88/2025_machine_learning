import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

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
    
    def fit(self, X_train, y_train, X_valid, y_valid, epochs=2000, lr=0.001, verbose=True):

        # === Data Conversion from dataframe to tensor ===
        X_train = torch.tensor(X_train, dtype=torch.float32)
        X_valid = torch.tensor(X_valid, dtype=torch.float32)
        y_train = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
        y_valid = torch.tensor(y_valid.values, dtype=torch.float32).unsqueeze(1)

        # === Optimizer & Loss ===
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        loss_fn = nn.BCEWithLogitsLoss()

        # === Training Loop ===
        self.train_losses = []
        self.valid_losses = []

        for epoch in range(epochs):
            self.train()
            optimizer.zero_grad()
            y_train_pred = self(X_train)
            train_loss = loss_fn(y_train_pred, y_train)
            train_loss.backward()
            optimizer.step()

            self.eval()
            with torch.no_grad():
                y_valid_pred = self(X_valid)
                valid_loss = loss_fn(y_valid_pred, y_valid)

            self.train_losses.append(train_loss.item())
            self.valid_losses.append(valid_loss.item())

            if verbose and epoch % 100 == 0:
                print(f"Epoch {epoch} | Train Loss: {train_loss.item():.4f} | Valid Loss: {valid_loss.item():.4f}")
    
    def predict(self, X):
        self.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32)
            y_prob = self.forward(X_tensor)
            y_pred = (y_prob >= 0.5).float().numpy().squeeze()
        return y_pred