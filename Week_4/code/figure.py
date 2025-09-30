import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("regression_data.csv")
plt.figure(figsize=(8, 6))
sc = plt.scatter(
    df["lon"], df["lat"], c=df["value"],
    cmap="coolwarm", s=15, alpha=0.8
)

plt.colorbar(sc, label="Temperature (°C)")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Spatial Distribution of Temperature Observations")
plt.grid(True)
plt.tight_layout()
plt.show()


df = pd.read_csv("classification_data.csv")

plt.figure(figsize=(8, 6))
sc = plt.scatter(
    df["lon"], df["lat"],
    c=df["label"],
    cmap="coolwarm",
    s=15,
    alpha=0.8,
    edgecolor='k',
    linewidth=0.2
)

# 只顯示 0 和 1 的 colorbar 刻度
cbar = plt.colorbar(sc, ticks=[0, 1])
cbar.ax.set_yticklabels(['Invalid (0)', 'Valid (1)']) 

plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Spatial Distribution of Valid Grid Points")
plt.grid(True)
plt.tight_layout()
plt.show()