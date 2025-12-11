import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Load data
df = pd.read_csv("wireless_data.csv")

# Features and label
X = df[[
    "distance_m",
    "tx_power_dBm",
    "freq_GHz",
    "walls",
    "los",
    "rssi_dBm"
]]
y = df["quality_label"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# Train model
model = RandomForestClassifier(n_estimators=300, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# Feature importance
print("\nFeature importance:")
for name, imp in zip(X.columns, model.feature_importances_):
    print(f"{name}: {imp:.3f}")

# Example prediction
example = [[9.0, 15, 2.4, 2, 0, -66]]
print("\nExample prediction:", model.predict(example)[0])


# ===============================
# Plot: RSSI vs Throughput
# ===============================
import matplotlib.pyplot as plt

# Reload dataset for plotting
df_plot = pd.read_csv("wireless_data.csv")

plt.figure()
plt.scatter(df_plot["rssi_dBm"], df_plot["throughput_Mbps"])
plt.xlabel("RSSI (dBm)")
plt.ylabel("Throughput (Mbps)")
plt.title("RSSI vs Throughput")
plt.grid(True)
plt.tight_layout()
plt.savefig("rssi_vs_throughput.png", dpi=150)
print("Plot saved as rssi_vs_throughput.png")

