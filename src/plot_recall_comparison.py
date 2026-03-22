import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

classes = ["0", "D0", "D1", "D2", "D3", "D4"]

rf_recall = [0.91, 0.13, 0.13, 0.12, 0.06, 0.01]
balanced_rf_recall = [0.91, 0.11, 0.12, 0.11, 0.07, 0.01]
xgb_recall = [0.95, 0.09, 0.07, 0.12, 0.06, 0.02]

x = np.arange(len(classes))
width = 0.25

plt.figure(figsize=(10, 6))
plt.bar(x - width, rf_recall, width, label="Random Forest")
plt.bar(x, balanced_rf_recall, width, label="Balanced Random Forest")
plt.bar(x + width, xgb_recall, width, label="XGBoost")

plt.xticks(x, classes)
plt.ylabel("Recall")
plt.xlabel("Drought Class")
plt.title("Recall Comparison by Drought Class")
plt.legend()
plt.ylim(0, 1.0)

plt.tight_layout()
plt.savefig("recall_comparison.png")
plt.show()