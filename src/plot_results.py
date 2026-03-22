import pandas as pd
import matplotlib.pyplot as plt

results = pd.DataFrame({
    "Model": ["Random Forest", "Balanced Random Forest", "XGBoost"],
    "Accuracy": [0.5298, 0.5276, 0.5355]
})

plt.figure(figsize=(8, 5))
plt.bar(results["Model"], results["Accuracy"])
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")
plt.ylim(0.52, 0.54)

for i, v in enumerate(results["Accuracy"]):
    plt.text(i, v + 0.0002, f"{v:.4f}", ha="center")

plt.tight_layout()
plt.savefig("model_accuracy_comparison.png")
plt.show()