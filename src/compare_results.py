import pandas as pd

results = pd.DataFrame({
    "Model": ["Random Forest", "Balanced Random Forest", "XGBoost"],
    "Accuracy": [0.5298, 0.5276, 0.5355],
    "Main Result": [
        "Better on class 0, weak on drought classes",
        "Little improvement on minority classes",
        "Slightly higher accuracy, minority classes still weak"
    ]
})

print(results)
results.to_csv("model_comparison.csv", index=False)