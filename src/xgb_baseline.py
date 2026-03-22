import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier

print("Step 1: reading data...")
df = pd.read_csv("data/USDMDataAvg.csv")

print("Step 2: sorting...")
df = df.sort_values("time")

print("Step 3: splitting...")
train_df = df[df["time"] < 20180000]
test_df = df[df["time"] >= 20180000]

drop_cols = ["Unnamed: 0", "time", "grid", "drought"]
X_train = train_df.drop(columns=drop_cols)
X_test = test_df.drop(columns=drop_cols)

y_train = train_df["drought"]
y_test = test_df["drought"]

label_map = {
    "0": 0,
    "D0": 1,
    "D1": 2,
    "D2": 3,
    "D3": 4,
    "D4": 5
}

y_train_num = y_train.map(label_map)
y_test_num = y_test.map(label_map)

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)

print("Step 4: training XGBoost model...")
model = XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    objective="multi:softmax",
    num_class=6,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train_num)

print("Step 5: predicting...")
y_pred_num = model.predict(X_test)

reverse_label_map = {v: k for k, v in label_map.items()}
y_pred = pd.Series(y_pred_num).map(reverse_label_map)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))