import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

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

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)

print("Step 4: training model...")
model = RandomForestClassifier(n_estimators=20, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

print("Step 5: predicting...")
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))