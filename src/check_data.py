import pandas as pd

df = pd.read_csv("data/USDMDataAvg.csv")

print("Shape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nHead:")
print(df.head())

print("\nDtypes:")
print(df.dtypes)