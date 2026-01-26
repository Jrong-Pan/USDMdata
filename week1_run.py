import os
import pandas as pd
from sklearn.model_selection import train_test_split

def main():
    # dataset path (do NOT upload the big file to GitHub)
    data_path = os.path.join("data", "USDMDataAvg.csv")

    if not os.path.exists(data_path):
        print("ERROR: dataset not found.")
        print("Please download USDMDataAvg.csv from Dryad and put it into ./data/")
        return

    df = pd.read_csv(data_path)
    print("Loaded dataset successfully!")
    print("Shape:", df.shape)
    print(df.head(3))

    # 70/20/10 split
    train_df, temp_df = train_test_split(df, test_size=0.30, random_state=42)
    test_df, val_df = train_test_split(temp_df, test_size=1/3, random_state=42)

    print("Train size:", train_df.shape)
    print("Test size:", test_df.shape)
    print("Validation size:", val_df.shape)

    # save split files (optional)
    os.makedirs("output", exist_ok=True)
    train_df.to_csv("output/train.csv", index=False)
    test_df.to_csv("output/test.csv", index=False)
    val_df.to_csv("output/val.csv", index=False)

    print("Saved split datasets to ./output/")

if __name__ == "__main__":
    main()
