# USDM Drought Classification Project

This project studies class imbalance in USDM drought classification using machine learning.

## Topic
Addressing class imbalance in USDM drought classification with machine learning.

## Research Question
Can simple machine learning models such as Random Forest and XGBoost improve drought classification, especially for minority drought classes?

## Dataset
This project uses the parent paper dataset `USDMDataAvg.csv` from Dryad.

The dataset is used for drought classification experiments.  
Because the CSV file is very large, it is not uploaded to this repository.

Please download the dataset separately and place it in:

data/USDMDataAvg.csv

## Models
The current project includes the following models:

- Random Forest
- Balanced Random Forest
- XGBoost

The modified implementation platform also compares additional imbalance-aware methods in later experiments.

## Main Result
XGBoost achieved the best overall accuracy among the tested baseline models.

However, all models performed much better on the majority class `0` than on minority drought classes (`D0`–`D4`). This shows that class imbalance is a major challenge in USDM drought classification.

## Requirements
This project was developed with Python 3.12.

Install the required packages with:

pip install -r requirements.txt

## Repository Structure
- `parent_repo_files/`: original files from the parent repository
- `src/`: project code
- `result/`: result figures and tables
- `data/`: local dataset folder (not uploaded)

## How to Run
Run the scripts from the project root directory.

Example commands:

python src/check_data.py  
python src/rf_baseline.py  
python src/rf_balanced.py  
python src/xgb_baseline.py  
python src/week12_comparison.py

## Notes
- Make sure the dataset file is placed at `data/USDMDataAvg.csv`
- The dataset is large, so training may take a long time
- A computer with enough memory is recommended

## Project Goal
The goal of this project is not only to compare machine learning models by overall accuracy, but also to examine how class imbalance affects drought classification performance across different drought categories.

## Acknowledgment
This project is based on the dataset and supporting materials provided by the parent paper repository. The modeling and comparison code in this project was developed for course project experiments.