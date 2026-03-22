# USDM Drought Classification Project

This project studies class imbalance in USDM drought classification using machine learning.

## Topic
Addressing class imbalance in USDM drought classification with machine learning.

## Research Question
Can simple machine learning models such as Random Forest and XGBoost improve drought classification, especially for minority drought classes?

## Dataset
This project uses the parent paper dataset `USDMDataAvg.csv` from Dryad.
The dataset is used for drought classification experiments.
The large CSV file is not uploaded to this repository.

## Models
- Random Forest
- Balanced Random Forest
- XGBoost

## Main Result
XGBoost achieved the best overall accuracy among the tested models.
However, all models performed much better on the majority class `0` than on minority drought classes (`D0`–`D4`).
This shows that class imbalance is a major challenge in USDM drought classification.

## Repository Structure
- `parent_repo_files/`: original files from the parent repository
- `src/`: our project code
- `results/`: result table and figures
- `data/`: local dataset folder (not uploaded)