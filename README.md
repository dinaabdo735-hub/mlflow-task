# MLflow Experiment Tracking - House Price Prediction

## 1. Description
This project demonstrates how to track and compare machine learning experiments using MLflow on the California Housing dataset. We trained a GradientBoostingRegressor with three different sets of hyperparameter configurations.

## 2. How to Run
1. Install requirements:
pip install -r requirements.txt

2. Run the training script:
python train.py

3. View results in MLflow UI:
mlflow ui

## 3. Comparison Table

Run Name | Max Depth | Learning Rate | RMSE | MAE | R²
---|---|---|---|---|---
Run_1 | 3 | 0.1 | 0.5422 | 0.3716 | 0.7756
Run_2 | 5 | 0.05 | 0.5198 | 0.3531 | 0.7938
Run_3 | 7 | 0.01 | 0.6973 | 0.5311 | 0.6290

## 4. Best Model Selection
- Selected Model: Run_2
- Why: Run_2 achieved the lowest Root Mean Squared Error (RMSE ≈ 0.5198) and the highest R² score (≈ 0.7938), making it the most accurate and balanced configuration among the three experiments.
