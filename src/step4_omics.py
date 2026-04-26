# src/step4_omics.py
# Purpose: Train a Random Forest model on the Excel sheet to predict
#          5 omics values (gene_expr_1, gene_expr_2, gene_expr_3,
#          mutation_score, methylation) from 30 radiomics features.
#          Uses all 1550 rows from the Excel sheet for training.
#          Saves trained model to models/omics_predictor.pkl.
#          Prints R2 score and MSE for each omics target.

import os
import sys
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from step1_dataset import RADIOMICS_COLS, OMICS_COLS

EXCEL_PATH = r"D:\brain_tumor\mri_omics_dataset_correlated.xlsx"
MODEL_DIR  = r"D:\brain_tumor\models"
os.makedirs(MODEL_DIR, exist_ok=True)


def train():
    # Load Excel
    df = pd.read_excel(EXCEL_PATH, sheet_name="MRI Omics Dataset")
    print(f"Excel rows loaded  : {len(df)}")
    print(f"Radiomics features : {len(RADIOMICS_COLS)}")
    print(f"Omics targets      : {len(OMICS_COLS)}")

    X = df[RADIOMICS_COLS].values.astype(np.float32)
    y = df[OMICS_COLS].values.astype(np.float32)

    # Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    print(f"\nTrain rows : {len(X_train)}")
    print(f"Test rows  : {len(X_test)}")

    # Scale features
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    # Train Random Forest
    print("\nTraining Random Forest...")
    rf = MultiOutputRegressor(
        RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
    )
    rf.fit(X_train, y_train)
    print("Training complete.")

    # Evaluate
    y_pred = rf.predict(X_test)
    print(f"\nResults per omics target:")
    print(f"{'Target':<20} {'R2 Score':>10} {'MSE':>12}")
    print("-" * 44)
    for i, col in enumerate(OMICS_COLS):
        r2  = r2_score(y_test[:, i], y_pred[:, i])
        mse = mean_squared_error(y_test[:, i], y_pred[:, i])
        print(f"{col:<20} {r2:>10.4f} {mse:>12.4f}")

    overall_r2  = r2_score(y_test, y_pred)
    overall_mse = mean_squared_error(y_test, y_pred)
    print(f"\nOverall R2 score : {overall_r2:.4f}")
    print(f"Overall MSE      : {overall_mse:.4f}")

    # Save model and scaler
    joblib.dump(rf,     os.path.join(MODEL_DIR, "omics_predictor.pkl"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "omics_scaler.pkl"))
    print(f"\nModels saved to models/")
    print(f"  omics_predictor.pkl")
    print(f"  omics_scaler.pkl")


if __name__ == "__main__":
    train()