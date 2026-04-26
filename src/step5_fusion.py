# src/step5_fusion.py
# Purpose: Train a FusionNet neural network on all 35 features from the
#          Excel sheet (30 radiomics + 5 omics) to predict:
#          - aggressiveness score (regression, 0 to 1)
#          - survival risk (classification, HIGH=1 / LOW=0)
#          Uses all 1550 rows from the correlated Excel sheet for training.
#          Saves trained model to models/fusion_model.pth.
#          Prints train loss, val loss and accuracy after every epoch.

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from step1_dataset import RADIOMICS_COLS, OMICS_COLS, ALL_FEATURE_COLS

EXCEL_PATH = r"D:\brain_tumor\mri_omics_dataset_correlated.xlsx"
MODEL_DIR  = r"D:\brain_tumor\models"
os.makedirs(MODEL_DIR, exist_ok=True)

EPOCHS     = 20
BATCH_SIZE = 16
LR         = 1e-3


# ── Dataset ───────────────────────────────────────────────────────────────────
class FusionDataset(Dataset):
    def __init__(self, excel_path):
        df = pd.read_excel(excel_path, sheet_name="MRI Omics Dataset")
        df["survival_label"] = (df["survival_risk"] == "HIGH").astype(int)

        self.X   = torch.tensor(
            df[ALL_FEATURE_COLS].values.astype(np.float32))
        self.y   = torch.tensor(
            df["survival_label"].values.astype(np.float32))
        self.agg = torch.tensor(
            df["aggressiveness"].values.astype(np.float32))

        print(f"Excel rows loaded  : {len(df)}")
        print(f"Total features     : {len(ALL_FEATURE_COLS)}")
        print(f"HIGH risk slices   : {df['survival_label'].sum()}")
        print(f"LOW  risk slices   : {(df['survival_label']==0).sum()}")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.agg[idx]


# ── Model ─────────────────────────────────────────────────────────────────────
class FusionNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        self.risk_head = nn.Sequential(
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        self.agg_head = nn.Sequential(
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        shared = self.shared(x)
        risk   = self.risk_head(shared).squeeze(1)
        agg    = self.agg_head(shared).squeeze(1)
        return risk, agg


# ── Training ──────────────────────────────────────────────────────────────────
def train():
    dataset    = FusionDataset(EXCEL_PATH)
    train_size = int(0.8 * len(dataset))
    val_size   = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=False)

    model     = FusionNet(input_dim=len(ALL_FEATURE_COLS))
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    bce_loss  = nn.BCELoss()
    mse_loss  = nn.MSELoss()

    best_val_loss = float("inf")
    print(f"\nFusion model input features : {len(ALL_FEATURE_COLS)}")
    print(f"Train rows : {len(train_set)} | Val rows : {len(val_set)}")
    print(f"Training started — {EPOCHS} epochs\n")

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0

        for X_batch, y_batch, agg_batch in train_loader:
            optimizer.zero_grad()
            pred_risk, pred_agg = model(X_batch)
            loss_risk  = bce_loss(pred_risk, y_batch)
            loss_agg   = mse_loss(pred_agg,  agg_batch)
            loss       = loss_risk + 0.5 * loss_agg
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        model.eval()
        val_loss    = 0.0
        correct     = 0
        total_items = 0

        with torch.no_grad():
            for X_batch, y_batch, agg_batch in val_loader:
                pred_risk, pred_agg = model(X_batch)
                loss_risk  = bce_loss(pred_risk, y_batch)
                loss_agg   = mse_loss(pred_agg,  agg_batch)
                loss       = loss_risk + 0.5 * loss_agg
                val_loss  += loss.item()

                predicted    = (pred_risk > 0.5).float()
                correct     += (predicted == y_batch).sum().item()
                total_items += y_batch.size(0)

        val_loss /= len(val_loader)
        accuracy  = correct / total_items * 100

        print(f"  Epoch {epoch+1:02d}/{EPOCHS} — "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val Accuracy: {accuracy:.1f}%")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(),
                       os.path.join(MODEL_DIR, "fusion_model.pth"))
            print(f"  Model saved — best val loss: {best_val_loss:.4f}")

    print(f"\nFusion training complete. Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    train()