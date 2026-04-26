# src/step3_train.py
# Purpose: Train the 2D U-Net model on valid slices from 10 patients.
#          Uses Dice Loss to handle class imbalance (tumor pixels are small
#          compared to background). Trains for 10 epochs with batch size 4.
#          Saves best model weights to models/best_model.pth based on
#          validation loss. Prints train loss, val loss after every epoch.

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from step1_dataset import BraTSDataset
from step2_model   import UNet

DATA_DIR   = r"D:\brain_tumor\archive\BraTS2020_training_data\content\data"
EXCEL_PATH = r"D:\brain_tumor\mri_omics_dataset.xlsx"
MODEL_DIR  = r"D:\brain_tumor\models"
os.makedirs(MODEL_DIR, exist_ok=True)

EPOCHS     = 10
BATCH_SIZE = 4
LR         = 1e-3


# ── Dice Loss ─────────────────────────────────────────────────────────────────
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, preds, targets):
        preds        = preds.contiguous().view(-1)
        targets      = targets.contiguous().view(-1)
        intersection = (preds * targets).sum()
        return 1 - (2 * intersection + self.smooth) / (
            preds.sum() + targets.sum() + self.smooth)


# ── Training ──────────────────────────────────────────────────────────────────
def train():
    dataset    = BraTSDataset(DATA_DIR, EXCEL_PATH, patient_range=range(10))
    train_size = int(0.8 * len(dataset))
    val_size   = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=False)

    model     = UNet(in_channels=4, out_channels=3)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = DiceLoss()

    best_val_loss = float("inf")
    print(f"Training started — {EPOCHS} epochs\n")

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0

        for batch_idx, (images, masks, radiomics,
                        omics, labels, agg, fnames) in enumerate(train_loader):
            optimizer.zero_grad()
            pred_mask = model(images)
            loss      = criterion(pred_mask, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            percent = (batch_idx + 1) / len(train_loader) * 100
            print(f"\r  Epoch {epoch+1}/{EPOCHS} — "
                  f"Training: {percent:.1f}%", end="", flush=True)

        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_idx, (images, masks, radiomics,
                            omics, labels, agg, fnames) in enumerate(val_loader):
                pred_mask = model(images)
                loss      = criterion(pred_mask, masks)
                val_loss += loss.item()

                percent = (batch_idx + 1) / len(val_loader) * 100
                print(f"\r  Epoch {epoch+1}/{EPOCHS} — "
                      f"Validation: {percent:.1f}%", end="", flush=True)

        val_loss /= len(val_loader)
        print(f"\r  Epoch {epoch+1}/{EPOCHS} — "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(),
                       os.path.join(MODEL_DIR, "best_model.pth"))
            print(f"  Model saved — best val loss: {best_val_loss:.4f}")

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    train()