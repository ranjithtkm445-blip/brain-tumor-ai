# src/step1_dataset.py
# Purpose: Load BraTS .h5 slices and link each slice to its corresponding
#          row in mri_omics_dataset.xlsx by filename. Returns image, mask,
#          and all 40 Excel columns per slice. Filters empty slices.
#          Normalizes image to [0, 1]. Used by both U-Net training and
#          omics predictor training.

import os
import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split

DATA_DIR   = r"D:\brain_tumor\archive\BraTS2020_training_data\content\data"
EXCEL_PATH = r"D:\brain_tumor\mri_omics_dataset.xlsx"

RADIOMICS_COLS = [
    "necrosis_area", "necrosis_t1_mean", "necrosis_t1_std",
    "necrosis_t1ce_mean", "necrosis_t1ce_std",
    "necrosis_t2_mean", "necrosis_t2_std",
    "necrosis_flair_mean", "necrosis_flair_std",
    "necrosis_heterogeneity",
    "edema_area", "edema_t1_mean", "edema_t1_std",
    "edema_t1ce_mean", "edema_t1ce_std",
    "edema_t2_mean", "edema_t2_std",
    "edema_flair_mean", "edema_flair_std",
    "edema_heterogeneity",
    "enhancing_area", "enhancing_t1_mean", "enhancing_t1_std",
    "enhancing_t1ce_mean", "enhancing_t1ce_std",
    "enhancing_t2_mean", "enhancing_t2_std",
    "enhancing_flair_mean", "enhancing_flair_std",
    "enhancing_heterogeneity"
]

OMICS_COLS = [
    "gene_expr_1", "gene_expr_2", "gene_expr_3",
    "mutation_score", "methylation"
]

ALL_FEATURE_COLS = RADIOMICS_COLS + OMICS_COLS


class BraTSDataset(Dataset):
    def __init__(self, data_dir, excel_path, patient_range=range(10)):
        self.data_dir = data_dir
        self.samples  = []

        # Load Excel and index by filename
        df = pd.read_excel(excel_path, sheet_name="MRI Omics Dataset")
        df["survival_label"] = (df["survival_risk"] == "HIGH").astype(int)
        self.excel = df.set_index("filename")

        # Scan valid slices from target patients
        all_files      = sorted([f for f in os.listdir(data_dir)
                                  if f.endswith(".h5")])
        target_volumes = [f"volume_{i}_" for i in patient_range]
        candidates     = [f for f in all_files
                          if any(f.startswith(v) for v in target_volumes)]

        print(f"Scanning {len(candidates)} slices from "
              f"{len(list(patient_range))} patients...")
        total = len(candidates)

        for idx, fname in enumerate(candidates):
            path = os.path.join(data_dir, fname)
            with h5py.File(path, "r") as f:
                image = f["image"][:]
                mask  = f["mask"][:]

            if image.max() == 0 or mask.max() == 0:
                continue
            if fname not in self.excel.index:
                continue

            self.samples.append(fname)
            percent = (idx + 1) / total * 100
            print(f"\r  Progress: {percent:.1f}%  ({idx+1}/{total})",
                  end="", flush=True)

        print(f"\n  Valid slices linked to Excel: {len(self.samples)}\n")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fname = self.samples[idx]
        path  = os.path.join(self.data_dir, fname)

        with h5py.File(path, "r") as f:
            image = f["image"][:].astype(np.float32)
            mask  = f["mask"][:].astype(np.float32)

        # Normalize each channel to [0, 1]
        for i in range(image.shape[-1]):
            ch = image[:, :, i]
            mn, mx = ch.min(), ch.max()
            if mx > mn:
                image[:, :, i] = (ch - mn) / (mx - mn)

        # Get all columns from Excel
        row            = self.excel.loc[fname]
        radiomics      = torch.tensor(
            row[RADIOMICS_COLS].values.astype(np.float32))
        omics          = torch.tensor(
            row[OMICS_COLS].values.astype(np.float32))
        label          = torch.tensor(
            row["survival_label"], dtype=torch.float32)
        aggressiveness = torch.tensor(
            row["aggressiveness"], dtype=torch.float32)

        image = torch.tensor(image).permute(2, 0, 1)
        mask  = torch.tensor(mask).permute(2, 0, 1)

        return image, mask, radiomics, omics, label, aggressiveness, fname


if __name__ == "__main__":
    dataset    = BraTSDataset(DATA_DIR, EXCEL_PATH, patient_range=range(10))
    train_size = int(0.8 * len(dataset))
    val_size   = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=4, shuffle=True)
    val_loader   = DataLoader(val_set,   batch_size=4, shuffle=False)

    images, masks, radiomics, omics, labels, agg, fnames = next(
        iter(train_loader))

    print(f"Train slices       : {len(train_set)}")
    print(f"Validation slices  : {len(val_set)}")
    print(f"Image shape        : {images.shape}")
    print(f"Mask shape         : {masks.shape}")
    print(f"Radiomics shape    : {radiomics.shape}")
    print(f"Omics shape        : {omics.shape}")
    print(f"Labels shape       : {labels.shape}")
    print(f"Aggressiveness     : {agg}")
    print(f"Sample filenames   : {list(fnames)}")

    print(f"\nRadiomics columns  : {len(RADIOMICS_COLS)}")
    print(f"Omics columns      : {len(OMICS_COLS)}")
    print(f"Total features     : {len(ALL_FEATURE_COLS)}")