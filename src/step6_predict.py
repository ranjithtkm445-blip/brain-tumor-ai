# src/step6_predict.py
# Purpose: Full end-to-end inference pipeline. Takes a single .h5 MRI slice,
#          runs U-Net to predict tumor mask, computes 30 radiomics features
#          from the predicted mask, uses Random Forest to predict 5 omics
#          values, runs FusionNet to predict aggressiveness and survival risk.
#          Returns all 40 Excel columns for any new unseen slice.
#          Prints a full clinical report matching the Excel sheet format.

import os
import sys
import numpy as np
import h5py
import torch
import torch.nn as nn
import joblib

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from step1_dataset import RADIOMICS_COLS, OMICS_COLS, ALL_FEATURE_COLS
from step2_model   import UNet

DATA_DIR    = r"D:\brain_tumor\archive\BraTS2020_training_data\content\data"
MODEL_DIR   = r"D:\brain_tumor\models"
OUTPUT_DIR  = r"D:\brain_tumor\outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

UNET_PATH   = os.path.join(MODEL_DIR, "best_model.pth")
OMICS_PATH  = os.path.join(MODEL_DIR, "omics_predictor.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "omics_scaler.pkl")
FUSION_PATH = os.path.join(MODEL_DIR, "fusion_model.pth")


# ── FusionNet definition ──────────────────────────────────────────────────────
class FusionNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64),        nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 32),         nn.ReLU()
        )
        self.risk_head = nn.Sequential(nn.Linear(32, 1), nn.Sigmoid())
        self.agg_head  = nn.Sequential(nn.Linear(32, 1), nn.Sigmoid())

    def forward(self, x):
        shared = self.shared(x)
        return self.risk_head(shared).squeeze(1), \
               self.agg_head(shared).squeeze(1)


# ── Load all models ───────────────────────────────────────────────────────────
def load_models():
    unet = UNet(in_channels=4, out_channels=3)
    unet.load_state_dict(torch.load(UNET_PATH, map_location="cpu",
                                    weights_only=True))
    unet.eval()

    rf     = joblib.load(OMICS_PATH)
    scaler = joblib.load(SCALER_PATH)

    fusion = FusionNet(input_dim=len(ALL_FEATURE_COLS))
    fusion.load_state_dict(torch.load(FUSION_PATH, map_location="cpu",
                                      weights_only=True))
    fusion.eval()

    return unet, rf, scaler, fusion


# ── Load and normalize slice ──────────────────────────────────────────────────
def load_slice(fname):
    path = os.path.join(DATA_DIR, fname)
    with h5py.File(path, "r") as f:
        image = f["image"][:].astype(np.float32)
        mask  = f["mask"][:].astype(np.float32)
    for i in range(image.shape[-1]):
        ch = image[:, :, i]
        mn, mx = ch.min(), ch.max()
        if mx > mn:
            image[:, :, i] = (ch - mn) / (mx - mn)
    return image, mask


# ── Compute radiomics from predicted mask ─────────────────────────────────────
def compute_radiomics(image_np, pred_mask):
    region_names  = ["necrosis", "edema", "enhancing"]
    channel_names = ["t1", "t1ce", "t2", "flair"]
    radiomics     = {}

    for r, region in enumerate(region_names):
        rm   = (pred_mask[r] > 0.5).astype(np.float32)
        area = int(rm.sum())
        radiomics[f"{region}_area"] = area

        for c, ch in enumerate(channel_names):
            px = image_np[:, :, c][rm == 1]
            radiomics[f"{region}_{ch}_mean"] = float(px.mean()) if area > 0 else 0.0
            radiomics[f"{region}_{ch}_std"]  = float(px.std())  if area > 0 else 0.0

        if area > 0:
            all_px = np.stack([image_np[:, :, c][rm == 1]
                               for c in range(4)], axis=0)
            radiomics[f"{region}_heterogeneity"] = float(all_px.var())
        else:
            radiomics[f"{region}_heterogeneity"] = 0.0

    return radiomics


# ── Full inference pipeline ───────────────────────────────────────────────────
def predict(fname, unet, rf, scaler, fusion):
    # Parse patient and slice info
    parts      = fname.replace(".h5", "").split("_")
    patient_id = parts[1]
    slice_num  = parts[3]

    # Load slice
    image_np, gt_mask = load_slice(fname)

    # U-Net prediction
    input_tensor = torch.tensor(image_np).permute(2, 0, 1).unsqueeze(0)
    with torch.no_grad():
        pred_mask = unet(input_tensor).squeeze(0).numpy()

    # Compute 30 radiomics from predicted mask
    radiomics = compute_radiomics(image_np, pred_mask)

    # Random Forest predicts 5 omics values
    radio_vec    = np.array([[radiomics[c] for c in RADIOMICS_COLS]],
                             dtype=np.float32)
    radio_scaled = scaler.transform(radio_vec)
    omics_pred   = rf.predict(radio_scaled)[0]

    omics = {col: round(float(val), 4)
             for col, val in zip(OMICS_COLS, omics_pred)}

    # FusionNet predicts aggressiveness + survival risk
    all_features = [radiomics[c] for c in RADIOMICS_COLS] + \
                   [omics[c]     for c in OMICS_COLS]
    feat_tensor  = torch.tensor([all_features], dtype=torch.float32)

    with torch.no_grad():
        pred_risk, pred_agg = fusion(feat_tensor)
        risk_score = pred_risk.item()
        agg_score  = pred_agg.item()

    survival_risk = "HIGH" if risk_score > 0.5 else "LOW"

    # Build full 40-column result
    result = {
        "filename"      : fname,
        "patient_id"    : int(patient_id),
        "slice_num"     : int(slice_num),
    }
    result.update({k: round(v, 4) for k, v in radiomics.items()})
    result.update(omics)
    result["aggressiveness"] = round(agg_score,  4)
    result["survival_risk"]  = survival_risk
    result["risk_score"]     = round(risk_score, 4)

    return result, image_np, pred_mask, gt_mask


# ── Print clinical report ─────────────────────────────────────────────────────
def print_report(result):
    print("\n" + "=" * 55)
    print("       BRAIN TUMOR AI — CLINICAL REPORT")
    print("=" * 55)
    print(f"  Filename       : {result['filename']}")
    print(f"  Patient ID     : {result['patient_id']}")
    print(f"  Slice Number   : {result['slice_num']}")
    print("-" * 55)
    print("  RADIOMICS:")
    print(f"  Necrosis area  : {result['necrosis_area']} px")
    print(f"  Edema area     : {result['edema_area']} px")
    print(f"  Enhancing area : {result['enhancing_area']} px")
    print(f"  Edema hetero   : {result['edema_heterogeneity']:.4f}")
    print(f"  Necrosis hetero: {result['necrosis_heterogeneity']:.4f}")
    print("-" * 55)
    print("  OMICS (predicted by Random Forest):")
    print(f"  gene_expr_1    : {result['gene_expr_1']:.4f}")
    print(f"  gene_expr_2    : {result['gene_expr_2']:.4f}")
    print(f"  gene_expr_3    : {result['gene_expr_3']:.4f}")
    print(f"  mutation_score : {result['mutation_score']:.4f}")
    print(f"  methylation    : {result['methylation']:.4f}")
    print("-" * 55)
    print("  PREDICTION:")
    print(f"  Aggressiveness : {result['aggressiveness']:.4f}")
    print(f"  Risk score     : {result['risk_score']:.4f}")
    print(f"  Survival risk  : {result['survival_risk']}")
    print("=" * 55)


if __name__ == "__main__":
    print("Loading models...")
    unet, rf, scaler, fusion = load_models()
    print("Models loaded.\n")

    # Test on 3 unseen patient slices
    test_slices = [
        "volume_10_slice_70.h5",
        "volume_11_slice_80.h5",
        "volume_12_slice_60.h5",
    ]

    for fname in test_slices:
        try:
            result, image_np, pred_mask, gt_mask = predict(
                fname, unet, rf, scaler, fusion)
            print_report(result)
        except Exception as e:
            print(f"Skipped {fname}: {e}")