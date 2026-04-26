# app.py
# Purpose: Streamlit app for brain tumor inference on unseen patients
#          (volume_10 to volume_14). User selects a patient and slice.
#          Only slices with confirmed tumor annotation are shown.
#          Runs full pipeline: U-Net segmentation → radiomics extraction →
#          Random Forest omics prediction → FusionNet survival risk.
#          Displays all 40 Excel columns, GradCAM heatmap, tumor highlight,
#          segmentation overlay and generates a downloadable PDF report.
#          Includes 3 confidence scores: segmentation, omics, risk.
#          Uses relative paths for Hugging Face deployment.

import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import h5py
import os
import sys
import cv2
import joblib
import io
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm
from PIL import Image as PILImage

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, "src"))

from step1_dataset import RADIOMICS_COLS, OMICS_COLS, ALL_FEATURE_COLS
from step2_model   import UNet

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR    = os.path.join(BASE_DIR, "data")
MODEL_DIR   = os.path.join(BASE_DIR, "models")
OUTPUT_DIR  = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

UNET_PATH   = os.path.join(MODEL_DIR, "best_model.pth")
OMICS_PATH  = os.path.join(MODEL_DIR, "omics_predictor.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "omics_scaler.pkl")
FUSION_PATH = os.path.join(MODEL_DIR, "fusion_model.pth")

# ── FusionNet ─────────────────────────────────────────────────────────────────
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

# ── Load models ───────────────────────────────────────────────────────────────
@st.cache_resource
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

# ── Get valid slices ──────────────────────────────────────────────────────────
@st.cache_data
def get_patient_slices(patient_id):
    prefix = f"volume_{patient_id}_"
    files  = sorted([f for f in os.listdir(DATA_DIR)
                     if f.startswith(prefix) and f.endswith(".h5")])
    valid  = []
    for fname in files:
        with h5py.File(os.path.join(DATA_DIR, fname), "r") as f:
            img  = f["image"][:]
            mask = f["mask"][:]
        if img.max() > 0 and mask.max() > 0:
            valid.append(fname)
    return valid

# ── Load and normalize slice ──────────────────────────────────────────────────
def load_slice(fname):
    with h5py.File(os.path.join(DATA_DIR, fname), "r") as f:
        image = f["image"][:].astype(np.float32)
        mask  = f["mask"][:].astype(np.float32)
    for i in range(image.shape[-1]):
        ch = image[:, :, i]
        mn, mx = ch.min(), ch.max()
        if mx > mn:
            image[:, :, i] = (ch - mn) / (mx - mn)
    return image, mask

# ── Compute all 30 radiomics ──────────────────────────────────────────────────
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

# ── Confidence scores ─────────────────────────────────────────────────────────
def compute_confidence(pred_mask, risk_score, omics_r2=0.9063):
    tumor_pixels = pred_mask[pred_mask > 0.5]
    if len(tumor_pixels) > 0:
        seg_conf = round(float(tumor_pixels.mean()) * 100, 2)
    else:
        seg_conf = 0.0
    omics_conf = round(omics_r2 * 100, 2)
    risk_conf  = round(abs(risk_score - 0.5) * 2 * 100, 2)
    return seg_conf, omics_conf, risk_conf

# ── GradCAM ───────────────────────────────────────────────────────────────────
def compute_gradcam(unet, image_np):
    input_tensor = torch.tensor(image_np).permute(2, 0, 1).unsqueeze(0)
    input_tensor.requires_grad_(True)
    gradients   = []
    activations = []

    def fwd_hook(m, i, o): activations.append(o)
    def bwd_hook(m, gi, go): gradients.append(go[0])

    hf = unet.bottleneck.block[-1].register_forward_hook(fwd_hook)
    hb = unet.bottleneck.block[-1].register_backward_hook(bwd_hook)

    out = unet(input_tensor)
    unet.zero_grad()
    out[0, 1].mean().backward()
    hf.remove()
    hb.remove()

    grads   = gradients[0].detach().numpy()[0]
    acts    = activations[0].detach().numpy()[0]
    weights = grads.mean(axis=(1, 2))
    cam     = np.zeros(acts.shape[1:], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * acts[i]
    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (240, 240))
    if cam.max() > 0:
        cam = (cam - cam.min()) / (cam.max() - cam.min())
    return cam

# ── Figures ───────────────────────────────────────────────────────────────────
def make_modality_figure(image_np):
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.patch.set_facecolor("#0e1117")
    for i, title in enumerate(["T1", "T1ce", "T2", "FLAIR"]):
        axes[i].imshow(image_np[:, :, i], cmap="gray")
        axes[i].set_title(title, color="white", fontsize=11)
        axes[i].axis("off")
    plt.tight_layout()
    return fig

def make_segmentation_figure(image_np, pred_mask, gt_mask):
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.patch.set_facecolor("#0e1117")
    region_titles = ["Necrosis", "Edema", "Enhancing"]
    cmaps         = [plt.cm.Reds, plt.cm.Blues, plt.cm.Greens]

    for i in range(3):
        axes[0][i].imshow(image_np[:, :, 3], cmap="gray")
        axes[0][i].imshow(gt_mask[:, :, i], cmap=cmaps[i], alpha=0.5)
        axes[0][i].set_title(f"GT: {region_titles[i]}", color="yellow", fontsize=10)
    axes[0][3].imshow(image_np[:, :, 3], cmap="gray")
    for i in range(3):
        axes[0][3].imshow(gt_mask[:, :, i], cmap=cmaps[i], alpha=0.3)
    axes[0][3].set_title("GT: All regions", color="yellow", fontsize=10)

    for i in range(3):
        axes[1][i].imshow(image_np[:, :, 3], cmap="gray")
        axes[1][i].imshow(pred_mask[i], cmap=cmaps[i], alpha=0.5)
        axes[1][i].set_title(f"Pred: {region_titles[i]}", color="cyan", fontsize=10)
    axes[1][3].imshow(image_np[:, :, 3], cmap="gray")
    for i in range(3):
        axes[1][3].imshow(pred_mask[i], cmap=cmaps[i], alpha=0.3)
    axes[1][3].set_title("Pred: All regions", color="cyan", fontsize=10)

    for row in axes:
        for ax in row:
            ax.axis("off")
    fig.text(0.01, 0.75, "Ground Truth", color="yellow",
             fontsize=11, va="center", rotation=90)
    fig.text(0.01, 0.25, "Predicted",    color="cyan",
             fontsize=11, va="center", rotation=90)
    plt.tight_layout()
    return fig

def make_tumor_highlight(image_np, gt_mask):
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.patch.set_facecolor("#0e1117")
    flair    = image_np[:, :, 3]
    combined = np.zeros_like(flair)
    for i in range(3):
        combined = np.maximum(combined, gt_mask[:, :, i])

    rgb = np.stack([flair, flair, flair], axis=-1)
    rgb[gt_mask[:, :, 0] > 0.5] = [1.0, 0.0, 0.0]
    rgb[gt_mask[:, :, 1] > 0.5] = [0.0, 0.4, 1.0]
    rgb[gt_mask[:, :, 2] > 0.5] = [0.0, 1.0, 0.0]

    axes[0].imshow(flair, cmap="gray")
    axes[0].set_title("Original FLAIR", color="white", fontsize=11)

    rows = np.any(combined > 0.5, axis=1)
    cols = np.any(combined > 0.5, axis=0)
    if rows.any() and cols.any():
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        pad  = 20
        rmin = max(0,   rmin - pad)
        rmax = min(240, rmax + pad)
        cmin = max(0,   cmin - pad)
        cmax = min(240, cmax + pad)
        axes[1].imshow(flair[rmin:rmax, cmin:cmax], cmap="gray")
        axes[2].imshow(rgb[rmin:rmax,   cmin:cmax])
    else:
        axes[1].imshow(flair, cmap="gray")
        axes[2].imshow(rgb)

    axes[1].set_title("Tumor Zoomed",      color="white", fontsize=11)
    axes[2].set_title("Tumor Highlighted", color="white", fontsize=11)
    axes[3].set_facecolor("#0e1117")
    axes[3].text(0.5, 0.70, "■  Necrosis",  color="red",
                 fontsize=14, ha="center", transform=axes[3].transAxes)
    axes[3].text(0.5, 0.50, "■  Edema",     color="#0066ff",
                 fontsize=14, ha="center", transform=axes[3].transAxes)
    axes[3].text(0.5, 0.30, "■  Enhancing", color="lime",
                 fontsize=14, ha="center", transform=axes[3].transAxes)
    axes[3].set_title("Legend", color="white", fontsize=11)
    axes[3].axis("off")
    for ax in axes[:3]:
        ax.axis("off")
    plt.tight_layout()
    return fig

def make_gradcam_figure(image_np, cam):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.patch.set_facecolor("#0e1117")
    axes[0].imshow(image_np[:, :, 3], cmap="gray")
    axes[0].set_title("FLAIR", color="white", fontsize=11)
    axes[1].imshow(cam, cmap="jet")
    axes[1].set_title("GradCAM", color="white", fontsize=11)
    flair_rgb = np.stack([image_np[:, :, 3]] * 3, axis=-1)
    overlay   = (0.6 * flair_rgb + 0.4 * plt.cm.jet(cam)[:, :, :3]).clip(0, 1)
    axes[2].imshow(overlay)
    axes[2].set_title("Overlay", color="white", fontsize=11)
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    return fig

def fig_to_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    buf.seek(0)
    return buf

# ── PDF report ────────────────────────────────────────────────────────────────
def generate_pdf(result, radiomics, omics, seg_conf, omics_conf,
                 risk_conf, seg_buf, tumor_buf, cam_buf):
    pdf_path = os.path.join(OUTPUT_DIR,
                            f"report_volume_{result['patient_id']}.pdf")
    doc    = SimpleDocTemplate(pdf_path, pagesize=A4)
    styles = getSampleStyleSheet()
    story  = []

    story.append(Paragraph("Brain Tumor AI — Clinical Report", styles["Title"]))
    story.append(Spacer(1, 0.4*cm))
    story.append(Paragraph(f"Patient       : volume_{result['patient_id']}",
                            styles["Normal"]))
    story.append(Paragraph(f"Slice         : {result['filename']}",
                            styles["Normal"]))
    story.append(Paragraph(
        f"Survival Risk : {result['survival_risk']} "
        f"(score: {result['risk_score']:.4f})", styles["Normal"]))
    story.append(Spacer(1, 0.4*cm))

    # Confidence scores
    story.append(Paragraph("Confidence Scores", styles["Heading2"]))
    conf_data = [
        ["Metric",                      "Confidence", "Description"],
        ["Segmentation Confidence",     f"{seg_conf:.1f}%",
         "Mean probability of predicted tumor pixels"],
        ["Omics Prediction Confidence", f"{omics_conf:.1f}%",
         "Random Forest R2 score on omics targets"],
        ["Risk Prediction Confidence",  f"{risk_conf:.1f}%",
         "Distance of risk score from decision boundary"],
    ]
    t_conf = Table(conf_data, colWidths=[6*cm, 3.5*cm, 6.5*cm])
    t_conf.setStyle(TableStyle([
        ("BACKGROUND",     (0, 0), (-1, 0), colors.HexColor("#2C3E50")),
        ("TEXTCOLOR",      (0, 0), (-1, 0), colors.white),
        ("FONTNAME",       (0, 0), (-1, 0), "Helvetica-Bold"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.HexColor("#EBF5FB"), colors.white]),
        ("GRID",    (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTSIZE",(0, 0), (-1, -1), 9),
    ]))
    story.append(t_conf)
    story.append(Spacer(1, 0.4*cm))

    # Segmentation
    story.append(Paragraph("Segmentation (Ground Truth vs Predicted)",
                            styles["Heading2"]))
    seg_tmp = os.path.join(OUTPUT_DIR, "_seg_tmp.png")
    PILImage.open(seg_buf).save(seg_tmp)
    story.append(Image(seg_tmp, width=16*cm, height=8*cm))
    story.append(Spacer(1, 0.3*cm))

    # Tumor highlight
    story.append(Paragraph("Tumor Area Highlight", styles["Heading2"]))
    tumor_tmp = os.path.join(OUTPUT_DIR, "_tumor_tmp.png")
    PILImage.open(tumor_buf).save(tumor_tmp)
    story.append(Image(tumor_tmp, width=16*cm, height=4*cm))
    story.append(Spacer(1, 0.3*cm))

    # GradCAM
    story.append(Paragraph("GradCAM Heatmap", styles["Heading2"]))
    cam_tmp = os.path.join(OUTPUT_DIR, "_cam_tmp.png")
    PILImage.open(cam_buf).save(cam_tmp)
    story.append(Image(cam_tmp, width=14*cm, height=5*cm))
    story.append(Spacer(1, 0.3*cm))

    # All 30 radiomics
    story.append(Paragraph("Radiomics Features (All 30)", styles["Heading2"]))
    for region, hdr_color, row_color in [
        ("necrosis",  "#C0392B", "#FADBD8"),
        ("edema",     "#1A5276", "#D6EAF8"),
        ("enhancing", "#1E8449", "#D5F5E3"),
    ]:
        story.append(Paragraph(region.capitalize() + " Region",
                                styles["Heading3"]))
        keys   = [c for c in RADIOMICS_COLS if c.startswith(region)]
        r_data = [["Feature", "Value"]]
        for k in keys:
            r_data.append([k, f"{radiomics[k]:.4f}"])
        t_r = Table(r_data, colWidths=[10*cm, 6*cm])
        t_r.setStyle(TableStyle([
            ("BACKGROUND",     (0, 0), (-1, 0), colors.HexColor(hdr_color)),
            ("TEXTCOLOR",      (0, 0), (-1, 0), colors.white),
            ("FONTNAME",       (0, 0), (-1, 0), "Helvetica-Bold"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1),
             [colors.HexColor(row_color), colors.white]),
            ("GRID",    (0, 0), (-1, -1), 0.5, colors.grey),
            ("FONTSIZE",(0, 0), (-1, -1), 9),
        ]))
        story.append(t_r)
        story.append(Spacer(1, 0.2*cm))

    # Omics
    story.append(Paragraph("Omics Features (Predicted by Random Forest)",
                            styles["Heading2"]))
    omics_data = [["Feature", "Value"]]
    for k in OMICS_COLS:
        omics_data.append([k, f"{omics[k]:.4f}"])
    t_o = Table(omics_data, colWidths=[10*cm, 6*cm])
    t_o.setStyle(TableStyle([
        ("BACKGROUND",     (0, 0), (-1, 0), colors.HexColor("#1E8449")),
        ("TEXTCOLOR",      (0, 0), (-1, 0), colors.white),
        ("FONTNAME",       (0, 0), (-1, 0), "Helvetica-Bold"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.HexColor("#D5F5E3"), colors.white]),
        ("GRID",    (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTSIZE",(0, 0), (-1, -1), 10),
    ]))
    story.append(t_o)
    story.append(Spacer(1, 0.3*cm))

    # Prediction results
    story.append(Paragraph("Prediction Results", styles["Heading2"]))
    pred_data = [
        ["Metric",                    "Value"],
        ["Aggressiveness Score",      f"{result['aggressiveness']:.4f}"],
        ["Risk Score",                f"{result['risk_score']:.4f}"],
        ["Survival Risk",             result["survival_risk"]],
        ["Segmentation Confidence",   f"{seg_conf:.1f}%"],
        ["Omics Confidence",          f"{omics_conf:.1f}%"],
        ["Risk Confidence",           f"{risk_conf:.1f}%"],
    ]
    t_p = Table(pred_data, colWidths=[10*cm, 6*cm])
    t_p.setStyle(TableStyle([
        ("BACKGROUND",     (0, 0), (-1, 0), colors.HexColor("#2C3E50")),
        ("TEXTCOLOR",      (0, 0), (-1, 0), colors.white),
        ("FONTNAME",       (0, 0), (-1, 0), "Helvetica-Bold"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.HexColor("#FDEBD0"), colors.white]),
        ("GRID",    (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTSIZE",(0, 0), (-1, -1), 10),
    ]))
    story.append(t_p)
    story.append(Spacer(1, 0.4*cm))
    story.append(Paragraph(
        "Disclaimer: This report is generated by an AI model for research "
        "purposes only. Not validated for clinical use.", styles["Italic"]))

    doc.build(story)
    return pdf_path

# ── Streamlit UI ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Brain Tumor AI", layout="wide")
st.title("🧠 Brain Tumor AI — Inference Dashboard")
st.markdown("Unseen patients: **volume_10 to volume_14**")

unet, rf, scaler, fusion = load_models()

col1, col2 = st.columns(2)
with col1:
    patient_id = st.selectbox("Select Patient",
                              list(range(10, 15)),
                              format_func=lambda x: f"volume_{x}")
with col2:
    with st.spinner("Scanning slices with tumor..."):
        valid_slices = get_patient_slices(patient_id)
    st.caption(f"{len(valid_slices)} slices with tumor found")
    slice_name = st.selectbox("Select Slice", valid_slices)

if st.button("Run Inference", type="primary"):
    with st.spinner("Running full pipeline..."):

        image_np, gt_mask = load_slice(slice_name)
        input_tensor = torch.tensor(image_np).permute(2, 0, 1).unsqueeze(0)
        with torch.no_grad():
            pred_mask = unet(input_tensor).squeeze(0).numpy()

        cam       = compute_gradcam(unet, image_np)
        radiomics = compute_radiomics(image_np, pred_mask)

        radio_vec    = np.array([[radiomics[c] for c in RADIOMICS_COLS]],
                                 dtype=np.float32)
        radio_scaled = scaler.transform(radio_vec)
        omics_pred   = rf.predict(radio_scaled)[0]
        omics        = {col: round(float(val), 4)
                        for col, val in zip(OMICS_COLS, omics_pred)}

        all_features = [radiomics[c] for c in RADIOMICS_COLS] + \
                       [omics[c]     for c in OMICS_COLS]
        feat_tensor  = torch.tensor([all_features], dtype=torch.float32)
        with torch.no_grad():
            pred_risk, pred_agg = fusion(feat_tensor)
            risk_score = pred_risk.item()
            agg_score  = pred_agg.item()

        seg_conf, omics_conf, risk_conf = compute_confidence(
            pred_mask, risk_score)

        parts  = slice_name.replace(".h5", "").split("_")
        result = {
            "filename"      : slice_name,
            "patient_id"    : int(parts[1]),
            "slice_num"     : int(parts[3]),
            "risk_score"    : round(risk_score, 4),
            "aggressiveness": round(agg_score,  4),
            "survival_risk" : "HIGH" if risk_score > 0.5 else "LOW",
        }
        result.update({k: round(v, 4) for k, v in radiomics.items()})
        result.update(omics)
        risk = "🔴 HIGH RISK" if risk_score > 0.5 else "🟢 LOW RISK"

    st.markdown("---")
    st.subheader("Survival Risk Prediction")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Risk Category",  risk)
    c2.metric("Risk Score",     f"{risk_score:.4f}")
    c3.metric("Aggressiveness", f"{agg_score:.4f}")
    c4.metric("Enhancing Area", f"{radiomics['enhancing_area']} px")

    st.markdown("---")
    st.subheader("Confidence Scores")
    d1, d2, d3 = st.columns(3)
    d1.metric("Segmentation Confidence", f"{seg_conf:.1f}%",
              help="Mean probability of predicted tumor pixels")
    d2.metric("Omics Confidence",        f"{omics_conf:.1f}%",
              help="Random Forest R2 score on omics prediction")
    d3.metric("Risk Confidence",         f"{risk_conf:.1f}%",
              help="Distance of risk score from decision boundary")

    st.markdown("---")
    st.subheader("MRI Modalities")
    mod_fig = make_modality_figure(image_np)
    st.pyplot(mod_fig)
    plt.close(mod_fig)

    st.markdown("---")
    st.subheader("Segmentation — Ground Truth vs Predicted")
    seg_fig = make_segmentation_figure(image_np, pred_mask, gt_mask)
    st.pyplot(seg_fig)
    plt.close(seg_fig)

    st.markdown("---")
    st.subheader("Tumor Area Highlight")
    tumor_fig = make_tumor_highlight(image_np, gt_mask)
    st.pyplot(tumor_fig)
    plt.close(tumor_fig)

    st.markdown("---")
    st.subheader("GradCAM Heatmap")
    cam_fig = make_gradcam_figure(image_np, cam)
    st.pyplot(cam_fig)
    plt.close(cam_fig)

    st.markdown("---")
    st.subheader("Radiomics Features — All 30")
    tab1, tab2, tab3 = st.tabs(["Necrosis", "Edema", "Enhancing"])
    with tab1:
        necrosis_keys = [c for c in RADIOMICS_COLS if c.startswith("necrosis")]
        st.dataframe(
            pd.DataFrame([(k, f"{radiomics[k]:.4f}") for k in necrosis_keys],
                         columns=["Feature", "Value"]),
            use_container_width=True)
    with tab2:
        edema_keys = [c for c in RADIOMICS_COLS if c.startswith("edema")]
        st.dataframe(
            pd.DataFrame([(k, f"{radiomics[k]:.4f}") for k in edema_keys],
                         columns=["Feature", "Value"]),
            use_container_width=True)
    with tab3:
        enhancing_keys = [c for c in RADIOMICS_COLS if c.startswith("enhancing")]
        st.dataframe(
            pd.DataFrame([(k, f"{radiomics[k]:.4f}") for k in enhancing_keys],
                         columns=["Feature", "Value"]),
            use_container_width=True)

    st.markdown("---")
    st.subheader("Omics Features (Predicted by Random Forest)")
    st.dataframe(
        pd.DataFrame(omics.items(), columns=["Feature", "Value"]),
        use_container_width=True)

    st.markdown("---")
    st.subheader("Download PDF Report")
    seg_buf   = fig_to_bytes(make_segmentation_figure(image_np, pred_mask, gt_mask))
    tumor_buf = fig_to_bytes(make_tumor_highlight(image_np, gt_mask))
    cam_buf   = fig_to_bytes(make_gradcam_figure(image_np, cam))
    pdf_path  = generate_pdf(result, radiomics, omics,
                             seg_conf, omics_conf, risk_conf,
                             seg_buf, tumor_buf, cam_buf)

    with open(pdf_path, "rb") as f:
        st.download_button(
            label="📄 Download Clinical Report (PDF)",
            data=f,
            file_name=f"report_volume_{result['patient_id']}.pdf",
            mime="application/pdf"
        )
    st.success("✅ Pipeline complete. PDF ready for download.")