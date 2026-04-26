Brain Tumor AI — MRI-Based Tumor Segmentation, Radiomics Extraction and Survival Risk Prediction
End-to-end brain tumor analysis pipeline using BraTS 2020 MRI data. Performs tumor segmentation (2D U-Net), extracts 30 radiomics features, predicts 5 omics biomarkers (Random Forest, R2=0.91), and estimates survival risk (FusionNet, 93.5% accuracy) with GradCAM explainability and PDF reporting.
🔗 Live Demo → https://ranjith445-brain-tumor-ai.hf.space/

What does this project do?
When you upload a brain MRI scan, this system:

Finds the tumor — draws boundaries around necrosis, edema and enhancing tumor
Measures it — extracts 30 clinical measurements (size, shape, intensity)
Predicts biomarkers — estimates gene expression and mutation scores from imaging
Assesses risk — predicts whether the patient is HIGH or LOW survival risk
Explains itself — shows a GradCAM heatmap of what the model focused on
Reports it — generates a downloadable PDF clinical report


How it works
Brain MRI scan (.h5)
        ↓
U-Net → finds tumor regions
        ↓
Radiomics → measures tumor (area, intensity, heterogeneity)
        ↓
Random Forest → predicts gene expression and mutation scores
        ↓
FusionNet → predicts aggressiveness and survival risk
        ↓
Streamlit App → visualize everything + download PDF

Dataset

BraTS 2020 — Brain Tumor Segmentation Challenge
Each MRI slice has 4 modalities: T1, T1ce, T2, FLAIR
Each slice has 3 tumor region masks: Necrosis, Edema, Enhancing tumor
Training — 10 patients (volume_0 to volume_9), 572 valid slices
Inference — 5 unseen patients (volume_10 to volume_14)


Models
ModelWhat it doesPerformance🔵 2D U-NetSegments tumor regions from MRIVal Loss: 0.2523🟢 Random ForestPredicts 5 omics biomarkers from radiomicsR2 Score: 0.91🟠 FusionNetPredicts survival risk from 35 featuresAccuracy: 93.5%

App Features
FeatureDescription🖼️ MRI ViewerShows T1, T1ce, T2, FLAIR side by side🎯 SegmentationGround truth vs predicted tumor overlay🔍 Tumor HighlightZoomed in, color-coded tumor region🔥 GradCAMHeatmap showing what the model focused on📊 Radiomics TableAll 30 features in 3 tabs (Necrosis, Edema, Enhancing)🧬 Omics Prediction5 biomarkers predicted by Random Forest📈 Confidence ScoresSegmentation, omics and risk confidence📄 PDF ReportFull clinical report downloadable

Pipeline Steps
Step 1 — Data Loading (step1_dataset.py)
Loads .h5 MRI slices and links each one to its row in the Excel sheet by filename. Returns image, mask, radiomics, omics and survival label together.
Step 2 — U-Net Model (step2_model.py)
Lightweight 2D U-Net with encoder-decoder architecture and skip connections. Takes 4-channel MRI as input and outputs 3-channel tumor mask.
Step 3 — Segmentation Training (step3_train.py)
Trains U-Net using Dice Loss on 457 training slices for 10 epochs. Saves best model based on validation loss.
Step 4 — Omics Predictor (step4_omics.py)
Trains a Multi-Output Random Forest on 1,550 Excel rows to predict 5 omics values from 30 radiomics features. Achieves overall R2 of 0.91.
Step 5 — Fusion Model (step5_fusion.py)
Trains a two-headed FusionNet on 35 combined features to predict aggressiveness score and survival risk simultaneously. Achieves 93.5% validation accuracy.
Step 6 — Inference (step6_predict.py)
Full end-to-end inference combining all three models. Given a new MRI slice — predicts mask, computes radiomics, predicts omics, predicts risk and prints a clinical report.

Project Structure
brain_tumor_ai/
├── app.py                      ← Streamlit web app
├── Dockerfile                  ← Docker config for deployment
├── requirements.txt            ← Python dependencies
├── mri_omics_dataset.xlsx      ← Radiomics + omics dataset (1,550 rows)
├── src/
│   ├── step1_dataset.py        ← Data loader + Excel linker
│   ├── step2_model.py          ← 2D U-Net architecture
│   ├── step3_train.py          ← U-Net training loop
│   ├── step4_omics.py          ← Random Forest omics predictor
│   ├── step5_fusion.py         ← FusionNet training
│   └── step6_predict.py        ← Full inference pipeline
├── models/
│   ├── best_model.pth          ← Trained U-Net weights
│   ├── fusion_model.pth        ← Trained FusionNet weights
│   ├── omics_predictor.pkl     ← Trained Random Forest
│   └── omics_scaler.pkl        ← Feature scaler
├── data/                       ← BraTS 2020 .h5 slice files
└── outputs/                    ← Generated PDF reports

Technologies Used
CategoryToolsDeep LearningPyTorch, 2D U-NetMachine LearningScikit-learn, Random ForestMedical ImagingH5py, NumPy, OpenCVExplainabilityGradCAMWeb AppStreamlitReportingReportLabDeploymentDocker, Hugging Face Spaces, Git LFSDataPandas, OpenPyXL

Results Summary
U-Net Segmentation    → Val Loss: 0.2523  (trained on 10 patients, 10 epochs)
Random Forest Omics   → R2 Score: 0.9063  (trained on 1,550 Excel rows)
FusionNet Risk        → Accuracy: 93.5%   (trained on 1,550 Excel rows)

Limitations

Trained on only 10 patients — a production model would require 300+
Omics data is synthetically generated with realistic correlations
CPU-only inference — GPU would significantly improve speed
Not validated for clinical use


Disclaimer
This project is built for research and portfolio demonstration purposes only. It is not validated for clinical use and should not be used for medical diagnosis.

Author
Built by Ranjith Kumar as a portfolio demonstration of end-to-end biomedical AI development.
🔗 Hugging Face: https://huggingface.co/Ranjith445
🔗 Live Demo: https://ranjith445-brain-tumor-ai.hf.space/
