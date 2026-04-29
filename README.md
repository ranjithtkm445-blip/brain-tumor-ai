Here is your **layman-friendly README** for the brain tumor project—simple, clear, and easy to understand.

---

# Brain Tumor Analysis using AI

**Live Demo:** [https://ranjith445-brain-tumor-ai.hf.space/](https://ranjith445-brain-tumor-ai.hf.space/) 

---

## What is this project about?

Doctors use MRI scans to detect brain tumors.
But analyzing these scans:

* Takes time
* Requires expert knowledge
* Involves multiple steps

This project builds an **AI system that can analyze brain MRI scans automatically** and provide useful insights.

---

## What does this system do?

When you upload a brain MRI scan, the system:

* Finds the tumor in the image
* Measures its size and structure
* Predicts important biological information
* Estimates patient risk level (high or low)
* Shows what the AI focused on
* Generates a full report

---

## How does it work (simple explanation)

The system works step by step:

### 1. Looks at the MRI scan

MRI images show different views of the brain.

---

### 2. Finds the tumor

The AI identifies:

* Tumor core
* Swelling around it
* Active tumor region

It draws boundaries around these areas.

---

### 3. Measures the tumor

The system calculates details like:

* Size
* Shape
* Intensity

These are called **features**.

---

### 4. Predicts biological signals

Using these features, the AI estimates:

* Gene activity
* Mutation-related information

---

### 5. Predicts risk level

Based on all information, it predicts:

* Low risk
* High risk

---

### 6. Explains its decision

The system shows a heatmap highlighting:

* Which part of the image influenced the decision

---

### 7. Generates a report

A downloadable PDF report is created with:

* Tumor details
* Predictions
* Visualizations

---

## What data was used?

* Dataset: BraTS 2020 (brain tumor MRI dataset)
* Contains MRI scans with tumor labels

For this project:

* Trained on **10 patients**
* Tested on unseen patients

---

## What models are used?

The system uses three AI models:

1. **U-Net**

   * Finds tumor regions in the image

2. **Random Forest**

   * Predicts biological signals from measurements

3. **Fusion Model (FusionNet)**

   * Combines all information and predicts risk

---

## What results does it give?

* Tumor detection (segmentation)
* Biological predictions (omics)
* Risk prediction accuracy: **93.5%**

---

## Features of the application

* View MRI images (different types)
* See tumor highlighted clearly
* View measurements and predictions
* See AI explanation (heatmap)
* Download full PDF report

---

## Important Note

* Trained on a **small dataset (10 patients)**
* Uses **synthetic biological data**

This project is built for learning and demonstration.

---

## Limitations

* Not trained on large real-world data
* Not tested in clinical settings
* Needs more data for real use

---

## Disclaimer

This project is for educational and research purposes only.
It should not be used for medical diagnosis.

---

## One-Line Summary

An AI system that analyzes brain MRI scans, detects tumors, predicts risk, and explains its decisions.

---

## Author

Built by Ranjith Kumar as a biomedical AI portfolio project.

---

