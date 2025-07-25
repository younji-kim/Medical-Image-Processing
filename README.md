# CT ↔ MRI Image Bi-Directional Conversion using Deep Learning

This project proposes a medical imaging solution to convert CT and MRI images bi-directionally using generative models. CT and MRI scans each have benefits and limitations depending on patient conditions, cost, and imaging needs. By enabling image translation between the two modalities, this project aims to improve accessibility, reduce diagnostic limitations, and streamline imaging workflows — particularly for patients who cannot undergo both scans.

> Developed as a part of the *Medical Image Processing* course at Ewha Womans University.

---

## 🚀 Overview

- 🩺 **Objective:** Build a program that translates CT ↔ MRI images using deep learning  
- ⚙️ **Methods Used:** Semantic segmentation (DeepLabV3+) + Image translation (CycleGAN & StarGAN)  
- 📊 **Evaluation Metrics:** PSNR and SSIM for quantitative comparison  
- ✅ **Result:** StarGAN significantly outperformed CycleGAN in translation quality and flexibility  

---

## 📚 Background

CT (Computed Tomography) and MRI (Magnetic Resonance Imaging) are widely used in diagnostics but are not interchangeable. MRI provides clearer images of soft tissue but is more expensive and restricted for certain patients. CT is faster and more affordable but uses ionizing radiation.

### Limitations in current practice:
- MRI is not suitable for patients with pacemakers or claustrophobia
- CT is unsuitable for those sensitive to radiation or contrast agents
- Both scans are costly to perform together

By creating synthetic MRI/CT images, hospitals can **reduce patient burden** and **enhance diagnostic reach**.

---

## 🧪 Methodology

### 1. 📦 Dataset & Preprocessing
- **Source:** CT/MRI data from 30 subjects (9 randomly selected for this project)
- **Annotation:** Done manually using [Labelme](https://github.com/wkentaro/labelme)
- **Segmentation Format:** JSON to binary masks using DeepLabV3+

### 2. 🎯 Image Segmentation
- **Model Used:** DeepLabV3+
- **Purpose:** Extract body from raw images for cleaner GAN input
- **Result:** High training/validation accuracy, but test set overfit — segmentation output was masked manually for GAN input

### 3. 🔁 Image Translation
Two GAN models were implemented and compared:

#### ✅ **StarGAN**
- One model handles both CT → MRI and MRI → CT
- Trained for 10,000 iterations
- PSNR > 30, SSIM ≈ 1.0 → excellent structural retention
- **Selected as final model for deployment**

#### ⚠️ **CycleGAN**
- Requires two separate models (CT→MRI and MRI→CT)
- Lower PSNR (~17) and SSIM (~0.7)
- Still useful for visual comparison

---

## 📊 Results

| Model      | Direction       | PSNR  | SSIM   |
|------------|-----------------|-------|--------|
| **StarGAN**  | CT ↔ MRI         | ~30+  | ~0.99  |
| **CycleGAN** | CT → MRI        | 17.1  | 0.625  |
|              | MRI → CT        | 17.7  | 0.707  |

- Despite **StarGAN’s superior metrics**, CycleGAN occasionally produced more realistic images to the human eye.
- Further tuning of StarGAN could improve visual results.

---

## ⚙️ How to Run This Project

### 🖥️ 1. Environment Setup

Install the required packages with:

```bash
pip install -r requirements.txt
```

You’ll also need:
- Google Colab (for GAN models)
- GPU support (recommended for training efficiency)

### ✂️ 2. Data Annotation & Preprocessing
Step 1: Annotate Images
Use Labelme to label body regions in CT and MRI scans.

Step 2: Convert Annotations
Use the provided scripts to convert annotation files:

```bash
# Convert labelme JSONs to binary masks
python label_convert.py

# Convert Labelme data to VOC format for segmentation
python labelme2voc.py
```

### 🧠 3. Train Image Segmentation Model (DeepLabV3+)
```bash
# Train model on annotated body region masks
python Deeplabv3+_TRAIN.py

# Evaluate model performance (IOU on test set)
python Deeplabv3+_TEST_IOU.py
```
💡 If the test output appears blank or overfitted, consider using raw or masked images directly in GAN training.

### 🔁 4. Image Translation (CT ↔ MRI)
#### Option A: Use StarGAN
Notebook: StarGAN_Final.ipynb (Google Colab recommended)

Steps:
- Mount your Google Drive
- Place CT and MRI images in appropriate domain folders (RaFD-style format)
- Run training and generate output

#### Option B: Use CycleGAN
Two separate notebooks:
- CycleGAN[ct2mri].ipynb (CT → MRI)
- CycleGAN[mri2ct].ipynb (MRI → CT)

Each notebook trains a unidirectional model and saves synthetic images for evaluation.

### 📊 5. Evaluate Results
Use evaluation metrics to compare StarGAN and CycleGAN:
- PSNR: Measures similarity to original image (higher is better)
- SSIM: Measures structural similarity (closer to 1 is better)

Results are printed in each notebook and saved in the results/ folder.


---

## 🧠 Key Technologies

- **Deep Learning Models:**  
  - `DeepLabV3+` (Segmentation)  
  - `StarGAN`, `CycleGAN` (Translation)

- **Frameworks & Tools:**  
  - Python  
  - TensorFlow / PyTorch  
  - OpenCV, NumPy, Matplotlib  
  - Labelme

---

## 💡 Future Work

- Improve generalization of segmentation model (avoid overfitting)
- Integrate automatic CT/MRI classifier for full pipeline automation
- Develop a single executable pipeline:
  - Input: raw CT or MRI
  - Output: translated image (MRI or CT)
- Experiment with newer generative models (e.g., diffusion models)

---

## 📄 License

This project is for academic and research use only.

