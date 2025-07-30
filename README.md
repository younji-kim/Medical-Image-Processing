# CT ↔ MRI Image Bi-Directional Conversion using Deep Learning

This project proposes a medical imaging solution to convert CT and MRI images bi-directionally using generative models. CT and MRI scans each have benefits and limitations depending on patient conditions, cost, and imaging needs. By enabling image translation between the two modalities, this project aims to improve accessibility, reduce diagnostic limitations, and streamline imaging workflows — particularly for patients who cannot undergo both scans.

> Developed as a part of the *Medical Image Processing* course at Ewha Womans University.

![Slide1](https://github.com/user-attachments/assets/c07d1af4-1b71-438a-acc8-1c9f9ae21316)

---

## ⚙️ How to Run This Project

### 🖥️ 1. Environment Setup

Install the required packages by running:

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
Notebook: `StarGAN.ipynb` (Google Colab recommended)  
Based on the official [StarGAN repository](https://github.com/yunjey/stargan) by Yunjey Choi.

Steps:
- Mount your Google Drive
- Place CT and MRI images in appropriate domain folders (RaFD-style format)
- Run training and generate output

#### Option B: Use CycleGAN
Two separate notebooks:
- `CycleGAN[ct2mri].ipynb` (CT → MRI)
- `CycleGAN[mri2ct].ipynb` (MRI → CT)  
Based on the official [CycleGAN repository](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) by Jun-Yan Zhu et al.

Each notebook trains a unidirectional model and saves synthetic images for evaluation.

### 📊 5. Evaluate Results
Use evaluation metrics to compare StarGAN and CycleGAN:
- PSNR: Measures similarity to original image (higher is better)
- SSIM: Measures structural similarity (closer to 1 is better)

Results are printed in each notebook and saved in the results/ folder.

---

## 📄 License

This project is for academic and research use only.

