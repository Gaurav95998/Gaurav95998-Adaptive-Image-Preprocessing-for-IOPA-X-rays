# 🦷 Adaptive Image Preprocessing Pipeline for IOPA X-ray Images

## 📌 Problem Understanding
Dental radiographs, especially intraoral periapical (IOPA) X-rays, are crucial for detecting pathologies such as caries, bone loss, and impacted teeth. However, raw DICOM (`.dcm`) and RVG (`.rvg`) images often suffer from inconsistent quality due to variations in acquisition hardware, exposure settings, and noise. Enhancing these images through preprocessing is essential for both human interpretation and automated diagnostic models.

This project focuses on designing a **dynamic, adaptive preprocessing pipeline** to improve image quality by assessing and responding to characteristics like contrast, sharpness, and noise levels—thereby ensuring consistent image enhancement across heterogeneous input data.

---

## 📂 Dataset Description

### Formats Handled:
- **DICOM (`.dcm`)**: Standard grayscale radiographs.
- **RVG (`.rvg`)**: Simulated as DICOM-like structures (compatible with `pydicom`).

### 📁 Project Structure

```
DS_IN.../
├── Data Science Images/
│   ├── dcm/                  # Folder containing DICOM (.dcm) files
│   └── rvg/                  # Folder containing RVG (.rvg) files
├── Reference_Output_.../     # Folder to save or visualize output images
├── main.py                   # Main script to run preprocessing pipeline
└── requirement.txt           # Python dependencies for the project
```



### Reading Strategy:
- Images loaded using `pydicom`
- Normalized to 8-bit grayscale using OpenCV

---

## 🧪 Methodology

### ✅ Image Quality Metrics
| Metric     | Description                                 |
|------------|---------------------------------------------|
| Brightness | Average pixel intensity                     |
| Contrast   | Standard deviation / Michelson contrast     |
| Sharpness  | Variance of Laplacian                       |
| Noise      | Difference from smoothed image              |

These metrics guide the intensity of enhancement methods like contrast boosting, sharpening, and denoising.

---

### ⚙️ Static Preprocessing (Baseline)
Applies fixed enhancement regardless of image variability:
- Histogram Equalization (`cv2.equalizeHist`)
- Sharpening using a fixed kernel
- Non-local Means Denoising

---

### 🧠 Adaptive Preprocessing Pipeline

Heuristics based on quality metrics:

#### Contrast Enhancement:
- CLAHE (`cv2.createCLAHE`)
- Clip limit adjusted based on Michelson contrast

#### Denoising:
- High noise: `cv2.fastNlMeansDenoising`
- Moderate noise: `cv2.bilateralFilter`

#### Sharpening:
- Low sharpness: High-boost filtering
- Moderate sharpness: Unsharp masking via Gaussian blur

#### Parameters Table:
| Metric     | Range         | Strategy                          |
|------------|---------------|-----------------------------------|
| Contrast   | <0.2, 0.2-0.5 | CLAHE clip limit: 4, 2, 1         |
| Noise      | >20, >10      | NLM, bilateral filter             |
| Sharpness  | <100, <300    | Filter2D, unsharp masking         |

---

## 📊 Results & Evaluation

### 📈 Quantitative Metrics
- **PSNR (Peak Signal-to-Noise Ratio)**
- **SSIM (Structural Similarity Index)**
- **Edge Count** (via Canny edge detection)

#### Sample Output:
![alt text](<Screenshot 2025-05-23 115305-1.png>)


---

### 🖼️ Visual Comparison
3-way comparison per image:
- **Original**
- **Static Enhanced**
- **Adaptive Enhanced**

Adaptive enhancement consistently achieves:
- Higher edge detail
- Improved local contrast
- Better structural integrity

---

## 🧠 Discussion & Future Work

### ⚠️ Challenges Faced
- Pixel depth & orientation inconsistencies in RVG images
- Risk of over-enhancement from strong CLAHE

### 🔍 Potential Improvements
- **Learned Enhancement Models**: Use Autoencoders / U-Nets
- **Metric Prediction**: Use regressors to determine optimal preprocessing params
- **ROI-Aware Processing**: Focus on clinically important regions

### 🩺 Clinical Impact
- Enhances diagnostic accuracy for radiologists
- Improves robustness of AI models for:
  - Caries detection
  - Bone loss grading
  - Pathology classification

---

### ▶️ How to Run the Project

#### 🔧 Install Dependencies & 🚀 Execute the Script

1. Install the required Python packages:

   ```bash
   pip install -r requirement.txt

2. Run the main script:

    ```bash
   python main.py