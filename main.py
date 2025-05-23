import os
import cv2
import numpy as np
import pydicom
from matplotlib import pyplot as plt
from skimage import exposure, filters, util, restoration
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
from glob import glob

#  Utility Functions 
def read_dcm_image(path):
    ds = pydicom.dcmread(path)
    img = ds.pixel_array.astype(np.float32)
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return img  # return only the image


import pydicom

def read_rvg_image(path):
    try:
        ds = pydicom.dcmread(path)
        img = ds.pixel_array.astype(np.float32)
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        img = img.astype(np.uint8)
        return img
    except Exception as e:
        print(f"Error reading RVG file {path}: {e}")
        return None



def compute_image_metrics(img):
    brightness = np.mean(img)
    contrast = np.std(img)
    sharpness = cv2.Laplacian(img, cv2.CV_64F).var()
    noise = estimate_noise(img)
    return brightness, contrast, sharpness, noise

def estimate_noise(img):
    H, W = img.shape
    M = cv2.blur(img, (3, 3))
    noise = np.sqrt(np.mean((img - M)**2))
    return noise

def visualize_comparison(original, static, adaptive, title="", metrics=None):
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    
    # Display images
    axs[0].imshow(original, cmap='gray')
    axs[0].set_title("Original")
    
    axs[1].imshow(static, cmap='gray')
    axs[1].set_title(f"Static\nPSNR: {metrics['PSNR'][0]:.2f}, SSIM: {metrics['SSIM'][0]:.3f}\nEdges: {metrics['Edge Count'][0]}")
    
    axs[2].imshow(adaptive, cmap='gray')
    axs[2].set_title(f"Adaptive\nPSNR: {metrics['PSNR'][1]:.2f}, SSIM: {metrics['SSIM'][1]:.3f}\nEdges: {metrics['Edge Count'][1]}")

    for ax in axs:
        ax.axis('off')

    plt.suptitle(f"{title}\nOriginal Edges: {metrics['Edge Count'][2]}", fontsize=14, y=1.05)
    plt.tight_layout()
    plt.show()


# ========== Preprocessing Pipelines ==========
def static_preprocessing(img):
    img_eq = cv2.equalizeHist(img)
    sharpened = cv2.filter2D(img_eq, -1, np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]]))
    denoised = cv2.fastNlMeansDenoising(sharpened, None, 10, 7, 21)
    return denoised

def compute_michelson_contrast(img):
    return (np.max(img) - np.min(img)) / (np.max(img) + np.min(img) + 1e-5)

def compute_sharpness(img):
    return np.mean(cv2.Laplacian(img, cv2.CV_64F)**2)

def estimate_noise(img):
    return np.std(cv2.medianBlur(img, 3) - img)

def adaptive_preprocessing(img):
    original = img.copy()

    # Normalize to 8-bit
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Metrics
    contrast = compute_michelson_contrast(img)
    sharpness = compute_sharpness(img)
    noise_level = estimate_noise(img)

    # 1. Contrast Enhancement (CLAHE)
    if contrast < 0.2:
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
    elif contrast < 0.5:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    else:
        clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8,8))  # Mild

    img = clahe.apply(img)

    # 2. Denoising
    if noise_level > 20:
        img = cv2.fastNlMeansDenoising(img, None, h=15)
    elif noise_level > 10:
        img = cv2.bilateralFilter(img, 9, 75, 75)

    # 3. Sharpening
    if sharpness < 100:
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        img = cv2.filter2D(img, -1, kernel)
    elif sharpness < 300:
        gaussian = cv2.GaussianBlur(img, (9, 9), 10.0)
        img = cv2.addWeighted(img, 1.5, gaussian, -0.5, 0)

    return img

def evaluate_pipeline(original, static, adaptive):
    results = {}

    # PSNR & SSIM
    psnr_static = psnr(original, static, data_range=255)
    psnr_adaptive = psnr(original, adaptive, data_range=255)

    ssim_static = ssim(original, static, data_range=255)
    ssim_adaptive = ssim(original, adaptive, data_range=255)

    # Edge count comparison
    edge_orig = cv2.Canny(original, 100, 200)
    edge_static = cv2.Canny(static, 100, 200)
    edge_adaptive = cv2.Canny(adaptive, 100, 200)

    edge_count_orig = np.sum(edge_orig > 0)
    edge_count_static = np.sum(edge_static > 0)
    edge_count_adaptive = np.sum(edge_adaptive > 0)

    results['PSNR'] = (psnr_static, psnr_adaptive)
    results['SSIM'] = (ssim_static, ssim_adaptive)
    results['Edge Count'] = (edge_count_static, edge_count_adaptive, edge_count_orig)

    print("\n--- Evaluation Metrics ---")
    print(f"PSNR: Static = {psnr_static:.2f}, Adaptive = {psnr_adaptive:.2f}")
    print(f"SSIM: Static = {ssim_static:.4f}, Adaptive = {ssim_adaptive:.4f}")
    print(f"Edge Count: Original = {edge_count_orig}, Static = {edge_count_static}, Adaptive = {edge_count_adaptive}")
    print("--------------------------\n")

    return results

    

# ========== Main Processing ==========
def process_directory(directory, read_function, tag):
    print(f"Processing {tag.upper()} files...")
    for filename in os.listdir(directory):
        if not filename.lower().endswith(('.dcm', '.rvg')):
            continue
        path = os.path.join(directory, filename)
        img = read_function(path)

        if img is None or img.size == 0:
            print(f"Skipping invalid image: {filename}")
            continue

        print(f"Processing {filename}...")
        static = static_preprocessing(img)
        adaptive = adaptive_preprocessing(img)

        # Evaluate quality metrics
        metrics = evaluate_pipeline(img, static, adaptive)

        # Show visual comparison with overlaid metrics
        visualize_comparison(img, static, adaptive, f"{tag.upper()}: {filename}", metrics)



if __name__ == '__main__':
    dcm_dir = 'Data Science Images/dcm'
    rvg_dir = 'Data Science Images/rvg'

    print("Processing DICOM files...")
    process_directory(dcm_dir, read_dcm_image, 'dcm')

    print("Processing RVG files...")
    process_directory(rvg_dir, read_rvg_image, 'rvg')
