import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color
from skimage.transform import resize
import os
from urllib.request import urlopen
from io import BytesIO
from PIL import Image
import seaborn as sns
from matplotlib.gridspec import GridSpec
import numpy as np
from skimage import color
from skimage.transform import resize
from urllib.request import urlopen
from io import BytesIO
from PIL import Image
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage import feature
from scipy import fftpack
import pandas as pd
from matplotlib.colors import LogNorm
import seaborn as sns
#
#  Reusing the custom convolution function from Day 8
def custom_convolution_2d(image, kernel, padding='same', stride=1):
    # Determine padding size based on the padding type
    if padding == 'same':
        pad_height = kernel.shape[0] // 2
        pad_width = kernel.shape[1] // 2
    elif padding == 'valid':
        pad_height = 0
        pad_width = 0
    else:
        raise ValueError("Unsupported padding type")
    
    # Pad the image
    if len(image.shape) == 3:  # RGB image
        padded_image = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width), (0, 0)), mode='constant')
    else:  # Grayscale image
        padded_image = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant')
    
    # Determine output dimensions
    output_height = (image.shape[0] - kernel.shape[0] + 2 * pad_height) // stride + 1
    output_width = (image.shape[1] - kernel.shape[1] + 2 * pad_width) // stride + 1
    output = np.zeros((output_height, output_width, *image.shape[2:])) if len(image.shape) == 3 else np.zeros((output_height, output_width))
    
    # Perform convolution
    for y in range(output_height):
        for x in range(output_width):
            y_start = y * stride
            x_start = x * stride
            if len(image.shape) == 3:  # RGB image
                for c in range(image.shape[2]):
                    output[y, x, c] = (kernel * padded_image[y_start:y_start + kernel.shape[0], x_start:x_start + kernel.shape[1], c]).sum()
            else:  # Grayscale image
                output[y, x] = (kernel * padded_image[y_start:y_start + kernel.shape[0], x_start:x_start + kernel.shape[1]]).sum()

    return output

# Function to visualize kernels
def visualize_kernel(kernel, title):
    plt.figure(figsize=(3, 3))
    sns.heatmap(kernel, annot=True, cmap='viridis', cbar=False, square=True, fmt=".2f")
    plt.title(title)
    plt.tight_layout()
    plt.show()

# Function to calculate image quality metrics
def calculate_metrics(original, processed):
    # Mean Squared Error
    mse = np.mean((original - processed) ** 2)
    
    # Peak Signal-to-Noise Ratio
    if mse == 0:
        psnr = float('inf')
    else:
        max_pixel = 1.0
        psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    
    # Structural content - ratio of original to processed signal power
    # Higher values indicate more structural preservation
    orig_power = np.sum(original ** 2)
    proc_power = np.sum(processed ** 2)
    sc = orig_power / proc_power if proc_power > 0 else float('inf')
    
    # Average gradient magnitude (edge content)
    dx = np.diff(processed, axis=1)
    dy = np.diff(processed, axis=0)
    
    # Pad to maintain size
    dx_padded = np.pad(dx, ((0, 0), (0, 1)), mode='constant')
    dy_padded = np.pad(dy, ((0, 1), (0, 0)), mode='constant')
    
    grad_mag = np.sqrt(dx_padded**2 + dy_padded**2)
    avg_grad = np.mean(grad_mag)
    
    return {
        'MSE': mse,
        'PSNR': psnr,
        'Structural Content': sc,
        'Average Gradient': avg_grad
    }

# Function to display images and metrics
def display_results(original, processed_images, titles, roi=None):
    n_images = len(processed_images) + 1
    
    if roi is None:
        # Only full images
        fig, axes = plt.subplots(1, n_images, figsize=(n_images*4, 5))
        
        # Original image
        axes[0].imshow(original, cmap='gray')
        axes[0].set_title('Original')
        axes[0].set_xlabel('Width (pixels)')
        axes[0].set_ylabel('Height (pixels)')
        
        # Processed images
        for i, (img, title) in enumerate(zip(processed_images, titles)):
            axes[i+1].imshow(img, cmap='gray')
            axes[i+1].set_title(title)
            axes[i+1].set_xlabel('Width (pixels)')
            axes[i+1].set_ylabel('Height (pixels)')
    else:
        # Full images and ROI
        fig = plt.figure(figsize=(n_images*4, 10))
        gs = GridSpec(2, n_images)
        
        # Original full image
        ax_orig_full = fig.add_subplot(gs[0, 0])
        ax_orig_full.imshow(original, cmap='gray')
        ax_orig_full.set_title('Original')
        ax_orig_full.set_xlabel('Width (pixels)')
        ax_orig_full.set_ylabel('Height (pixels)')
        
        # Draw ROI rectangle on original
        y1, y2, x1, x2 = roi
        rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, edgecolor='r', facecolor='none', linewidth=2)
        ax_orig_full.add_patch(rect)
        
        # Processed full images
        for i, (img, title) in enumerate(zip(processed_images, titles)):
            ax = fig.add_subplot(gs[0, i+1])
            ax.imshow(img, cmap='gray')
            ax.set_title(title)
            ax.set_xlabel('Width (pixels)')
            ax.set_ylabel('Height (pixels)')
            
            # Add ROI rectangle
            rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, edgecolor='r', facecolor='none', linewidth=2)
            ax.add_patch(rect)
        
        # Original ROI
        ax_orig_roi = fig.add_subplot(gs[1, 0])
        ax_orig_roi.imshow(original[y1:y2, x1:x2], cmap='gray')
        ax_orig_roi.set_title('Original (ROI)')
        ax_orig_roi.set_xlabel('Width (pixels)')
        ax_orig_roi.set_ylabel('Height (pixels)')
        
        # Processed ROIs
        for i, (img, title) in enumerate(zip(processed_images, titles)):
            ax = fig.add_subplot(gs[1, i+1])
            ax.imshow(img[y1:y2, x1:x2], cmap='gray')
            ax.set_title(f'{title} (ROI)')
            ax.set_xlabel('Width (pixels)')
            ax.set_ylabel('Height (pixels)')
    
    plt.tight_layout()
    plt.show()


def calculate_edge_preservation(img1, img2):
    """
    Calculate edge preservation ratio between original and filtered images.
    Adapted from: "Image Quality Assessment: From Error Visibility to Structural Similarity"
    """
    # Detect edges in both images using Canny edge detector
    edges1 = feature.canny(img1, sigma=1.0)
    edges2 = feature.canny(img2, sigma=1.0)
    
    # Calculate the ratio of matching edge pixels
    matching_edges = np.sum(np.logical_and(edges1, edges2))
    total_edges_original = np.sum(edges1)
    
    # Avoid division by zero
    if total_edges_original == 0:
        return 0
    
    # Return the edge preservation ratio (higher is better)
    return matching_edges / total_edges_original

def calculate_detail_variance_ratio(img1, img2):
    """
    Calculate the ratio of local variance in filtered image compared to original.
    A measure of detail preservation (values close to 1 are best).
    """
    # Calculate local variance using a 5x5 window
    from scipy.ndimage import uniform_filter
    
    # Mean filter
    mean1 = uniform_filter(img1, size=5)
    mean2 = uniform_filter(img2, size=5)
    
    # Squared mean filter
    mean1_sq = uniform_filter(img1**2, size=5)
    mean2_sq = uniform_filter(img2**2, size=5)
    
    # Variance = E[X^2] - E[X]^2
    var1 = mean1_sq - mean1**2
    var2 = mean2_sq - mean2**2
    
    # Avoid division by zero
    var1[var1 < 1e-10] = 1e-10
    
    # Calculate ratio (avoid extreme values)
    ratio = var2 / var1
    ratio = np.clip(ratio, 0.01, 100)
    
    # Return mean ratio (closer to 1 is better)
    return np.mean(ratio)

def calculate_contrast_enhancement(img1, img2):
    """
    Measure the improvement in local contrast after filtering.
    For historical photos, enhanced contrast is often desired.
    Returns a positive value if contrast is enhanced (higher is better).
    """
    # Calculate local contrast using standard deviation in small patches
    from scipy.ndimage import uniform_filter
    
    def local_contrast(img, window_size=5):
        # Mean of squared values
        mean_sq = uniform_filter(img**2, size=window_size)
        # Square of means
        sq_mean = uniform_filter(img, size=window_size)**2
        # Local standard deviation
        return np.sqrt(np.maximum(mean_sq - sq_mean, 0))
    
    # Get local contrast for both images
    contrast1 = local_contrast(img1)
    contrast2 = local_contrast(img2)
    
    # Calculate average contrast ratio (>1 means enhancement)
    contrast_ratio = np.mean(contrast2) / np.mean(contrast1)
    
    return contrast_ratio
