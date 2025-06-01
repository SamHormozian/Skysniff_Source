import cv2
import numpy as np
import os
from skimage.metrics import structural_similarity as ssim

# -------------------------------
# Helper Functions
# -------------------------------

def apply_colormap_ir(ir_img):
    ir_normalized = cv2.normalize(ir_img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    color_mapped = cv2.applyColorMap(ir_normalized, cv2.COLORMAP_HOT)
    return color_mapped

def upscale_image(ir_img, target_size):
    resized_ir = cv2.resize(ir_img, target_size, interpolation=cv2.INTER_LINEAR)
    return resized_ir

def downscale_image(img, target_size):
    return cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)

def fuse_images_with_overlay(vis_path, ir_path, output_path, target_size=None, opacity=0.5):
    vis = cv2.imread(vis_path, cv2.IMREAD_GRAYSCALE)
    ir = cv2.imread(ir_path, cv2.IMREAD_GRAYSCALE)

    if vis is None or ir is None:
        print(f"Error loading images: {vis_path}, {ir_path}")
        return

    # Optionally resize IR before processing
    if target_size:
        ir = downscale_image(ir, target_size)

    # Resize IR back to original VIS size
    target_size_full = (vis.shape[1], vis.shape[0])
    resized_color_ir = upscale_image(apply_colormap_ir(ir), target_size_full)

    # Convert visible image to BGR for overlay
    vis_bgr = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

    # Overlay with transparency
    fused_image = cv2.addWeighted(vis_bgr, 1 - opacity, resized_color_ir, opacity, 0)

    # Save output
    cv2.imwrite(output_path, fused_image)

# -------------------------------
# Batch Processing & Downscaling
# -------------------------------

def batch_fuse_with_downscaled_ir(
    vis_dir, ir_dir, output_dir, resolution=None, opacity=0.5
):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in sorted(os.listdir(vis_dir)):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            vis_path = os.path.join(vis_dir, filename)
            ir_path = os.path.join(ir_dir, filename)
            out_path = os.path.join(output_dir, filename)

            if not os.path.exists(ir_path):
                print(f"IR image not found for {filename}")
                continue

            fuse_images_with_overlay(vis_path, ir_path, out_path, resolution, opacity)

# -------------------------------
# Image Comparison (SSIM + PSNR)
# -------------------------------

def compare_images_folder(golden_dir, test_dir):
    ssim_scores = []
    psnr_scores = []

    for filename in os.listdir(golden_dir):
        golden_path = os.path.join(golden_dir, filename)
        test_path = os.path.join(test_dir, filename)

        if not os.path.exists(test_path):
            print(f"Missing file in test dir: {filename}")
            continue

        img_golden = cv2.imread(golden_path)
        img_test = cv2.imread(test_path)

        if img_golden.shape != img_test.shape:
            print(f"Shape mismatch for {filename}: {img_golden.shape} vs {img_test.shape}")
            continue

        # SSIM
        gray_golden = cv2.cvtColor(img_golden, cv2.COLOR_BGR2GRAY)
        gray_test = cv2.cvtColor(img_test, cv2.COLOR_BGR2GRAY)
        ssim_score = ssim(gray_golden, gray_test, multichannel=False, channel_axis=None)
        ssim_scores.append(ssim_score)

        # PSNR
        mse = np.mean((img_golden - img_test) ** 2)
        if mse == 0:
            psnr = float('inf')
        else:
            psnr = cv2.PSNR(img_golden, img_test)
        psnr_scores.append(psnr)

    avg_ssim = np.mean(ssim_scores) if ssim_scores else 0
    avg_psnr = np.mean(psnr_scores) if psnr_scores else 0

    return {
        'avg_ssim': avg_ssim,
        'avg_psnr': avg_psnr,
        'total_images': len(ssim_scores)
    }
# -------------------------------
# Real-time Fusion for GUI
# -------------------------------
def get_fused_frame(vis_path, ir_path, target_size=(256, 192)):
    """
    Return fused image as a NumPy BGR array.
    Used in real-time simulation like Streamlit GUI.
    """
    temp_dir = "temp_fused_debug"
    os.makedirs(temp_dir, exist_ok=True)

    filename = os.path.basename(vis_path)
    output_path = os.path.join(temp_dir, filename)

    fuse_images_with_overlay(vis_path, ir_path, output_path, target_size=target_size)
    fused_img = cv2.imread(output_path)

    return fused_img
# -------------------------------
# Main Execution
# -------------------------------

if __name__ == "__main__":
    vis_directory = "fusion_train_data/vi"
    ir_directory = "fusion_train_data/ir"

    # Golden run (no downscaling)
    golden_dir = "fused_output_golden"
    print("🚀 Generating golden reference...")
    batch_fuse_with_downscaled_ir(vis_directory, ir_directory, golden_dir, opacity=0.5)

    # Define downscale targets
    downscale_targets = {
        # "360p": (640, 360),
        "256x192": (256, 192),
        # "144p": (256, 144),
        "30x60": (30, 60),
        "16x16": (16, 16),
    }

    results = {}

    # Run tests at different resolutions
    for name, resolution in downscale_targets.items():
        test_dir = f"fused_output_{name}"
        print(f"\n🚀 Testing {name} ({resolution})...")
        batch_fuse_with_downscaled_ir(vis_directory, ir_directory, test_dir, resolution, opacity=0.5)

        print(f"📊 Comparing {test_dir} to golden...")
        metrics = compare_images_folder(golden_dir, test_dir)
        results[name] = metrics

    # Print final comparison
    print("\n📈 Final Results:")
    for name, metric in results.items():
        print(f"{name.upper()}:")
        print(f"  Avg SSIM: {metric['avg_ssim']:.4f}")
        print(f"  Avg PSNR: {metric['avg_psnr']:.2f} dB")
        print(f"  Images compared: {metric['total_images']}")