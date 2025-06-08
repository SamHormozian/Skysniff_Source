# streamlit_app.py

import streamlit as st
import cv2
import numpy as np
import os
import time
from PIL import Image
from ultralytics import YOLO
import random
import sensorfusion  # custom module
import os
import torch
import numpy as np
from torch.utils.data import DataLoader, random_split
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch.nn as nn
from tqdm import tqdm
from dataset import GasLeakSegDataset  # Ensure dataset.py exists and works
# --- Configuration ---
BATCH_SIZE = 8
NUM_EPOCHS = 5
NUM_CLASSES = 3  # 0=Background, 1=Gas Leak Day, 2=Gas Leak Night
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.mps.is_available else 'cpu')
ENCODER = 'resnet34'
ENCODER_WEIGHTS = 'imagenet'

# MUST BE FIRST STREAMLIT COMMAND
st.set_page_config(page_title="SkySniff Gas Leak Detection GUI", layout="wide")
# COCO Class Labels
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]
# Load YOLOv8 model (small version)
@st.cache_resource
def load_model():
    return YOLO("yolov8s.pt")  # Automatically downloads if not present

model = load_model()
results = []  # Placeholder for results
# Simulated Functions
def getGPSCoordinates():
    base_lat = 37.7749
    base_lon = -122.4194
    lat = round(base_lat + random.uniform(-0.005, 0.005), 6)
    lon = round(base_lon + random.uniform(-0.005, 0.005), 6)
    return f"{lat}, {lon}"

def getGasLeak(results: list):
    # Check if any person is detected in the results
    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls = int(box.cls)
            conf = float(box.conf)
            if COCO_CLASSES[cls] == "person" and conf > 0.5:
                return True
    return False

def open_camera(webcam_id):
    cap = cv2.VideoCapture(webcam_id)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open webcam ID {webcam_id}")
    return cap

# Title
st.markdown("<h1 style='text-align:center;'>SkySniff Gas Leak Detection GUI</h1>", unsafe_allow_html=True)

tabs = st.tabs(["DEBUG mode", "OPERATOR mode", "GAS LEAK DETECTOR"])

# --- DEBUG MODE ---
with tabs[0]:
    st.markdown("<h2 style='color:red;text-align:center;'>Engineer GUI</h2>", unsafe_allow_html=True)

    # Section 1: AI Upscaling Demonstration
    st.header("AI Upscaling Demonstration")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("HD CAM")
        hd_frame_window = st.image([])
    with col2:
        st.subheader("IR IMAGER (Downscaled)")
        ir_frame_window = st.image([])
    with col3:
        st.subheader("FUSED IMAGE")
        fused_frame_window = st.image([])

    # Section 2: AI Detection Demonstration
    st.header("AI Detection Demonstration")
    col4, col5 = st.columns(2)

    with col4:
        st.subheader("GPS Coordinates")
        gps_text = st.empty()
        gas_leak_text = st.empty()

    with col5:
        st.subheader("U-Net Model Output (Bounding Box - Person)")
        unet_frame_window = st.image([])
        # GPS and Gas Leak
        gps_text.write(f"Current GPS: `{getGPSCoordinates()}`")
        gas_leak_text.markdown(f"**Gas Leak Detected:** {'✅ TRUE' if getGasLeak(results) else '❌ FALSE'}")

    # Start button for simulation
    run = st.checkbox("Start Simulation", key="debug_run")

    # Paths
    vis_dir = "fusion_train_data/vi"
    ir_dir = "fusion_train_data/ir"

    # Get list of files
    vis_files = sorted([
        f for f in os.listdir(vis_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])

    idx = 0

    while run and idx < len(vis_files):
        start_time = time.time()

        vis_path = os.path.join(vis_dir, vis_files[idx])
        ir_path = os.path.join(ir_dir, vis_files[idx])

        if not os.path.exists(ir_path):
            st.warning(f"IR image missing for {vis_files[idx]}")
            idx += 1
            continue

        # Load VIS image
        vis_img = cv2.imread(vis_path)
        if vis_img is None:
            st.warning(f"Could not read image: {vis_path}")
            idx += 1
            continue

        # Resize VIS to 640x480
        vis_resized = cv2.resize(vis_img, (640, 480), interpolation=cv2.INTER_LINEAR)

        # Load IR image and downscale for fusion
        ir_img = cv2.imread(ir_path, cv2.IMREAD_GRAYSCALE)
        if ir_img is None:
            st.warning(f"Could not read IR image: {ir_path}")
            idx += 1
            continue

        # Downscale IR for fusion, then upscale back to 640x480 for display
        ir_downscaled = cv2.resize(ir_img, (256, 192), interpolation=cv2.INTER_AREA)
        ir_display = cv2.resize(ir_downscaled, (640, 480), interpolation=cv2.INTER_NEAREST)
        ir_display_color = cv2.applyColorMap(cv2.normalize(ir_display, None, 0, 255, cv2.NORM_MINMAX), cv2.COLORMAP_HOT)

        # Fuse using imported function
        fused_img = sensorfusion.get_fused_frame(vis_path, ir_path, target_size=(256, 192))

        # Resize fused image to 640x480 if needed
        fused_resized = cv2.resize(fused_img, (640, 480), interpolation=cv2.INTER_LINEAR)

        # Update placeholders
        hd_frame_window.image(cv2.cvtColor(vis_resized, cv2.COLOR_BGR2RGB), caption="HD CAM")
        ir_frame_window.image(cv2.cvtColor(ir_display_color, cv2.COLOR_BGR2RGB), caption="IR IMAGER")
        fused_frame_window.image(cv2.cvtColor(fused_resized, cv2.COLOR_BGR2RGB), caption="FUSED IMAGE")

        # Increment index
        idx = (idx + 1) % len(vis_files)
            # Open HD camera
        try:
            cap0  = open_camera(0)  # Assuming webcam ID 0 for HD camera
            ret0, frame0 = cap0.read()
            if not ret0:
                raise RuntimeError("Failed to read from HD camera")
            frame0 = cv2.resize(frame0, (640, 480), interpolation=cv2.INTER_LINEAR)
        except RuntimeError as e:
            st.warning(str(e))
            cap0 = None

        # U-Net Output using YOLOv8 for "person"
        if cap0 and ret0:
            rgb_frame = cv2.cvtColor(frame0, cv2.COLOR_BGR2RGB)
            results = model(rgb_frame)

            annotated_frame = np.array(rgb_frame)

            for result in results:
                boxes = result.boxes
                for box in boxes:
                    cls = int(box.cls)
                    conf = float(box.conf)
                    if COCO_CLASSES[cls] == "person" and conf > 0.5:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        label = f"Person {conf:.2f}"
                        color = (0, 255, 0)  # Green
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(
                            annotated_frame,
                            label,
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            color,
                            2
                        )

            unet_frame_window.image(annotated_frame)
            gas_leak_text.markdown(f"**Gas Leak Detected:** {'✅ TRUE' if getGasLeak(results) else '❌ FALSE'}")

        # Maintain ~10 FPS
        elapsed = time.time() - start_time
        # time.sleep(max(0.1, 0.1 - elapsed))
        if not run and idx > 0:
            st.write("Simulation stopped.")




# --- OPERATOR MODE ---
with tabs[1]:
    st.header("Operator Mode")
    colA, colB, colC = st.columns(3)

    with colA:
        st.subheader("GPS Coordinates")
        gps_op_text = st.empty()
        gas_op_text = st.empty()

    with colB:
        st.subheader("Fused Sensor Image with Bounding Boxes")
        fused_op_window = st.image([])

    with colC:
        st.subheader("Detected Leak Coordinates")
        leak_coords_table = st.empty()

    # Run checkbox
    run_op = st.checkbox("Start Live Feed", key="operator_run")

    try:
        cap_fused = open_camera(0)  # Assuming webcam ID 2 for fused image
    except:
        st.warning("Fused image source not available")
        cap_fused = None

    while run_op and cap_fused:
        start_time = time.time()

        # Open fused camera
        try:
            ret0, frame0 = cap_fused.read()
            if not ret0:
                raise RuntimeError("Failed to read from HD camera")
            frame0 = cv2.resize(frame0, (640, 480), interpolation=cv2.INTER_LINEAR)
        except RuntimeError as e:
            st.warning(str(e))

        # U-Net Output using YOLOv8 for "person"
        if cap_fused and ret0:
            rgb_frame = cv2.cvtColor(frame0, cv2.COLOR_BGR2RGB)
            results = model(rgb_frame)

            annotated_frame = np.array(rgb_frame)

            for result in results:
                boxes = result.boxes
                for box in boxes:
                    cls = int(box.cls)
                    conf = float(box.conf)
                    if COCO_CLASSES[cls] == "person" and conf > 0.5:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        label = f"Person {conf:.2f}"
                        color = (0, 255, 0)  # Green
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(
                            annotated_frame,
                            label,
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            color,
                            2
                        )

            fused_op_window.image(annotated_frame)

        # Update GPS and coordinates
        gps_op_text.write(f"Current GPS: `{getGPSCoordinates()}`")
        gas_op_text.markdown(f"**Gas Leak Detected:** {'✅ TRUE' if getGasLeak(results) else '❌ FALSE'}")
        leak_coords = [getGPSCoordinates() for _ in range(random.randint(3, 5))]
        leak_coords_table.table(leak_coords)

        # Maintain ~10 FPS
        time.sleep(max(0.01, 0.1 - (time.time() - start_time)))

    st.write("Operator feed stopped.")

# --- GAS LEAK DETECTOR ---
with tabs[2]: 
    st.markdown("<h2 style='color:#0088cc;text-align:center;'>Gas Leak Detector</h2>", unsafe_allow_html=True)

    colA, colB = st.columns(2)

    with colA:
        st.subheader("IR Camera Feed")
        ir_feed_window = st.image([])

    with colB:
        st.subheader("Leak Detection Output")
        leak_feed_window = st.image([])

    run_detector = st.checkbox("Start Gas Leak Detection", key="leak_run")

    # Load model
    @st.cache_resource
    def load_model():
        model = smp.Unet(
            encoder_name=ENCODER,
            encoder_weights=None,  # Not needed for inference
            in_channels=3,
            classes=NUM_CLASSES
        ).to(DEVICE)

        model.load_state_dict(torch.load("unet_gas_leak_segmentation_final.pth", map_location=DEVICE))
        model.eval()
        return model

    model = load_model()

    # Color maps for visualization
    class_colors = np.array([
        [0, 0, 0],           # Background - black
        [0, 255, 255],       # Gas Leak Day - cyan
        [255, 0, 0],         # Gas Leak Night - red
    ], dtype=np.uint8)

    # Start webcam capture
    if run_detector:
        cap = cv2.VideoCapture(1)
        if not cap.isOpened():
            st.error("❌ Could not open webcam (ID 2)")
        else:
            progress_bar = st.progress(0)
            frame_count = 0
            try:
                while run_detector:
                    start_time = time.time()

                    ret, frame = cap.read()
                    if not ret:
                        st.warning("⚠️ Failed to read from webcam.")
                        break

                    # Resize and normalize
                    original_frame = frame.copy()
                    resized_frame = cv2.resize(frame, (480, 320))  # Match model input size
                    rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
                    normalized = A.Normalize()(image=rgb_frame)['image']
                    tensor_img = ToTensorV2()(image=normalized)['image'].unsqueeze(0).to(DEVICE)

                    # Run inference
                    with torch.no_grad():
                        output = model(tensor_img)
                        pred_mask = torch.argmax(output, dim=1).cpu().numpy()[0]

                    # Map mask to colors
                    color_mask = class_colors[pred_mask]

                    # Resize mask to match original frame
                    color_mask_resized = cv2.resize(color_mask, (original_frame.shape[1], original_frame.shape[0]), interpolation=cv2.INTER_NEAREST)

                    # Overlay mask on original frame
                    overlayed = cv2.addWeighted(original_frame, 0.6, color_mask_resized, 0.4, 0)

                    # Display frames
                    ir_feed_window.image(cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB), caption="IR Input")
                    leak_feed_window.image(cv2.cvtColor(overlayed, cv2.COLOR_BGR2RGB), caption="Detected Leaks")

                    # Update progress bar
                    frame_count = (frame_count + 1) % 100
                    progress_bar.progress((frame_count + 1) % 100)

                    # Maintain ~10 FPS
                    elapsed = time.time() - start_time
                    time.sleep(max(0.01, 0.1 - elapsed))

            finally:
                cap.release()
                progress_bar.empty()
                st.write("✅ Gas leak detection stopped.")


# Footer
st.markdown("<p style='text-align:center;'>(c) 2025 Recombinant Solutions Incorporated</p>", unsafe_allow_html=True)