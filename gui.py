import streamlit as st
import cv2
import numpy as np
import time
from PIL import Image
from ultralytics import YOLO
import random

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

# Simulated Functions
def getGPSCoordinates():
    base_lat = 37.7749
    base_lon = -122.4194
    lat = round(base_lat + random.uniform(-0.005, 0.005), 6)
    lon = round(base_lon + random.uniform(-0.005, 0.005), 6)
    return f"{lat}, {lon}"

def getGasLeak():
    return random.choice([0, 1])

def open_camera(webcam_id):
    cap = cv2.VideoCapture(webcam_id)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open webcam ID {webcam_id}")
    return cap

# Title
st.markdown("<h1 style='text-align:center;'>SkySniff Gas Leak Detection GUI</h1>", unsafe_allow_html=True)

tabs = st.tabs(["DEBUG mode", "OPERATOR mode"])

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
        st.subheader("IR IMAGER")
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

    # Start button for webcam stream
    run = st.checkbox("Start Live Feed", key="debug_run")

    # Open cameras
    try:
        cap0 = open_camera(0)
    except:
        st.error("Webcam 0 (HD CAM) failed to open")
        cap0 = None

    try:
        cap1 = open_camera(1)
    except:
        st.warning("Webcam 1 (IR Imager) failed to open")
        cap1 = None

    try:
        cap2 = open_camera(2)
    except:
        st.warning("Webcam 2 (Fused) failed to open")
        cap2 = None

    while run and (cap0 or cap1 or cap2):
        start_time = time.time()

        # Update HD CAM
        if cap0:
            ret0, frame0 = cap0.read()
            if ret0:
                hd_frame_window.image(cv2.cvtColor(frame0, cv2.COLOR_BGR2RGB))

        # Update IR IMAGER
        if cap1:
            ret1, frame1 = cap1.read()
            if ret1:
                ir_frame_window.image(cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB))

        # Update FUSED IMAGE
        if cap2:
            ret2, frame2 = cap2.read()
            if ret2:
                fused_frame_window.image(cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB))

        # GPS and Gas Leak
        gps_text.write(f"Current GPS: `{getGPSCoordinates()}`")
        gas_leak_text.markdown(f"### Gas Leak Detection: **{'TRUE' if getGasLeak() else 'FALSE'}**")

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

        # Maintain ~10 FPS
        time.sleep(max(0.01, 0.1 - (time.time() - start_time)))

    st.write("Stopped live feed.")

# --- OPERATOR MODE ---
with tabs[1]:
    st.header("Operator Mode")
    colA, colB, colC = st.columns(3)

    with colA:
        st.subheader("GPS Coordinates")
        gps_op_text = st.empty()
        gas_op_text = st.empty()

    with colB:
        st.subheader("Fused Sensor Image")
        fused_op_window = st.image([])

    with colC:
        st.subheader("Detected Leak Coordinates")
        leak_coords_table = st.empty()

    # Run checkbox
    run_op = st.checkbox("Start Live Feed", key="operator_run")

    # Open fused camera
    try:
        cap_fused = open_camera(2)
    except:
        st.warning("Fused image source not available")
        cap_fused = None

    while run_op and cap_fused:
        start_time = time.time()

        # Update fused image
        ret, frame = cap_fused.read()
        if ret:
            fused_op_window.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Update GPS and coordinates
        gps_op_text.write(f"Current GPS: `{getGPSCoordinates()}`")
        gas_op_text.markdown(f"**Gas Leak Detected:** {'✅ TRUE' if getGasLeak() else '❌ FALSE'}")
        leak_coords = [getGPSCoordinates() for _ in range(random.randint(3, 5))]
        leak_coords_table.table(leak_coords)

        # Maintain ~10 FPS
        time.sleep(max(0.01, 0.1 - (time.time() - start_time)))

    st.write("Operator feed stopped.")

# Footer
st.markdown("<p style='text-align:center;'>(c) 2025 Recombinant Solutions Incorporated</p>", unsafe_allow_html=True)

