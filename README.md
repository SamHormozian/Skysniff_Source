# SkySniff Gas Leak Drone

## Setup

### Create Virtual Environment:

```bash
cd /path/to/your/project
python3 -m venv .env
```

### Activate It

```bash
source .env/bin/activate
```

### Install Dependencies

```bash
pip install --upgrade pip
pip install torch torchvision pycocotools opencv-python tensorboard streamlit streamlit-autorefresh ultralytics numpy pillow scikit-image
```

> 💡 **Note**: The additional packages (`streamlit`, `ultralytics`, etc.) are required for running and developing the GUI.

### Deactivate After Finished

```bash
deactivate
```

To clean up:

```bash
rm -rf .env
```

---

## Pipeline

### `dataset.py` — Dataset Module

- **Path sanity checks**

  As soon as the module loads, it verifies that both the `dataset/images` directory and the `dataset/annotations.json` file exist, catching path errors early.
- **COCO annotation loading & class mapping**

  Uses the COCO API to read the JSON file, extracts category entries, normalizes names (lowercase + trim), and builds two dictionaries:

  - `name2label`: maps `"gas leak day"` → `1`, `"gas leak night"` → `2`
  - `name2catid`: maps normalized names back to original COCO category IDs for filtering
- **Image indexing**

  Queries COCO using those category IDs to collect all image IDs containing gas leak annotations. This list becomes the dataset universe.
- **Sample retrieval (`__getitem__`)**

  For each index:

  - Loads RGB image into a NumPy array → converts to `[3×H×W]` float tensor.
  - Builds a single-channel mask initialized with zeros.
  - Rasterizes polygon annotations from relevant classes and stamps in their labels (1 or 2).
  - Converts mask to `[H×W]` long tensor.
  - Applies user-defined transforms (e.g., normalization, random flips) to both image and mask.
- **Smoke-test entry point**

  Running `python dataset.py` prints the dataset length and shapes/dtypes of one `(image_tensor, mask_tensor)` pair to verify setup before training.

---

## Hardware Setup

Before launching the application, ensure the following hardware components are properly configured:

### Required Components

- Two **5.8GHz Video Transmitter/Receiver Modules**
- HD Camera (for visible light feed)
- IR Camera (for thermal imaging)
- Drone or mounting system for cameras
- USB ports available for receiver devices

### Pairing Instructions

1. **Plug in both 5.8GHz video receivers** into available USB ports on your computer/device.
2. Make sure the receivers are set to the **same frequency channel** using dip switches or auto-pairing method provided by the manufacturer.
3. Power on both camera modules (HD and IR).
4. Verify that the signal is being received by checking for stable video output from both devices.

> 📌 **Important:** The receivers must be successfully paired and recognized by the system **before starting the application**, otherwise the webcam feeds will fail to initialize.

---

## GUI (SkySniff Gas Leak Detection GUI)

A real-time visualization interface built with **Streamlit** to support both engineering debugging and field operator use cases.

### Features

| Mode                    | Description                                                                                                     |
| ----------------------- | --------------------------------------------------------------------------------------------------------------- |
| **DEBUG mode**    | Shows VIS, IR (downscaled to 256x192), and fused images side-by-side at 10 FPS from a pre-recorded dataset loop |
| **OPERATOR mode** | Live webcam feed showing fused sensor output, GPS coordinates, and gas leak detection status                    |

### GUI Components

- **Live Webcam Fusion**

  - Supports multiple camera inputs (HD CAM, IR IMAGER, FUSED IMAGE)
  - Uses `sensorfusion.py` for real-time overlay logic
- **Bounding Box Detection**

  - Uses YOLOv8s model to detect **"person"** class only
  - Draws green bounding boxes around people in U-Net Output section
- **Sensor Fusion Engine**

  - Integrated via `sensorfusion.py`
  - IR image is downsampled before fusion, then upscaled to match HD CAM resolution (640x480)

### How to Run

Make sure you're in the activated virtual environment:

```bash
streamlit run gui.py
```

> 📁 Ensure that your webcam devices are available at indices `0`, `1`, and `2` for full functionality.

---

## Optional: Sensor Fusion System (`sensorfusion.py`)

Handles:

- IR colormap application
- Image resizing/downscaling
- Real-time image fusion using transparency overlay
- Batch processing with downscaling
- SSIM/PSNR comparison between outputs

---
