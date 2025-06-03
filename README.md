# SkySniff Gas Leak Drone
Welcome to the Basketball Smart Referee project! This README will guide you through the project's purpose, implementation details, usage, and show the results of our method.

Team:

- Russ Sobti
- Sam Hormozian 
- Eric Vu

## Abstract
Gas leaks are a large driver of unexpected greenhouse gas emissions and can pose major safety risks in the industrial and utility sectors. Volatile Organic Compounds (VOCs) such as Carbon Monoxide are often imperceptible in low concentrations, and thus problems are hard to notice until they reach criticality. Aerial OGI systems are beginning to emerge for the Oil and Gas sectors, how-ever the six-figure cost of an aerial OGI system in addition to the large size of currently-fielded drones from ArcSky and similar lead to safety and fieldability concerns for smaller operations. SkySniff offers a sub-$1000 platform that uses a microdrone equipped with a ITAR-free IR camera to provide similar ability to identify gas leaks while enabling indoor operation and increased fieldability. SkySniff is a middle-ground solution between manual inspection with a handheld IR imager such as the FLIR One® and the currently commercially available drones with IR payload options such as the ArcSky X55 with Aerial OGI payload. We successfully demonstrate the ability to 1) automatically identify gas leaks from IR spectrum data, 2) the ability to integrate IR data into visual camera data, and 3) the integration of both models and systems into an affordable drone platform using off-the-shelf analog 5.8Ghz video telemetry. We did this through showing the system working with real-time streamed data in an operable format.

## Setup/Installation

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

## How to Run

### `dataset.py` — Dataset and Annotation Module

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

### U-Net Trainer
  - Trains the U-net model on the data prepared by `dataset.py` and returns a Pytorch model file for consumption by gui.py

### Operator GUI
  - Run the Operator GUI to connect the models with the telemetry. See below for the User Guide. 

---

## Hardware Setup

Before launching the application, ensure the following hardware components are properly configured:

### Required Components

- Two **5.8GHz Video Transmitter/Receiver Modules**
- HD Camera (for visible light feed) + 5.8GHz VTX
- IR Camera (for thermal imaging) + 5.8GHz VTX
- Drone or mounting system for cameras + 3.3V power for VTXes)
- 2 USB ports available for receiver devices

### Pairing Instructions

1. **Plug in both 5.8GHz video receivers** into available USB ports on your computer/device.
2. Make sure the receivers are set to the **correct frequency channel for their corresponding VTX** using buttons or autopair mode. VTXes can be configured to select frequencies in Betaflight.
4. Power on both camera modules (HD and IR).
5. Verify that the signal is being received by checking for stable video output from both devices.

> 📌 **Important:** The receivers must be successfully paired and recognized by the system **before starting the application**, otherwise the webcam feeds will fail to initialize or just show Blue Screen/Static.

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
