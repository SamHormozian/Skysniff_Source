# Gas_Leak_Drone

## Create Virtual Env:

```
cd /path/to/your/project
python3 -m venv .env

```

## Activate it

```

source .env/bin/activate

```

## Install Dependencies

```
pip install --upgrade pip
pip install torch torchvision pycocotools opencv-python tensorboard

```

## Deactivate after finished

```

deactivate
rm -rf .env

```

# Pipeline

## Dataset.py:
- Path sanity checks

    - As soon as the module loads, it verifies that both the dataset/images directory and the dataset/annotations.json file actually exist, to path errors immediately.

- COCO annotation loading & class mapping

    - It uses the COCO API to read JSON file, pulls out all category entries, normalizes their names (lower-cases and trims) and builds two dictionaries:

    - name2label: maps "gas leak day"→1, "gas leak night"→2 for your mask values

    - name2catid: maps normalized names back to the original COCO category IDs for filtering

 - Image indexing

    - By querying COCO with those two category IDs, it collects the set of all image IDs that contain at least one “day” or “night” gas-leak annotation.

    - That list becomes the universe of samples exposed by the dataset.

- Sample retrieval (__getitem__)

    - For a given index it:

        - Loads the RGB image from disk into a NumPy array, converts to a [3×H×W] float tensor.

        - Initializes a single-channel mask of zeros (H×W), then for each polygon annotation of the two classes rasterizes it into a binary mask and stamps in its class label (1 or 2).

        - Converts that mask into a [H×W] long tensor.

        - Applies any user-supplied transforms to both image and mask (e.g. random flips, normalization).

- Smoke-test entry point

    - Running python dataset.py prints out the dataset length and the shapes/dtypes of one (image_tensor, mask_tensor) pair to confirm everything’s wired up before training.







