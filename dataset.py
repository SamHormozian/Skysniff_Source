import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from pycocotools.coco import COCO
from torchvision.io import read_image

# Module-level paths
IMAGES_DIR = './classification_dataset/'
ANNOTATIONS_JSON = 'classification_dataset/annotations.json'

# Sanity checks for dataset paths
def _check_paths():
    assert os.path.isdir(IMAGES_DIR), f"Images directory not found: {IMAGES_DIR}"
    assert os.path.isfile(ANNOTATIONS_JSON), f"Annotations JSON not found: {ANNOTATIONS_JSON}"

_check_paths()

class GasLeakSegDataset(Dataset):
    """
    A PyTorch Dataset for U-Net-style segmentation of gas leaks.
    Returns (image_tensor, mask_tensor) where mask_tensor
    has values 0=background, 1=Gas Leak Day, 2=Gas Leak Night.
    """
    def __init__(self, transforms=None):
        self.images_dir = IMAGES_DIR
        self.transforms = transforms
        # Load COCO annotations
        self.coco = COCO(ANNOTATIONS_JSON)
        # Build mappings: normalized category name -> label and COCO category ID
        cats = self.coco.loadCats(self.coco.getCatIds())
        self.name2label = {}
        self.name2catid = {}
        for idx, cat in enumerate(sorted(cats, key=lambda x: x['id'])):
            name = cat['name'].strip().lower()
            self.name2label[name] = idx + 1
            self.name2catid[name] = cat['id']
        print(f"Segmentation labels: {self.name2label}")
        # Gather image IDs containing these categories
        img_ids = set()
        for cat_id in self.name2catid.values():
            img_ids |= set(self.coco.getImgIds(catIds=[cat_id]))
        self.image_ids = sorted(img_ids)
        assert self.image_ids, "No images found for segmentation classes"

    def __len__(self):
        return len(self.image_ids)

    # def __getitem__(self, idx):
    #     # Load image info
    #     img_id = self.image_ids[idx]
    #     info = self.coco.imgs[img_id]
    #     img_path = os.path.join(self.images_dir, info['file_name'])
    #     img = Image.open(img_path).convert('RGB')
    #     # Initialize segmentation mask
    #     height, width = info['height'], info['width']
    #     seg_mask = np.zeros((height, width), dtype=np.uint8)
    #     # Load and rasterize annotations
    #     ann_ids = self.coco.getAnnIds(imgIds=[img_id], catIds=list(self.name2catid.values()))
    #     anns = self.coco.loadAnns(ann_ids)
    #     for ann in anns:
    #         raw_name = self.coco.cats[ann['category_id']]['name']
    #         name = raw_name.strip().lower()
    #         if name not in self.name2label:
    #             continue
    #         label = self.name2label[name]
    #         mask = self.coco.annToMask(ann)  # H×W binary
    #         seg_mask[mask > 0] = label
    #     # Convert to torch tensors
    #     image_tensor = torch.as_tensor(np.array(img), dtype=torch.float32).permute(2,0,1) / 255.0
    #     mask_tensor  = torch.as_tensor(seg_mask, dtype=torch.long)
    #     # Apply transforms if provided (should handle both image & mask)
    #     if self.transforms:
    #         image_tensor, mask_tensor = self.transforms(image_tensor, mask_tensor)
    #     return image_tensor, mask_tensor
    def __getitem__(self, idx):
        # 1. Find image & annotation info
        img_id   = self.image_ids[idx]
        info     = self.coco.imgs[img_id]
        img_path = os.path.join(self.images_dir, info['file_name'])

        # 2. Load image all at once (no lingering file handles)
        #    Returns a 3×H×W uint8 Tensor
        image_tensor = read_image(img_path).float() / 255.0

        # 3. Prepare empty segmentation mask
        height, width = info['height'], info['width']
        seg_mask = np.zeros((height, width), dtype=np.uint8)

        # 4. Rasterize each annotation into the mask
        ann_ids = self.coco.getAnnIds(
            imgIds=[img_id],
            catIds=list(self.name2catid.values())
        )
        anns = self.coco.loadAnns(ann_ids)
        for ann in anns:
            raw_name = self.coco.cats[ann['category_id']]['name']
            name     = raw_name.strip().lower()
            if name not in self.name2label:
                continue
            label = self.name2label[name]
            mask  = self.coco.annToMask(ann)  # H×W binary array
            seg_mask[mask > 0] = label

        # 5. Convert mask to a LongTensor
        mask_tensor = torch.as_tensor(seg_mask, dtype=torch.long)

        # 6. Apply any paired transforms (image, mask)
        if self.transforms:
            image_tensor, mask_tensor = self.transforms(image_tensor, mask_tensor)

        return image_tensor, mask_tensor
# Smoke test
if __name__ == '__main__':
    ds = GasLeakSegDataset(transforms=None)
    print(f"Dataset length: {len(ds)}")
    img, msk = ds[0]
    print(f"Image tensor shape: {img.shape}, dtype: {img.dtype}")
    print(f"Mask tensor shape:  {msk.shape}, dtype: {msk.dtype}")
