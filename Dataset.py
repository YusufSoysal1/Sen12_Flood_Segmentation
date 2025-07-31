import os
import json
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import rasterio
import rasterio.features
from shapely.geometry import shape
from rasterio.warp import transform_geom
import albumentations as A


class FloodDataset(Dataset):
    def __init__(self,
                 base_dir=".",
                 use_s2=True,
                 use_s1_labels=True,
                 use_s2_labels=False,
                 target_size=(512, 512)):

        self.base = base_dir
        self.use_s2 = use_s2
        self.use_s1_labels = use_s1_labels
        self.use_s2_labels = use_s2_labels
        self.target_size = target_size

        self.s1_source = os.path.join(base_dir, "sen12floods_s1_source", "sen12floods_s1_source")
        self.s2_source = os.path.join(base_dir, "sen12floods_s2_source", "sen12floods_s2_source")
        self.s1_labels = os.path.join(base_dir, "sen12floods_s1_labels", "sen12floods_s1_labels")
        self.s2_labels = os.path.join(base_dir, "sen12floods_s2_labels", "sen12floods_s2_labels")

        self.s2_bands = ['B02', 'B03', 'B04', 'B08', 'B8A', 'B11']

        self.sample_names = sorted([
            f for f in os.listdir(self.s1_source)
            if os.path.isdir(os.path.join(self.s1_source, f))
        ])

        self.augment = A.Compose([
            A.RandomCrop(height=target_size[0], width=target_size[1]),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
        ])

    def __len__(self):
        return len(self.sample_names)

    def load_band(self, path):
        if not os.path.exists(path):
            return np.zeros(self.target_size, dtype=np.float32)
        img = Image.open(path).convert("L").resize(self.target_size, Image.BILINEAR)
        return np.array(img, dtype=np.float32) / 255.0

    def load_cloud_mask(self, qa60_path):
        if not os.path.exists(qa60_path):
            return np.ones(self.target_size, dtype=np.float32)
        img = Image.open(qa60_path).resize(self.target_size, Image.NEAREST)
        arr = np.array(img, dtype=np.uint16)
        cloud_mask = ((arr & (1 << 10)) == 0).astype(np.float32)  # bit 10 = cloud
        return cloud_mask

    def load_geojson_mask(self, geojson_path, reference_tif):
        with rasterio.open(reference_tif) as src:
            transform = src.transform
            width, height = src.width, src.height
            dst_crs = src.crs

        with open(geojson_path) as f:
            geojson = json.load(f)

        shapes = []

        #  FLOODING=True filtrelemesi
        if geojson.get("type") == "FeatureCollection":
            features = geojson.get("features", [])
        elif geojson.get("type") == "Feature":
            features = [geojson]
        else:
            features = []

        for feat in features:
            props = feat.get("properties", {})
            if str(props.get("FLOODING", "False")).lower() != "true":
                continue
            geom = feat.get("geometry", None)
            if geom is None:
                continue
            geom_proj = transform_geom("EPSG:4326", dst_crs, geom)
            shapes.append((shape(geom_proj), 1))

        if len(shapes) == 0:
            return np.zeros(self.target_size, dtype=np.float32)

        mask = rasterio.features.rasterize(
            shapes=shapes,
            out_shape=(height, width),
            transform=transform,
            fill=0,
            dtype=np.uint8
        )

        mask_img = Image.fromarray(mask).resize(self.target_size, Image.NEAREST)
        return (np.array(mask_img, dtype=np.float32) > 0).astype(np.float32)

    def __getitem__(self, idx):
        sample = self.sample_names[idx]

        #  Sentinel-1
        folder_s1 = os.path.join(self.s1_source, sample)
        vv = self.load_band(os.path.join(folder_s1, "VV.tif"))
        vh = self.load_band(os.path.join(folder_s1, "VH.tif"))
        s1 = np.stack([vv, vh], axis=0)

        #  Sentinel-2
        if self.use_s2:
            folder_s2 = os.path.join(self.s2_source, sample)
            s2 = []
            if os.path.isdir(folder_s2):
                cloud_mask = self.load_cloud_mask(os.path.join(folder_s2, "QA60.tif"))
                for band in self.s2_bands:
                    path_tif = os.path.join(folder_s2, f"{band}.tif")
                    path_png = os.path.join(folder_s2, f"{band}.png")
                    path = path_tif if os.path.exists(path_tif) else path_png
                    b = self.load_band(path) * cloud_mask
                    s2.append(b)
            else:
                s2 = [np.zeros(self.target_size, dtype=np.float32) for _ in self.s2_bands]
            s2 = np.stack(s2, axis=0)
            x = np.concatenate([s1, s2], axis=0)
        else:
            x = s1

        #  Etiket maskesi
        label = np.zeros(self.target_size, dtype=np.float32)
        label_dir = self.s2_labels if self.use_s2_labels else self.s1_labels
        label_sample = sample.replace("s1_source", "s1_labels") if not self.use_s2_labels else sample.replace("s2_source", "s2_labels")
        geojson_path = os.path.join(label_dir, label_sample, "labels.geojson")
        ref_tif = os.path.join(folder_s1, "VV.tif")

        if os.path.exists(geojson_path):
            label = self.load_geojson_mask(geojson_path, ref_tif)

        #  Veri arttırma
        x = np.transpose(x, (1, 2, 0))
        augmented = self.augment(image=x, mask=label)
        x = np.transpose(augmented["image"], (2, 0, 1))
        label = augmented["mask"]

        return torch.tensor(x, dtype=torch.float32), torch.tensor(label, dtype=torch.float32).unsqueeze(0)
