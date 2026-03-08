import json
import os
from collections import defaultdict
from typing import Dict, Tuple

import numpy as np
import torch
from datasets import Dataset, load_dataset
from tqdm import tqdm

from src.models.utils import get_adapted_clip, get_base_clip
from src.sae_training.config import Config
from src.sae_training.hooked_vit import HookedVisionTransformer
from src.sae_training.sparse_autoencoder import SparseAutoencoder

# Dataset configurations
DATASET_INFO = {
    "imagenet": {
        "path": "path to datasets",
        "split": "val",
    },
    "imagenet-sketch": {
        "path": "path to datasets",
        "split": "train",
    },
 
    "caltech101": {
        "path": "path to datasets",
        "split": "train",
        "name": "default",
    },
    "dtd" :{
        "path": "path to datasets",
        "split": "train",
    },
    "standfordcars": {
        "path": "imagefolder",
        "data_files": {"train": "path to datasets"},
        "split": "train",
        "cache_dir": "/tmp/hf_cache_cars",
    },

    "food101": {
        "path": "path to datasets",
        "split": "train",   
    },
    "ucf101": {
        "path": "imagefolder",
        "data_dir": "path to datasets",
        "split": "train",
    },

    "eurosat": {
        "path": "imagefolder",
        "data_files": {"train": "path to datasets"},
        "split": "train",

    },
    "fgvc_aircraft_variant": {
        "path": "imagefolder",
        "data_files": {"train": "path to datasets"},
        "split": "train",
    },
    "sun397": {
        "path": "parquet",
        "data_files": {
            "train": "path to datasets",
            "test":  "path to datasets",
        },
        "split": "train",
    },
    "oxfordpets": {
        "path": "path to datasets",  
        "split": "train",              
    },
    "oxford_flowers": {
        "path": "imagefolder",
        "data_dir": "path to datasets",  
        "split": "train", 
    },

}

SAE_DIM = 49152


def load_sae(sae_path: str, device: str) -> tuple[SparseAutoencoder, Config]:
    """Load a sparse autoencoder model from a checkpoint file."""
    checkpoint = torch.load(sae_path, map_location="cpu")
    
    if "cfg" in checkpoint:
        cfg = Config(checkpoint["cfg"])
    else:
        cfg = Config(checkpoint["config"])
    sae = SparseAutoencoder(cfg, device)
    sae.load_state_dict(checkpoint["state_dict"], strict=False)
    sae.eval().to(device)

    return sae, cfg


def load_hooked_vit(
    cfg: Config,
    vit_type: str,
    backbone: str,
    device: str,
    model_path: str = None,
    config_path: str = None,
    classnames: list[str] = None,
) -> HookedVisionTransformer:
    """Load a vision transformer model with hooks."""
    if vit_type == "base":
        model, processor = get_base_clip(backbone)
    else:
        model, processor = get_adapted_clip(
            cfg, vit_type, model_path, config_path, backbone, classnames
        )

    return HookedVisionTransformer(model, processor, device=device)


def get_sae_and_vit(
    sae_path: str,
    vit_type: str,
    device: str,
    backbone: str,
    model_path: str = None,
    config_path: str = None,
    classnames: list[str] = None,
) -> tuple[SparseAutoencoder, HookedVisionTransformer, Config]:
    """Load both SAE and ViT models."""
    sae, cfg = load_sae(sae_path, device)
    vit = load_hooked_vit(
        cfg, vit_type, backbone, device, model_path, config_path, classnames
    )
    return sae, vit, cfg
from datasets import load_from_disk
def load_and_organize_dataset(dataset_name: str) -> Tuple[list, Dict]:
    print(DATASET_INFO[dataset_name])
    dataset = load_dataset(**DATASET_INFO[dataset_name])
    classnames = get_classnames(dataset_name, dataset)

    data_by_class = defaultdict(list)
    for data_item in tqdm(dataset):
        label = data_item["label"]
        classname = classnames[label]  
        data_by_class[classname].append(data_item)

    return classnames, data_by_class

def _ucf_camel_to_underscore(s: str) -> str:
    parts = re.findall(r"[A-Z][^A-Z]*", s)
    return "_".join([p.strip("_") for p in parts if p])

def get_classnames(
    dataset_name: str, dataset: Dataset = None, data_root: str = "./configs/classnames"
) -> list[str]:
    """Get class names for a dataset."""

    filename = f"{data_root}/{dataset_name}_classnames"
    txt_filename = filename + ".txt"
    json_filename = filename + ".json"

    if not os.path.exists(txt_filename) and not os.path.exists(json_filename):
        raise ValueError(f"Dataset {dataset_name} not supported")

    filename = json_filename if os.path.exists(json_filename) else txt_filename

    with open(filename, "r") as file:
        if dataset_name == "caltech101":
            class_names = [line.strip() for line in file.readlines()]
        elif dataset_name == "imagenet" or dataset_name == "imagenet-sketch":
            class_names = [
                " ".join(line.strip().split(" ")[1:]) for line in file.readlines()
            ]
        elif dataset_name == "oxford_flowers":
            assert dataset is not None, "Dataset must be provided for Oxford Flowers"
            new_class_dict = {}
            class_names = json.load(file)
            # print(dataset)
            # print(dataset.features)
            classnames_from_hf = dataset.features["label"].names
            for i, class_name in enumerate(classnames_from_hf):
                new_class_dict[i] = class_names[class_name]
            class_names = list(new_class_dict.values())


        elif dataset_name == "dtd":
            assert dataset is not None, "Dataset must be provided for DTD"
            new_class_dict = {}
            class_names = json.load(file)
            # print(dataset.features)
            classnames_from_hf = dataset.features["label"].names
            for i, class_name in enumerate(classnames_from_hf):
                new_class_dict[i] = class_names[class_name]
            class_names = list(new_class_dict.values())
        elif dataset_name == "food101":
            assert dataset is not None, "Dataset must be provided for Food101"
            new_class_dict = {}
            class_names = json.load(file)
            # print(dataset.features)
            classnames_from_hf = dataset.features["label"].names
            for i, class_name in enumerate(classnames_from_hf):
                new_class_dict[i] = class_names[class_name]
            class_names = list(new_class_dict.values())
                    # === Stanford Cars ===
        elif dataset_name =="standfordcars":
            assert dataset is not None, "Dataset must be provided for Stanford Cars"
            obj = json.load(file)  
            print(dataset.features)
            classnames_from_hf = dataset.features["label"].names  
            if isinstance(obj, dict):
                class_names = [obj.get(name, name) for name in classnames_from_hf]
            elif isinstance(obj, list):
                if len(obj) != len(classnames_from_hf):
                    raise ValueError(f"Stanford Cars JSON has {len(obj)} names but dataset has {len(classnames_from_hf)} classes")
                class_names = obj
            else:
                raise ValueError("Unsupported JSON format for Stanford Cars")

        elif dataset_name == "ucf101":
            obj = json.load(file) 
            if isinstance(obj, dict):
                class_names = list(obj.values())
            elif isinstance(obj, list):
                class_names = obj
            else:
                raise ValueError("Unsupported JSON format for ucf101")
                # === EuroSAT ===
        elif dataset_name in ("eurosat", "eurosat_rgb", "eurosat-rgb"):
            assert dataset is not None, "Dataset must be provided for EuroSAT"

            if filename.endswith(".json"):
                obj = json.load(file)
                if isinstance(obj, dict):
                    hf_names = list(dataset.features["label"].names)
                    class_names = [obj.get(name, name) for name in hf_names]
                elif isinstance(obj, list):
                    class_names = obj
                else:
                    raise ValueError("Unsupported JSON format for EuroSAT")
            else:
                class_names = [line.strip() for line in file if line.strip()]
        elif dataset_name in ("fgvc_aircraft_variant", "fgvc-aircraft-variant", "fgvc_aircraft"):
            assert dataset is not None, "Dataset must be provided for FGVC-Aircraft (variant)"
            obj = json.load(file)

            hf_names = None
            try:
                hf_names = list(dataset.features["label"].names)
            except Exception:
                hf_names = None

            if isinstance(obj, dict):
                if hf_names is not None:
                    class_names = [obj.get(name, name) for name in hf_names]
                else:
                    class_names = list(obj.values())
            elif isinstance(obj, list):
                class_names = obj
            else:
                raise ValueError("Unsupported JSON format for FGVC-Aircraft (variant)")
        elif dataset_name == "sun397":
            if filename.endswith(".json"):
                print("Using JSON for SUN397 class names")
                obj = json.load(file)  
                if isinstance(obj, dict):
                    class_names = list(obj.keys())
                elif isinstance(obj, list):
                    class_names = obj
                else:
                    raise ValueError("Unsupported JSON format for SUN397")
            else:
                raw_lines = [ln.strip() for ln in file if ln.strip()]
                class_names = []
                seen = set()
                for ln in raw_lines:
                    rel = ln.lstrip("/")           
                    parts = rel.split("/")
                    if len(parts) < 2:
                        continue
                    cls_key = "/".join(parts[1:]) 
                    if cls_key not in seen:
                        seen.add(cls_key)
                        class_names.append(cls_key)
        elif dataset_name in ("ucf101", "UCF101", "ucf-101"):
            obj = json.load(file)
            if isinstance(obj, dict):
                if dataset is not None and hasattr(dataset, "features") and "label" in dataset.features:
                    hf_names = list(dataset.features["label"].names)  
                    underscore_to_display = {_ucf_camel_to_underscore(k): v for k, v in obj.items()}
                    class_names = [underscore_to_display.get(name, name) for name in hf_names]
                else:
                    class_names = list(obj.values())

            elif isinstance(obj, list):
                class_names = obj
            else:
                raise ValueError("Unsupported JSON format for ucf101")

        elif dataset_name in ("oxfordpets", "oxford_pets", "oxford-iiit-pet"):
            assert dataset is not None, "Dataset must be provided for Oxford Pets"
            class_names = list(dataset.features["label"].names)
        elif dataset_name in ("eurosat", "eurosat_rgb", "eurosat-rgb"):
            obj = json.load(file)
            hf_names = None
            try:
                if dataset is not None and "label" in dataset.features:
                    hf_names = list(dataset.features["label"].names)
            except Exception:
                hf_names = None

            if isinstance(obj, dict):
                if hf_names is not None:
                    class_names = [obj.get(name, name) for name in hf_names]
                else:
                    class_names = list(obj.values())
            elif isinstance(obj, list):
                class_names = obj
            else:
                raise ValueError("Unsupported JSON format for EuroSAT")

        else:
            raise ValueError(f"Dataset {dataset_name} not supported")   

    return class_names


def setup_save_directory(
    root_dir: str, save_name: str, sae_path: str, vit_type: str, dataset_name: str
) -> str:
    """Set and create the save directory path."""
    sae_run_name = sae_path.split("/")[-2]
    save_directory = (
        f"{root_dir}/{save_name}/sae_{sae_run_name}/{vit_type}/{dataset_name}"
    )
    os.makedirs(save_directory, exist_ok=True)
    return save_directory


def get_sae_activations(
    model_activations: torch.Tensor, sae: SparseAutoencoder
) -> torch.Tensor:
    """Extract and process activations from the sparse autoencoder."""
    hook_name = "hook_hidden_post"

    # Run SAE forward pass and get activations from cache
    _, cache = sae.run_with_cache(model_activations)
    sae_activations = cache[hook_name]
    np.save("all_hidden_post.npy", sae_activations.detach().cpu().numpy())
    # Average across sequence length dimension if needed
    if len(sae_activations.size()) > 2:
        sae_activations = sae_activations.mean(dim=1)

    return sae_activations


def process_batch(vit, batch_data, device):
    """Process a single batch of images."""
    images = [data["image"] for data in batch_data]

    inputs = vit.processor(
        images=images, text="", return_tensors="pt", padding=True
    ).to(device)
    return inputs


def get_max_acts_and_images(
    datasets: dict, feat_data_root: str, sae_runname: str, vit_name: str
) -> tuple[dict, dict]:
    """Load and return maximum activations and mean activations for each dataset."""
    max_act_imgs = {}
    mean_acts = {}
    for dataset_name in datasets:
        # Load max activating image indices
        max_act_path = os.path.join(
            feat_data_root,
            f"{sae_runname}/{vit_name}/{dataset_name}",
            "max_activating_image_indices.pt",
        )
        max_act_imgs[dataset_name] = torch.load(max_act_path, map_location="cpu").to(
            torch.int32
        )

        # Load mean activations
        mean_acts_path = os.path.join(
            feat_data_root,
            f"{sae_runname}/{vit_name}/{dataset_name}",
            "sae_mean_acts.pt",
        )
        mean_acts[dataset_name] = torch.load(mean_acts_path, map_location="cpu").numpy()

    return max_act_imgs, mean_acts


def load_datasets(include_imagenet: bool = False, seed: int = 1):
    """Load multiple datasets from HuggingFace."""
    if include_imagenet:
        return {
            "imagenet": load_dataset(
                "your dataset path", split="val"
            ).shuffle(seed=seed),
            "imagenet-sketch": load_dataset(
                "your dataset path", split="train"
            ).shuffle(seed=seed),
            "caltech101": load_dataset(
                "your dataset path",
                "default",
                split="train",
            ).shuffle(seed=seed),
        }
    else:
        return {
            "imagenet-sketch": load_dataset(
                "your dataset path", split="train"
            ).shuffle(seed=seed),
            "caltech101": load_dataset(
                "HuggingFaceM4/Caltech-101",
                "with_background_category",
                split="train",
            ).shuffle(seed=seed),
        }


def get_all_classnames(datasets, data_root):
    """Get class names for all datasets."""
    class_names = {}
    for dataset_name, dataset in datasets.items():
        class_names[dataset_name] = get_classnames(dataset_name, dataset, data_root)

    # imagenet classnames are required to classnames for maple
    if "imagenet" not in class_names:
        filename = f"{data_root}/imagenet_classnames"
        txt_filename = filename + ".txt"
        json_filename = filename + ".json"

        if not os.path.exists(txt_filename) and not os.path.exists(json_filename):
            raise ValueError(f"Dataset {dataset_name} not supported")

        filename = json_filename if os.path.exists(json_filename) else txt_filename

        with open(filename, "r") as file:
            class_names["imagenet"] = [
                " ".join(line.strip().split(" ")[1:]) for line in file.readlines()
            ]

    return class_names
