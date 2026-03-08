import os

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import torch
import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

DATASET_INFO = {
    "imagenet": {
        "path": "your datatset path",
        "split": "train",
    },
    "imagenet-sketch": {
        "path": "your datatset path",
        "split": "train",
    },
    "oxford_flowers": {
        "path": "your datatset path",
        "split": "train",
    },
    "caltech101": {
        "path": "your datatset path",
        "split": "train",
        "name": "with_background_category",
    },
}


def calculate_entropy(top_val, top_label, ignore_label_idx: int = None, eps=1e-9):
    dict_size = top_label.shape[0]
    entropy = torch.zeros(dict_size)

    for i in range(dict_size):
        unique_labels, counts = top_label[i].unique(return_counts=True)
        if ignore_label_idx is not None:
            counts = counts[unique_labels != ignore_label_idx]
            unique_labels = unique_labels[unique_labels != ignore_label_idx]
        if len(unique_labels) != 0:
            if counts.sum().item() < 10:
                entropy[i] = -1  
            else:
                summed_probs = torch.zeros_like(unique_labels, dtype=top_val.dtype)
                for j, label in enumerate(unique_labels):
                    summed_probs[j] = top_val[i][top_label[i] == label].sum().item()
                summed_probs = summed_probs / summed_probs.sum()
                entropy[i] = -torch.sum(summed_probs * torch.log(summed_probs + eps))
        else:
            entropy[i] = -1
    return entropy


def load_stats(save_directory: str, device: torch.device):
    mean_acts = torch.load(
        os.path.join(save_directory, "sae_mean_acts.pt"),
        map_location=torch.device(device),
    )
    sparsity = torch.load(
        os.path.join(save_directory, "sae_sparsity.pt"),
        map_location=torch.device(device),
    )
    top_val = torch.load(
        os.path.join(save_directory, "max_activating_image_values.pt"),
        map_location=torch.device(device),
    )
    top_idx = torch.load(
        os.path.join(save_directory, "max_activating_image_indices.pt"),
        map_location=torch.device(device),
    )
    top_label = torch.load(
        os.path.join(save_directory, "max_activating_image_label_indices.pt"),
        map_location=torch.device(device),
    )
    try:
        top_entropy = torch.load(
            os.path.join(save_directory, "top_entropy.pt"),
            map_location=torch.device(device),
        )
    except:  
        print("Calculating top entropy")
        top_entropy = calculate_entropy(top_val, top_label)
        torch.save(top_entropy, os.path.join(save_directory, "top_entropy.pt"))

    print(f"Stats loaded from {save_directory}")

    stats = {
        "mean_acts": mean_acts.to(device),
        "sparsity": sparsity.to(device),
        "top_val": top_val.to(device),
        "top_idx": top_idx.to(device).to(torch.int64),
        "top_label": top_label.to(device).to(torch.int64),
        "top_entropy": top_entropy.to(device),
    }
    return stats


def get_stats_scatter_plot(stats, mask=None, save_directory: str = None, eps=1e-9):
    if mask is None:
        mask = torch.ones_like(
            stats["sparsity"], dtype=torch.bool, device=stats["sparsity"].device
        )

    indices = torch.where(mask)[0]
    plotting_data = torch.stack(
        [
            torch.log10(stats["sparsity"][mask] + eps),
            torch.log10(stats["mean_acts"][mask] + eps),
            stats["top_entropy"][mask],
            indices,
        ],
        dim=0,
    )
    plotting_data = plotting_data.transpose(0, 1)

    df = pd.DataFrame(
        plotting_data.cpu().numpy(),
        columns=["log10_sparsity", "log10_mean_acts", "entropy", "index"]
    )

    sns.set(style="whitegrid", font_scale=1.2)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        df["log10_sparsity"],
        df["log10_mean_acts"],
        c=df["entropy"],
        cmap="viridis",
        alpha=0.5
    )
    cbar = plt.colorbar(scatter)
    cbar.set_label("Entropy")

    plt.xlabel("log10(sparsity)")
    plt.ylabel("log10(mean_acts)")
    plt.title("Stats Scatter Plot")

    if save_directory is not None:
        os.makedirs(save_directory, exist_ok=True)
        save_path = os.path.join(save_directory, "scatter_plot.png")
        plt.savefig(save_path, dpi=300)
        print(f"Saving at: {save_path}")

    plt.show()

def plot_ref_images(stats, dataset, latent_idx: int, plot_top_k: int = 10, eps=1e-9):
    resize_size = 224
    num_cols = 5
    num_rows = plot_top_k // num_cols

    images = []
    labels = []

    for i, idx in enumerate(stats["top_idx"][latent_idx][:plot_top_k]):
        img = dataset[idx.item()]["image"]
        images.append(img.resize((resize_size, resize_size)))
        assert dataset[idx.item()]["label"] == stats["top_label"][latent_idx][i], (
            "label mismatch, try matching dataset shuffle seed"
        )
        labels.append(dataset[idx.item()]["label"])

    _, axes = plt.subplots(num_rows, num_cols, figsize=(4.5 * num_cols, 6 * num_rows))

    for i, (image, label) in enumerate(zip(images, labels)):
        ax = axes[i // num_cols, i % num_cols]
        ax.imshow(image)
        top_val_i = torch.log10(stats["top_val"][latent_idx][i] + eps).item()
        ax.set_title(f"{label}(a: {top_val_i:.2f})", fontsize=35)
        ax.axis("off")

    mean_acts = torch.log10(stats["mean_acts"][latent_idx] + eps).item()
    sparsity = torch.log10(stats["sparsity"][latent_idx] + eps).item()
    plt.suptitle(
        f"Index {latent_idx} (f: {sparsity:.2f}, a: {mean_acts:.2f}, e: {stats['top_entropy'][latent_idx]:.2f})\n",
        fontsize=35,
    )

    plt.tight_layout()
    plt.show()
