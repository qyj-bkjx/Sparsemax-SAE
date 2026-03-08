import torch
import matplotlib.pyplot as plt
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
from PIL import Image

data_dir = 'your data path'

indices = torch.load('your max_activating_image_indices.pt path')
n_feats, topk = indices.shape

feat_idx = 0
top_indices = indices[feat_idx][:5]
dataset = ImageFolder(data_dir, transform=ToTensor())
fig, axs = plt.subplots(1, 5, figsize=(15,3))
for i, idx in enumerate(top_indices):
    img, label = dataset[idx.item()]
    axs[i].imshow(img.permute(1,2,0))
    axs[i].set_title(f'idx:{idx.item()}')
    axs[i].axis('off')
plt.suptitle(f'SAE feature {feat_idx} top activating images')
plt.show()
