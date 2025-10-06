import torch
import numpy as np
from typing import Union, Tuple
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class HorizontalSwap(object):
    """
    Augmentation that splits an image vertically down its center and swaps left and right halves.

    Design Justification: Implemented as a standalone transform class following PyTorch's
    torchvision.transforms convention. This approach provides maximum flexibility and reusability,
    allowing the transform to be easily composed with other transforms using transforms.Compose.
    The transform can be applied to any tensor-based image and integrates seamlessly with existing
    PyTorch workflows.
    """

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        """
        Args:
            img (torch.Tensor): Image tensor of shape (C, H, W) or (H, W, C)

        Returns:
            torch.Tensor: Image with left and right halves swapped
        """
        if img.dim() == 3:
            # Handle both (C, H, W) and (H, W, C) formats
            if img.shape[-1] <= 3:  # Assume (H, W, C) if last dimension is small
                height, width = img.shape[:2]
                mid_width = width // 2
                left_half = img[:, :mid_width, :]
                right_half = img[:, mid_width:, :]
                return torch.cat([right_half, left_half], dim=1)
            else:  # Assume (C, H, W)
                channels, height, width = img.shape
                mid_width = width // 2
                left_half = img[:, :, :mid_width]
                right_half = img[:, :, mid_width:]
                return torch.cat([right_half, left_half], dim=2)
        elif img.dim() == 2:
            # Handle grayscale (H, W)
            height, width = img.shape
            mid_width = width // 2
            left_half = img[:, :mid_width]
            right_half = img[:, mid_width:]
            return torch.cat([right_half, left_half], dim=1)
        else:
            raise ValueError(f"Unsupported tensor dimensions: {img.shape}")


class GridShuffle(object):
    """
    Augmentation that divides an image into an M x N grid and randomly shuffles the patches.

    Design Justification: Implemented as a standalone transform class following PyTorch's
    torchvision.transforms convention. The random shuffling can be made deterministic by
    setting torch.manual_seed before calling the transform. This design allows for both
    random augmentation during training and reproducible results during testing.
    """

    def __init__(self, grid_height: int = 3, grid_width: int = 3):
        """
        Args:
            grid_height (int): Number of patches vertically (M)
            grid_width (int): Number of patches horizontally (N)
        """
        self.grid_height = grid_height
        self.grid_width = grid_width

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        """
        Args:
            img (torch.Tensor): Image tensor of shape (C, H, W) or (H, W, C)

        Returns:
            torch.Tensor: Image with shuffled patches
        """
        if img.dim() == 3:
            if img.shape[-1] <= 3:  # Assume (H, W, C)
                return self._shuffle_hwc_format(img)
            else:  # Assume (C, H, W)
                return self._shuffle_chw_format(img)
        elif img.dim() == 2:
            return self._shuffle_grayscale(img)
        else:
            raise ValueError(f"Unsupported tensor dimensions: {img.shape}")

    def _shuffle_chw_format(self, img: torch.Tensor) -> torch.Tensor:
        """Shuffle patches for (C, H, W) format"""
        channels, height, width = img.shape
        patch_h = height // self.grid_height
        patch_w = width // self.grid_width

        # Extract patches
        patches = []
        for i in range(self.grid_height):
            for j in range(self.grid_width):
                start_h = i * patch_h
                end_h = start_h + patch_h
                start_w = j * patch_w
                end_w = start_w + patch_w
                patch = img[:, start_h:end_h, start_w:end_w]
                patches.append(patch)

        # Shuffle patches
        indices = torch.randperm(len(patches))
        shuffled_patches = [patches[i] for i in indices]

        # Reassemble image
        result = torch.zeros_like(img)
        for idx, (i, j) in enumerate(
            [(i, j) for i in range(self.grid_height) for j in range(self.grid_width)]
        ):
            start_h = i * patch_h
            end_h = start_h + patch_h
            start_w = j * patch_w
            end_w = start_w + patch_w
            result[:, start_h:end_h, start_w:end_w] = shuffled_patches[idx]

        return result

    def _shuffle_hwc_format(self, img: torch.Tensor) -> torch.Tensor:
        """Shuffle patches for (H, W, C) format"""
        height, width, channels = img.shape
        patch_h = height // self.grid_height
        patch_w = width // self.grid_width

        # Extract patches
        patches = []
        for i in range(self.grid_height):
            for j in range(self.grid_width):
                start_h = i * patch_h
                end_h = start_h + patch_h
                start_w = j * patch_w
                end_w = start_w + patch_w
                patch = img[start_h:end_h, start_w:end_w, :]
                patches.append(patch)

        # Shuffle patches
        indices = torch.randperm(len(patches))
        shuffled_patches = [patches[i] for i in indices]

        # Reassemble image
        result = torch.zeros_like(img)
        for idx, (i, j) in enumerate(
            [(i, j) for i in range(self.grid_height) for j in range(self.grid_width)]
        ):
            start_h = i * patch_h
            end_h = start_h + patch_h
            start_w = j * patch_w
            end_w = start_w + patch_w
            result[start_h:end_h, start_w:end_w, :] = shuffled_patches[idx]

        return result

    def _shuffle_grayscale(self, img: torch.Tensor) -> torch.Tensor:
        """Shuffle patches for grayscale (H, W) format"""
        height, width = img.shape
        patch_h = height // self.grid_height
        patch_w = width // self.grid_width

        # Extract patches
        patches = []
        for i in range(self.grid_height):
            for j in range(self.grid_width):
                start_h = i * patch_h
                end_h = start_h + patch_h
                start_w = j * patch_w
                end_w = start_w + patch_w
                patch = img[start_h:end_h, start_w:end_w]
                patches.append(patch)

        # Shuffle patches
        indices = torch.randperm(len(patches))
        shuffled_patches = [patches[i] for i in indices]

        # Reassemble image
        result = torch.zeros_like(img)
        for idx, (i, j) in enumerate(
            [(i, j) for i in range(self.grid_height) for j in range(self.grid_width)]
        ):
            start_h = i * patch_h
            end_h = start_h + patch_h
            start_w = j * patch_w
            end_w = start_w + patch_w
            result[start_h:end_h, start_w:end_w] = shuffled_patches[idx]

        return result


class ImageDataset(Dataset):
    """
    Simple Dataset class that loads images from file paths and applies transforms.
    Compatible with the new shuffling augmentations.
    """

    def __init__(self, image_paths: list, transform=None):
        """
        Args:
            image_paths (list): List of paths to image files
            transform: Transform to apply to each image
        """
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        # Load image using PIL and convert to tensor
        image = Image.open(image_path).convert("RGB")
        image_tensor = transforms.ToTensor()(image)

        if self.transform:
            image_tensor = self.transform(image_tensor)

        return image_tensor, image_path
