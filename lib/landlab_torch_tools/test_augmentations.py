import pytest
import torch
import numpy as np
import tempfile
import os
from PIL import Image
from torchvision import transforms

from src.landlab_torch_tools.shuffling_augmentations import (
    HorizontalSwap,
    GridShuffle,
    ImageDataset,
)


class TestHorizontalSwap:
    def test_dimension_integrity_chw(self):
        """Test that output dimensions match input for (C, H, W) format"""
        transform = HorizontalSwap()
        input_tensor = torch.randn(3, 64, 64)
        output_tensor = transform(input_tensor)
        assert output_tensor.shape == input_tensor.shape

    def test_dimension_integrity_hwc(self):
        """Test that output dimensions match input for (H, W, C) format"""
        transform = HorizontalSwap()
        input_tensor = torch.randn(64, 64, 3)
        output_tensor = transform(input_tensor)
        assert output_tensor.shape == input_tensor.shape

    def test_dimension_integrity_grayscale(self):
        """Test that output dimensions match input for grayscale (H, W) format"""
        transform = HorizontalSwap()
        input_tensor = torch.randn(64, 64)
        output_tensor = transform(input_tensor)
        assert output_tensor.shape == input_tensor.shape

    def test_pixel_conservation_chw(self):
        """Test that no pixels are lost or altered in (C, H, W) format"""
        transform = HorizontalSwap()
        input_tensor = torch.randn(3, 64, 64)
        output_tensor = transform(input_tensor)

        # Sort all pixel values from both tensors and compare
        input_sorted = torch.sort(input_tensor.flatten())[0]
        output_sorted = torch.sort(output_tensor.flatten())[0]
        assert torch.allclose(input_sorted, output_sorted, atol=1e-6)

    def test_pixel_conservation_hwc(self):
        """Test that no pixels are lost or altered in (H, W, C) format"""
        transform = HorizontalSwap()
        input_tensor = torch.randn(64, 64, 3)
        output_tensor = transform(input_tensor)

        # Sort all pixel values from both tensors and compare
        input_sorted = torch.sort(input_tensor.flatten())[0]
        output_sorted = torch.sort(output_tensor.flatten())[0]
        assert torch.allclose(input_sorted, output_sorted, atol=1e-6)

    def test_pixel_conservation_grayscale(self):
        """Test that no pixels are lost or altered in grayscale format"""
        transform = HorizontalSwap()
        input_tensor = torch.randn(64, 64)
        output_tensor = transform(input_tensor)

        # Sort all pixel values from both tensors and compare
        input_sorted = torch.sort(input_tensor.flatten())[0]
        output_sorted = torch.sort(output_tensor.flatten())[0]
        assert torch.allclose(input_sorted, output_sorted, atol=1e-6)

    def test_horizontal_swap_correctness(self):
        """Test that horizontal swap actually swaps left and right halves"""
        transform = HorizontalSwap()
        # Create a simple test pattern
        input_tensor = torch.zeros(1, 4, 4)
        input_tensor[0, :, :2] = 1.0  # Left half = 1
        input_tensor[0, :, 2:] = 2.0  # Right half = 2

        output_tensor = transform(input_tensor)

        # After swap, left should be 2, right should be 1
        assert torch.all(output_tensor[0, :, :2] == 2.0)
        assert torch.all(output_tensor[0, :, 2:] == 1.0)

    def test_unsupported_dimensions(self):
        """Test that unsupported tensor dimensions raise ValueError"""
        transform = HorizontalSwap()
        with pytest.raises(ValueError):
            transform(torch.randn(2, 3, 4, 5))  # 4D tensor


class TestGridShuffle:
    def test_dimension_integrity_chw(self):
        """Test that output dimensions match input for (C, H, W) format"""
        transform = GridShuffle(grid_height=3, grid_width=3)
        input_tensor = torch.randn(3, 90, 90)  # Divisible by 3
        output_tensor = transform(input_tensor)
        assert output_tensor.shape == input_tensor.shape

    def test_dimension_integrity_hwc(self):
        """Test that output dimensions match input for (H, W, C) format"""
        transform = GridShuffle(grid_height=2, grid_width=4)
        input_tensor = torch.randn(80, 80, 3)  # 80 divisible by both 2 and 4
        output_tensor = transform(input_tensor)
        assert output_tensor.shape == input_tensor.shape

    def test_dimension_integrity_grayscale(self):
        """Test that output dimensions match input for grayscale format"""
        transform = GridShuffle(grid_height=4, grid_width=2)
        input_tensor = torch.randn(80, 60)  # 80 divisible by 4, 60 by 2
        output_tensor = transform(input_tensor)
        assert output_tensor.shape == input_tensor.shape

    def test_pixel_conservation_chw(self):
        """Test that no pixels are lost or altered in (C, H, W) format"""
        transform = GridShuffle(grid_height=3, grid_width=3)
        input_tensor = torch.randn(3, 90, 90)
        output_tensor = transform(input_tensor)

        # Sort all pixel values from both tensors and compare
        input_sorted = torch.sort(input_tensor.flatten())[0]
        output_sorted = torch.sort(output_tensor.flatten())[0]
        assert torch.allclose(input_sorted, output_sorted, atol=1e-6)

    def test_pixel_conservation_hwc(self):
        """Test that no pixels are lost or altered in (H, W, C) format"""
        transform = GridShuffle(grid_height=2, grid_width=4)
        input_tensor = torch.randn(80, 80, 3)
        output_tensor = transform(input_tensor)

        # Sort all pixel values from both tensors and compare
        input_sorted = torch.sort(input_tensor.flatten())[0]
        output_sorted = torch.sort(output_tensor.flatten())[0]
        assert torch.allclose(input_sorted, output_sorted, atol=1e-6)

    def test_pixel_conservation_grayscale(self):
        """Test that no pixels are lost or altered in grayscale format"""
        transform = GridShuffle(grid_height=4, grid_width=2)
        input_tensor = torch.randn(80, 60)
        output_tensor = transform(input_tensor)

        # Sort all pixel values from both tensors and compare
        input_sorted = torch.sort(input_tensor.flatten())[0]
        output_sorted = torch.sort(output_tensor.flatten())[0]
        assert torch.allclose(input_sorted, output_sorted, atol=1e-6)

    def test_determinism(self):
        """Test that GridShuffle produces identical results with same seed"""
        transform = GridShuffle(grid_height=3, grid_width=3)
        input_tensor = torch.randn(3, 90, 90)

        # First run with seed
        torch.manual_seed(42)
        output1 = transform(input_tensor)

        # Second run with same seed
        torch.manual_seed(42)
        output2 = transform(input_tensor)

        # Should be identical
        assert torch.allclose(output1, output2, atol=1e-6)

    def test_different_seeds_produce_different_results(self):
        """Test that different seeds produce different shuffled results"""
        transform = GridShuffle(grid_height=3, grid_width=3)
        input_tensor = torch.randn(3, 90, 90)

        # First run with seed 42
        torch.manual_seed(42)
        output1 = transform(input_tensor)

        # Second run with seed 123
        torch.manual_seed(123)
        output2 = transform(input_tensor)

        # Should be different (with high probability)
        assert not torch.allclose(output1, output2, atol=1e-6)

    def test_rectangular_grid(self):
        """Test that rectangular grids work correctly"""
        transform = GridShuffle(grid_height=2, grid_width=3)  # 2x3 grid
        input_tensor = torch.randn(1, 60, 90)  # Height divisible by 2, width by 3
        output_tensor = transform(input_tensor)

        assert output_tensor.shape == input_tensor.shape

        # Test pixel conservation
        input_sorted = torch.sort(input_tensor.flatten())[0]
        output_sorted = torch.sort(output_tensor.flatten())[0]
        assert torch.allclose(input_sorted, output_sorted, atol=1e-6)

    def test_single_patch_grid(self):
        """Test edge case with 1x1 grid (no shuffling should occur)"""
        transform = GridShuffle(grid_height=1, grid_width=1)
        input_tensor = torch.randn(3, 64, 64)
        output_tensor = transform(input_tensor)

        # With 1x1 grid, output should be identical to input
        assert torch.allclose(input_tensor, output_tensor, atol=1e-6)

    def test_unsupported_dimensions(self):
        """Test that unsupported tensor dimensions raise ValueError"""
        transform = GridShuffle(grid_height=3, grid_width=3)
        with pytest.raises(ValueError):
            transform(torch.randn(2, 3, 4, 5))  # 4D tensor


class TestDatasetIntegration:
    def setup_method(self):
        """Create temporary image for testing"""
        self.temp_dir = tempfile.mkdtemp()

        # Create a simple test image
        test_image = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        self.image_path = os.path.join(self.temp_dir, "test_image.png")
        Image.fromarray(test_image).save(self.image_path)

    def teardown_method(self):
        """Clean up temporary files"""
        import shutil

        shutil.rmtree(self.temp_dir)

    def test_dataset_with_horizontal_swap(self):
        """Test that HorizontalSwap transform works correctly with Dataset"""
        transform = transforms.Compose([HorizontalSwap()])
        dataset = ImageDataset([self.image_path], transform=transform)

        # Get original and transformed images
        original_dataset = ImageDataset([self.image_path])
        original_img, _ = original_dataset[0]
        transformed_img, _ = dataset[0]

        # Check dimensions are preserved
        assert original_img.shape == transformed_img.shape

        # Check pixel conservation
        original_sorted = torch.sort(original_img.flatten())[0]
        transformed_sorted = torch.sort(transformed_img.flatten())[0]
        assert torch.allclose(original_sorted, transformed_sorted, atol=1e-6)

    def test_dataset_with_grid_shuffle(self):
        """Test that GridShuffle transform works correctly with Dataset"""
        torch.manual_seed(42)  # For deterministic test
        transform = transforms.Compose([GridShuffle(grid_height=2, grid_width=2)])
        dataset = ImageDataset([self.image_path], transform=transform)

        # Get original and transformed images
        original_dataset = ImageDataset([self.image_path])
        original_img, _ = original_dataset[0]
        transformed_img, _ = dataset[0]

        # Check dimensions are preserved
        assert original_img.shape == transformed_img.shape

        # Check pixel conservation
        original_sorted = torch.sort(original_img.flatten())[0]
        transformed_sorted = torch.sort(transformed_img.flatten())[0]
        assert torch.allclose(original_sorted, transformed_sorted, atol=1e-6)

    def test_dataset_with_composed_transforms(self):
        """Test that transforms work correctly when composed together"""
        torch.manual_seed(42)
        transform = transforms.Compose(
            [HorizontalSwap(), GridShuffle(grid_height=2, grid_width=2)]
        )
        dataset = ImageDataset([self.image_path], transform=transform)

        # Get original and transformed images
        original_dataset = ImageDataset([self.image_path])
        original_img, _ = original_dataset[0]
        transformed_img, _ = dataset[0]

        # Check dimensions are preserved
        assert original_img.shape == transformed_img.shape

        # Check pixel conservation
        original_sorted = torch.sort(original_img.flatten())[0]
        transformed_sorted = torch.sort(transformed_img.flatten())[0]
        assert torch.allclose(original_sorted, transformed_sorted, atol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__])
