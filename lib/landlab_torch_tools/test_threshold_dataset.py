import pytest
import torch
import numpy as np
from src.landlab_torch_tools.threshold_dataset import (
    ThresholdDataset,
    MultiThresholdDataset,
    AdaptiveThresholdDataset,
)


class TestThresholdDataset:
    def test_basic_functionality(self):
        """Test basic thresholding functionality"""
        input_array = np.array([[1, 2], [3, 4]], dtype=np.float32)
        thresholds = [1.5, 2.5, 3.5]
        dataset = ThresholdDataset(input_array, thresholds)

        assert len(dataset) == 3

        # Test first threshold (1.5)
        thresholded, threshold = dataset[0]
        expected = np.array([[0, 1], [1, 1]], dtype=np.float32)
        assert torch.allclose(thresholded, torch.from_numpy(expected))
        assert torch.allclose(threshold, torch.tensor(1.5))

        # Test second threshold (2.5)
        thresholded, threshold = dataset[1]
        expected = np.array([[0, 0], [1, 1]], dtype=np.float32)
        assert torch.allclose(thresholded, torch.from_numpy(expected))
        assert torch.allclose(threshold, torch.tensor(2.5))

    def test_return_threshold_false(self):
        """Test with return_threshold=False"""
        input_array = np.array([1, 2, 3, 4])
        thresholds = [2.5]
        dataset = ThresholdDataset(input_array, thresholds, return_threshold=False)

        result = dataset[0]
        assert isinstance(result, torch.Tensor)
        expected = np.array([0, 0, 1, 1], dtype=np.float32)
        assert torch.allclose(result, torch.from_numpy(expected))

    def test_empty_thresholds_error(self):
        """Test error when no thresholds provided"""
        input_array = np.array([1, 2, 3])
        with pytest.raises(
            ValueError, match="At least one threshold value must be provided"
        ):
            ThresholdDataset(input_array, [])

    def test_different_input_types(self):
        """Test with different input array types"""
        # Test with list
        input_list = [1, 2, 3, 4]
        dataset = ThresholdDataset(input_list, [2.5])
        thresholded, _ = dataset[0]
        expected = np.array([0, 0, 1, 1], dtype=np.float32)
        assert torch.allclose(thresholded, torch.from_numpy(expected))

        # Test with numpy array of different dtype
        input_int = np.array([1, 2, 3, 4], dtype=int)
        dataset = ThresholdDataset(input_int, [2.5])
        thresholded, _ = dataset[0]
        assert torch.allclose(thresholded, torch.from_numpy(expected))


class TestMultiThresholdDataset:
    def test_basic_functionality(self):
        """Test basic functionality with multiple arrays"""
        array1 = np.array([1, 2, 3])
        array2 = np.array([4, 5, 6])
        thresholds = [2.5, 4.5]

        dataset = MultiThresholdDataset([array1, array2], thresholds)
        assert len(dataset) == 4  # 2 arrays × 2 thresholds

        # Test array1 with threshold 2.5 (index 0)
        thresholded, threshold = dataset[0]
        expected = np.array([0, 0, 1], dtype=np.float32)
        assert torch.allclose(thresholded, torch.from_numpy(expected))
        assert torch.allclose(threshold, torch.tensor(2.5))

        # Test array1 with threshold 4.5 (index 1)
        thresholded, threshold = dataset[1]
        expected = np.array([0, 0, 0], dtype=np.float32)
        assert torch.allclose(thresholded, torch.from_numpy(expected))
        assert torch.allclose(threshold, torch.tensor(4.5))

        # Test array2 with threshold 2.5 (index 2)
        thresholded, threshold = dataset[2]
        expected = np.array([1, 1, 1], dtype=np.float32)
        assert torch.allclose(thresholded, torch.from_numpy(expected))
        assert torch.allclose(threshold, torch.tensor(2.5))

    def test_return_array_index_only(self):
        """Test with return_array_index=True, return_threshold=False"""
        array1 = np.array([1, 2])
        array2 = np.array([3, 4])
        thresholds = [1.5]

        dataset = MultiThresholdDataset(
            [array1, array2],
            thresholds,
            return_threshold=False,
            return_array_index=True,
        )

        # Test array1 (index 0)
        thresholded, array_idx = dataset[0]
        expected = np.array([0, 1], dtype=np.float32)
        assert torch.allclose(thresholded, torch.from_numpy(expected))
        assert torch.equal(array_idx, torch.tensor(0, dtype=torch.long))

        # Test array2 (index 1)
        thresholded, array_idx = dataset[1]
        expected = np.array([1, 1], dtype=np.float32)
        assert torch.allclose(thresholded, torch.from_numpy(expected))
        assert torch.equal(array_idx, torch.tensor(1, dtype=torch.long))

    def test_return_both_flags(self):
        """Test with both return_threshold=True and return_array_index=True"""
        array1 = np.array([1, 2])
        thresholds = [1.5]

        dataset = MultiThresholdDataset(
            [array1], thresholds, return_threshold=True, return_array_index=True
        )

        thresholded, threshold, array_idx = dataset[0]
        expected = np.array([0, 1], dtype=np.float32)
        assert torch.allclose(thresholded, torch.from_numpy(expected))
        assert torch.allclose(threshold, torch.tensor(1.5))
        assert torch.equal(array_idx, torch.tensor(0, dtype=torch.long))

    def test_no_return_flags(self):
        """Test with both return flags set to False"""
        array1 = np.array([1, 2, 3])
        thresholds = [2.5]

        dataset = MultiThresholdDataset(
            [array1], thresholds, return_threshold=False, return_array_index=False
        )

        result = dataset[0]
        assert isinstance(result, torch.Tensor)
        expected = np.array([0, 0, 1], dtype=np.float32)
        assert torch.allclose(result, torch.from_numpy(expected))

    def test_empty_arrays_error(self):
        """Test error when no arrays provided"""
        with pytest.raises(
            ValueError, match="At least one input array must be provided"
        ):
            MultiThresholdDataset([], [1.0])

    def test_empty_thresholds_error(self):
        """Test error when no thresholds provided"""
        with pytest.raises(
            ValueError, match="At least one threshold value must be provided"
        ):
            MultiThresholdDataset([np.array([1, 2])], [])

    def test_index_calculation(self):
        """Test correct index calculation for array and threshold combinations"""
        array1 = np.array([1])
        array2 = np.array([2])
        thresholds = [0.5, 1.5, 2.5]  # 3 thresholds

        dataset = MultiThresholdDataset(
            [array1, array2], thresholds, return_threshold=True, return_array_index=True
        )

        # Should be 2 arrays × 3 thresholds = 6 total items
        assert len(dataset) == 6

        # Test specific index calculations
        # Index 0: array1, threshold 0.5
        _, threshold, array_idx = dataset[0]
        assert torch.allclose(threshold, torch.tensor(0.5))
        assert torch.equal(array_idx, torch.tensor(0))

        # Index 3: array2, threshold 0.5
        _, threshold, array_idx = dataset[3]
        assert torch.allclose(threshold, torch.tensor(0.5))
        assert torch.equal(array_idx, torch.tensor(1))

        # Index 5: array2, threshold 2.5
        _, threshold, array_idx = dataset[5]
        assert torch.allclose(threshold, torch.tensor(2.5))
        assert torch.equal(array_idx, torch.tensor(1))


class TestAdaptiveThresholdDataset:
    def test_basic_functionality(self):
        """Test basic adaptive thresholding"""
        # Create array with known distribution
        input_array = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.float32)

        dataset = AdaptiveThresholdDataset(
            input_array, num_thresholds=3, percentile_range=(20, 80)
        )

        assert len(dataset) == 3

        # Check that thresholds are reasonable
        thresholds = dataset.thresholds
        assert len(thresholds) == 3
        assert thresholds[0] < thresholds[1] < thresholds[2]

        # 20th percentile should be around 2.8, 80th around 8.2
        assert 2 <= thresholds[0] <= 4
        assert 7 <= thresholds[2] <= 9

    def test_return_threshold_false(self):
        """Test with return_threshold=False"""
        input_array = np.array([1, 2, 3, 4, 5])
        dataset = AdaptiveThresholdDataset(
            input_array, num_thresholds=2, return_threshold=False
        )

        result = dataset[0]
        assert isinstance(result, torch.Tensor)
        assert result.shape == (5,)

    def test_edge_case_percentiles(self):
        """Test edge case percentile ranges"""
        input_array = np.array([1, 2, 3, 4, 5])

        # Test full range (0, 100)
        dataset = AdaptiveThresholdDataset(
            input_array, num_thresholds=3, percentile_range=(0, 100)
        )
        thresholds = dataset.thresholds
        assert thresholds[0] == 1.0  # 0th percentile
        assert thresholds[2] == 5.0  # 100th percentile

    def test_single_threshold(self):
        """Test with single threshold"""
        input_array = np.array([1, 2, 3, 4, 5])
        dataset = AdaptiveThresholdDataset(input_array, num_thresholds=1)

        assert len(dataset) == 1
        thresholded, threshold = dataset[0]
        assert isinstance(thresholded, torch.Tensor)
        assert isinstance(threshold, torch.Tensor)

    def test_invalid_num_thresholds(self):
        """Test error for invalid num_thresholds"""
        input_array = np.array([1, 2, 3])
        with pytest.raises(ValueError, match="num_thresholds must be at least 1"):
            AdaptiveThresholdDataset(input_array, num_thresholds=0)

    def test_invalid_percentile_range(self):
        """Test error for invalid percentile range"""
        input_array = np.array([1, 2, 3])

        # Test inverted range
        with pytest.raises(ValueError, match="percentile_range must be"):
            AdaptiveThresholdDataset(input_array, percentile_range=(80, 20))

        # Test out of bounds
        with pytest.raises(ValueError, match="percentile_range must be"):
            AdaptiveThresholdDataset(input_array, percentile_range=(-10, 50))

        with pytest.raises(ValueError, match="percentile_range must be"):
            AdaptiveThresholdDataset(input_array, percentile_range=(50, 110))


class TestThresholdBehavior:
    def test_threshold_boundary_conditions(self):
        """Test boundary conditions for thresholding"""
        input_array = np.array([1, 2, 3, 4])

        # Test threshold exactly equal to a value
        dataset = ThresholdDataset(input_array, [2.0])
        thresholded, _ = dataset[0]
        # Values > 2.0 should be 1, values <= 2.0 should be 0
        expected = np.array([0, 0, 1, 1], dtype=np.float32)
        assert torch.allclose(thresholded, torch.from_numpy(expected))

        # Test very high threshold
        dataset = ThresholdDataset(input_array, [10.0])
        thresholded, _ = dataset[0]
        expected = np.array([0, 0, 0, 0], dtype=np.float32)
        assert torch.allclose(thresholded, torch.from_numpy(expected))

        # Test very low threshold
        dataset = ThresholdDataset(input_array, [0.0])
        thresholded, _ = dataset[0]
        expected = np.array([1, 1, 1, 1], dtype=np.float32)
        assert torch.allclose(thresholded, torch.from_numpy(expected))


if __name__ == "__main__":
    pytest.main([__file__])
