import torch
import numpy as np
from torch.utils.data import Dataset
from typing import Union, List, Tuple


class ThresholdDataset(Dataset):
    """
    Dataset that takes a numpy array and creates multiple thresholded versions.
    Each item in the dataset corresponds to a different threshold value applied to the input array.
    Values above the threshold become 1, values at or below become 0.

    Return Format:
    - return_threshold=True: (thresholded_array, threshold_value)
    - return_threshold=False: thresholded_array
    """

    def __init__(
        self,
        input_array: np.ndarray,
        thresholds: Union[List[float], np.ndarray],
        return_threshold: bool = True,
        transform=None,
    ):
        """
        Args:
            input_array (np.ndarray): Input numpy array to threshold
            thresholds (List[float] or np.ndarray): List of threshold values to apply
            return_threshold (bool): Whether to return the threshold value as a label. Default True.
            transform: Optional transform to apply to the thresholded array
        """
        self.input_array = np.array(input_array).astype(np.float32)
        self.thresholds = np.array(thresholds).astype(np.float32)
        self.return_threshold = return_threshold
        self.transform = transform

        if len(self.thresholds) == 0:
            raise ValueError("At least one threshold value must be provided")

    def __len__(self):
        return len(self.thresholds)

    def __getitem__(
        self, idx: int
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            idx (int): Index of the threshold to apply

        Returns:
            torch.Tensor or Tuple[torch.Tensor, torch.Tensor]:
                If return_threshold=True, returns (thresholded_array, threshold_value)
                If return_threshold=False, returns only thresholded_array
        """
        threshold = self.thresholds[idx]

        # Apply threshold: values > threshold become 1, others become 0
        thresholded = (self.input_array > threshold).astype(np.float32)

        # Convert to tensor
        thresholded_tensor = torch.from_numpy(thresholded)

        # Apply transform if provided
        if self.transform:
            thresholded_tensor = self.transform(thresholded_tensor)

        if self.return_threshold:
            threshold_tensor = torch.tensor(threshold, dtype=torch.float32)
            return thresholded_tensor, threshold_tensor
        else:
            return thresholded_tensor


class MultiThresholdDataset(Dataset):
    """
    Dataset that takes multiple numpy arrays and applies thresholding to each.
    Useful when you have multiple input arrays and want to apply the same set of thresholds to all.

    IMPORTANT - Different Return Format than other threshold datasets:
    This dataset has more complex return options due to tracking both array index and threshold:
    - No flags: thresholded_array
    - return_threshold only: (thresholded_array, threshold_value)
    - return_array_index only: (thresholded_array, array_index)
    - Both flags: (thresholded_array, threshold_value, array_index)
    """

    def __init__(
        self,
        input_arrays: List[np.ndarray],
        thresholds: Union[List[float], np.ndarray],
        return_threshold: bool = True,
        return_array_index: bool = False,
        transform=None,
    ):
        """
        Args:
            input_arrays (List[np.ndarray]): List of input arrays to threshold
            thresholds (List[float] or np.ndarray): List of threshold values to apply
            return_threshold (bool): Whether to return the threshold value. Default True.
            return_array_index (bool): Whether to return the array index. Default False.
            transform: Optional transform to apply to the thresholded array
        """
        self.input_arrays = [np.array(arr).astype(np.float32) for arr in input_arrays]
        self.thresholds = np.array(thresholds).astype(np.float32)
        self.return_threshold = return_threshold
        self.return_array_index = return_array_index
        self.transform = transform

        if len(self.input_arrays) == 0:
            raise ValueError("At least one input array must be provided")
        if len(self.thresholds) == 0:
            raise ValueError("At least one threshold value must be provided")

    def __len__(self):
        return len(self.input_arrays) * len(self.thresholds)

    def __getitem__(self, idx: int) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        Args:
            idx (int): Index combining array and threshold indices

        Returns:
            Various combinations based on return_* flags:
            - No flags: thresholded_array
            - return_threshold only: (thresholded_array, threshold)
            - return_array_index only: (thresholded_array, array_index)
            - Both flags: (thresholded_array, threshold, array_index)
        """
        # Convert linear index to array_idx, threshold_idx
        array_idx = idx // len(self.thresholds)
        threshold_idx = idx % len(self.thresholds)

        input_array = self.input_arrays[array_idx]
        threshold = self.thresholds[threshold_idx]

        # Apply threshold
        thresholded = (input_array > threshold).astype(np.float32)

        # Convert to tensor
        thresholded_tensor = torch.from_numpy(thresholded)

        # Apply transform if provided
        if self.transform:
            thresholded_tensor = self.transform(thresholded_tensor)

        # Prepare return values based on flags
        result = [thresholded_tensor]

        if self.return_threshold:
            result.append(torch.tensor(threshold, dtype=torch.float32))

        if self.return_array_index:
            result.append(torch.tensor(array_idx, dtype=torch.long))

        if len(result) == 1:
            return result[0]
        else:
            return tuple(result)


class AdaptiveThresholdDataset(Dataset):
    """
    Dataset that automatically generates thresholds based on percentiles of the input array.
    Useful when you want evenly distributed thresholds based on the data distribution.

    Return Format:
    - return_threshold=True: (thresholded_array, threshold_value)
    - return_threshold=False: thresholded_array
    """

    def __init__(
        self,
        input_array: np.ndarray,
        num_thresholds: int = 10,
        percentile_range: Tuple[float, float] = (10, 90),
        return_threshold: bool = True,
        transform=None,
    ):
        """
        Args:
            input_array (np.ndarray): Input numpy array to threshold
            num_thresholds (int): Number of threshold values to generate
            percentile_range (Tuple[float, float]): Range of percentiles to use (min, max)
            return_threshold (bool): Whether to return the threshold value as a label
            transform: Optional transform to apply to the thresholded array
        """
        self.input_array = np.array(input_array).astype(np.float32)
        self.return_threshold = return_threshold
        self.transform = transform

        if num_thresholds < 1:
            raise ValueError("num_thresholds must be at least 1")
        if not (0 <= percentile_range[0] < percentile_range[1] <= 100):
            raise ValueError(
                "percentile_range must be (min, max) where 0 <= min < max <= 100"
            )

        # Generate thresholds based on percentiles
        percentiles = np.linspace(
            percentile_range[0], percentile_range[1], num_thresholds
        )
        self.thresholds = np.percentile(self.input_array, percentiles).astype(
            np.float32
        )

    def __len__(self):
        return len(self.thresholds)

    def __getitem__(
        self, idx: int
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            idx (int): Index of the threshold to apply

        Returns:
            torch.Tensor or Tuple[torch.Tensor, torch.Tensor]:
                If return_threshold=True, returns (thresholded_array, threshold_value)
                If return_threshold=False, returns only thresholded_array
        """
        threshold = self.thresholds[idx]

        # Apply threshold
        thresholded = (self.input_array > threshold).astype(np.float32)

        # Convert to tensor
        thresholded_tensor = torch.from_numpy(thresholded)

        # Apply transform if provided
        if self.transform:
            thresholded_tensor = self.transform(thresholded_tensor)

        if self.return_threshold:
            threshold_tensor = torch.tensor(threshold, dtype=torch.float32)
            return thresholded_tensor, threshold_tensor
        else:
            return thresholded_tensor
