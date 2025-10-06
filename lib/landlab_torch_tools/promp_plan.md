# Prompt Plan: PyTorch Shuffling Augmentations and Tests

## 🎯 Objective
To generate a Python script that implements novel image "shuffling" augmentations for a PyTorch workflow, along with a comprehensive suite of unit tests to ensure correctness and robustness. The script should be flexible, well-designed, and demonstrate practical usage.

---

## 👨‍💻 Persona
You are an expert Python developer with a specialization in the PyTorch deep learning framework, computer vision pipelines, and test-driven development (TDD).

---

## 📝 Context & Input
You will be working with a pre-existing PyTorch `Dataset` class that handles the basic loading of an image from a file path. Your task is to build upon this by creating new augmentation capabilities and ensuring both the new and existing components are thoroughly tested.

---

## Core Task & Requirements
Implement two distinct image shuffling augmentations:

1.  **Horizontal Swap**: This augmentation should split an image vertically down its center and then swap the left and right halves.
2.  **Grid Shuffle**: This augmentation should divide an image into a configurable N x N grid of patches (the value of N, e.g., 3, should be a parameter). It must then randomly shuffle these patches and reassemble them to form a new, jumbled image.

---

## 🤔 Key Design Decision
A crucial part of your task is to decide on the **best architectural pattern** for implementing these augmentations within the PyTorch ecosystem. Analyze the following options and implement the one you determine is most flexible, reusable, and conventional.

* **Option 1 (Pure Transforms)**: Implement `HorizontalSwap` and `GridShuffle` as standalone classes compatible with `torchvision.transforms.Compose`.
* **Option 2 (Integrated Dataset)**: Create a new `ShuffledImageDataset` class where the shuffling logic is embedded directly within the `__getitem__` method, controlled by `__init__` parameters (e.g., `shuffle_type`, `grid_size`).
* **Option 3 (Hybrid Approach)**: A combination of the above, such as a new `Dataset` that internally utilizes the custom transform classes.

**You must justify your final design choice with comments in the generated code.**

---

## 🧪 Unit Testing
Create a separate test file (e.g., `test_augmentations.py`) using the **`pytest`** framework to validate the functionality. The tests must be robust and cover the following cases:

1.  **Dimension Integrity**: Assert that the output image tensor from a transform has the exact same dimensions (Height, Width, Channels) as the input tensor.
2.  **Pixel Conservation**: For the shuffling transforms, verify that no pixel data is lost or altered. The set of all pixel values in the output image must be identical to the set in the input image. A good way to test this is to check that the sorted lists of all pixel values from the input and output are identical.
3.  **Determinism**: For the `GridShuffle` transform, test that when a random seed is set (`torch.manual_seed`), the transform produces the exact same shuffled image every time for a given input.
4.  **Dataset Integration**: Write a test to confirm that passing one of your new transforms to the `Dataset` class results in the transform being correctly applied when an item is retrieved.

---

## ✅ Deliverable - COMPLETED ✅
~~Produce a zip file containing two Python scripts~~:

1.  **Main Application Script** ✅ COMPLETED:
    * ✅ Contains the class definitions for chosen implementation (standalone transform classes following PyTorch conventions)
    * ✅ Includes comments explaining the code and design justification for Option 1 (Pure Transforms)
    * ✅ `HorizontalSwap` class implemented with support for multiple tensor formats
    * ✅ `GridShuffle` class implemented with M x N grid support (not just square grids)
    * ✅ `ImageDataset` class for integration with transforms
    * ~~Example usage block removed per user request~~

2.  **Test Script (`test_augmentations.py`)** ✅ COMPLETED:
    * ✅ Complete `pytest` suite with 22 tests covering all requirements
    * ✅ Dimension integrity tests for all tensor formats
    * ✅ Pixel conservation tests using sorted pixel value comparison
    * ✅ Determinism tests for GridShuffle with torch.manual_seed
    * ✅ Dataset integration tests confirming transforms work with Dataset class
    * ✅ All tests pass successfully

**Status**: Implementation completed and committed (commit 98f3286). All requirements fulfilled including enhanced M×N grid support for GridShuffle.
