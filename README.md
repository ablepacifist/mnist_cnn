
```markdown
# A4P4: Manual and High-Level CNN Implementation

## Overview

This project is a solution to Assignment 4, Problem 4 for an AI course. The task was to implement a convolutional neural network (CNN) using a high-level library like TensorFlow. However, I went the extra mile by implementing a manual version of the CNN to better understand the fundamentals of neural networks. The manual version includes both CPU and GPU-accelerated implementations using CuPy.

The project allows you to choose between:

- **High-Level Version**: A faster, optimized implementation using TensorFlow.
- **Manual Version**: A fully custom implementation of a CNN, with options for CPU or GPU acceleration.

## How to Run the Code

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/A4P4.git
cd A4P4
```

### 2. Install Required Libraries

Ensure you have [Python 3.11.9](https://www.python.org/downloads/) installed. Then, install the required libraries using `pip`:

```bash
pip install numpy cupy tensorflow matplotlib
```

> **Note:** Depending on your system’s CUDA configuration, you might need to install a specific version of CuPy. For example, if you're using CUDA 12.8, install using `pip install cupy-cuda12x`.

### 3. Check GPU Compatibility

*Ensure that you have a compatible GPU with CUDA installed.* This project uses CUDA V12.8.93. You can verify your GPU setup:

- From the command line:
  ```bash
  nvcc --version
  ```
- Alternatively, in Python:
  ```python
  import cupy as cp
  print("Number of CUDA devices:", cp.cuda.runtime.getDeviceCount())
  ```

### 4. Run the Program

Execute the main script:

```bash
python A4P4.py
```

During execution, you will be prompted to choose between the manual and high-level versions:
- Enter `manual` for the custom implementation.
- Enter `highlevel` for the TensorFlow implementation.

If you choose manual, you will be further prompted to select between the CPU and GPU versions.

## Libraries Used

- **CuPy (13.4.1)**: For GPU-accelerated NumPy-like operations in the manual implementation.
- **TensorFlow (2.19.0)**: For the high-level CNN implementation.
- **Matplotlib (3.10.1)**: For visualizing training progress and validation loss.
- **CUDA (V12.8.93)**: Required for GPU acceleration with CuPy.

## Flow of Information Between Functions

The manual implementation follows the forward pass, backward pass, and parameter update paradigm, as described in [The UDL Book](https://udlbook.github.io/udlbook/). Here's a breakdown:

### Data Preprocessing
- **`load_and_preprocess`**:  
  Loads the MNIST dataset, normalizes the pixel values, and splits the data into training, validation, and test sets.

### Forward Pass
- **Convolutional Layers:**  
  - **`conv_forward`**: Applies convolutional filters to extract spatial features.
- **Activation Functions:**  
  - **`relu_forward`**: Applies the ReLU activation function for non-linearity.
- **Pooling Layers:**  
  - **`maxpool_forward`**: Reduces spatial dimensions while preserving important features.
- **Flattening:**  
  - **`flatten_forward`**: Converts 2D feature maps into a 1D vector.
- **Fully Connected Layer:**  
  - **`fc_forward`**: Computes output scores for each class.

### Loss Calculation
- **`softmax`**: Converts scores into probabilities.
- **`cross_entropy_loss`**: Computes the loss comparing predicted probabilities with true labels.

### Backward Pass
- **Fully Connected Layer:**  
  - **`fc_backward`**: Computes gradients for weights, biases, and inputs.
- **Flattening:**  
  - **`flatten_backward`**: Reshapes gradients back to original dimensions.
- **Pooling Layers:**  
  - **`maxpool_backward`**: Propagates gradients through the pooling layer.
- **Activation Functions:**  
  - **`relu_backward`**: Backpropagates gradients through the ReLU layer.
- **Convolutional Layers:**  
  - **`conv_backward`**: Computes gradients for convolutional filters, biases, and inputs.

### Parameter Updates
- **`update_params`**:  
  Updates weights and biases using the computed gradients and learning rate.

### Training Loop
- **`train_network`**:  
  Iteratively performs the forward pass, loss computation, backward pass, and parameter updates for each batch.

### Evaluation
- **`predict`**:  
  Uses the trained model to predict labels for the test set.
- **`accuracy`**:  
  Computes the accuracy of predictions compared to the true labels.

### Visualization
- **`plot_validation_loss`**:  
  Plots the validation loss for different data augmentation techniques.

## Key Features

- **Manual Implementation:**
  - Fully custom CNN including convolution, pooling, activations, and fully connected layers.
  - Supports both CPU and GPU execution using CuPy.
- **High-Level Implementation:**
  - Optimized CNN using TensorFlow for faster training and evaluation.
- **Data Augmentation:**
  - Functions to add Gaussian noise (`add_gaussian_noise`) and sharpen images (`sharpen_images`).
- **Visualization:**
  - Displays sample images from the MNIST dataset and plots the validation loss from experiments.

## Sources and Inspiration

- [UDL Book](https://udlbook.github.io/udlbook/): Provided the foundational concepts and terminology for neural networks.
- [AlexNet on Papers with Code](https://paperswithcode.com/method/alexnet): Inspired the architecture of the convolutional layers.
- *Pattern Recognition and Machine Learning* by Christopher Bishop: Deepened understanding of backpropagation and optimization.

## About the Author

**Name**: Alex Dyakin  
**University**: University of Manitoba  
**Date**: March 25, 2025  

This project was completed as part of Assignment 4, Problem 4 for my AI course at the University of Manitoba. Although the assignment only required a high-level implementation, I chose to implement a manual version to deeply understand the fundamentals of neural networks.

### Professors
- **AI Professor**: Dr. Sadaf Salehkalaibar  
- **Machine Learning Professor**: Dr. Christopher Henry  

### Contact Information
- **School Email**: [dyakina@myumanitoba.ca](mailto:dyakina@myumanitoba.ca)  
- **Personal Email**: [alexpdyak32@gmail.com](mailto:alexpdyak32@gmail.com)  


## Notes

- **Performance**: The manual implementation is slower than the high-level version, especially on the CPU. For faster execution, use the GPU version or the high-level implementation.
- **Dataset**: The project is designed to work with the MNIST dataset but can be extended to other datasets with minor modifications.
- **GPU Requirements**: Ensure your system meets the necessary CUDA requirements to use the GPU-accelerated version.
```
