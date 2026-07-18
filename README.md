# MNIST Optimizer Comparison: SGD vs Adam using PyTorch

## Overview

This project compares the performance of two popular optimization algorithms—**Stochastic Gradient Descent (SGD)** and **Adam**—while training a simple Multi-Layer Perceptron (MLP) on the MNIST handwritten digit dataset.

The model is trained twice using identical settings, with only the optimizer changed. The training loss and execution time are recorded, and the results are visualized using Matplotlib.

---

## Features

- Uses the MNIST handwritten digit dataset.
- Implements a simple Multi-Layer Perceptron (MLP).
- Compares **SGD** and **Adam** optimizers.
- Records training loss after each epoch.
- Measures training time for both optimizers.
- Plots loss curves for visual comparison.

---

## Requirements

- Python 3.x
- PyTorch
- Torchvision
- Matplotlib

Install the required packages:

```bash
pip install torch torchvision matplotlib
```

---

## Dataset

The project uses the **MNIST** dataset provided by `torchvision`.

- Training Images: 60,000
- Testing Images: 10,000
- Image Size: 28 × 28 pixels
- Number of Classes: 10 (Digits 0–9)

Dataset loading:

```python
transform = transforms.ToTensor()

train_data = datasets.MNIST(
    root="./data",
    train=True,
    transform=transform,
    download=True
)
```

---

## Model Architecture

The neural network consists of two fully connected layers.

```
Input Image (1 × 28 × 28)
          │
          ▼
Flatten
(784 Features)
          │
          ▼
Linear (784 → 128)
          │
          ▼
ReLU
          │
          ▼
Linear (128 → 10)
          │
          ▼
Output Scores (10 Classes)
```

---

## Model Summary

| Layer | Output Size |
|--------|------------:|
| Flatten | 784 |
| Linear | 128 |
| ReLU | 128 |
| Linear | 10 |

---

## Optimizers Compared

### SGD (Stochastic Gradient Descent)

```python
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.001
)
```

**Characteristics**

- Simple optimization algorithm.
- Uses the gradient to update weights.
- Generally slower to converge.
- May require careful learning rate tuning.

---

### Adam Optimizer

```python
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)
```

**Characteristics**

- Adaptive learning rate.
- Faster convergence.
- Widely used in deep learning.
- Combines Momentum and RMSProp ideas.

---

## Loss Function

The project uses CrossEntropy Loss for multi-class classification.

```python
criterion = nn.CrossEntropyLoss()
```

---

## Training Configuration

| Parameter | Value |
|-----------|------:|
| Epochs | 5 |
| Batch Size | 32 |
| Learning Rate | 0.001 |
| Loss Function | CrossEntropyLoss |
| Optimizers | SGD, Adam |

---

## Training Workflow

For each optimizer:

1. Create a new MLP model.
2. Select the optimizer (SGD or Adam).
3. Train the model for 5 epochs.
4. Compute the loss for each batch.
5. Perform backpropagation.
6. Update model parameters.
7. Store the loss after each epoch.
8. Measure total training time.

---

## Visualization

After training, Matplotlib plots the loss curves of both optimizers.

```python
plt.plot(sgd_losses, label="SGD")
plt.plot(adam_losses, label="Adam")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()
```

The legend also displays the total training time for each optimizer.

---

## Example Output

```
SGD losses:
[1.25, 0.94, 0.73, 0.59, 0.48]

Adam losses:
[0.18, 0.09, 0.05, 0.03, 0.02]
```

Example timing:

```
SGD Training Time : 18.42 s
Adam Training Time: 11.35 s
```

> Actual loss values and training times depend on your hardware and PyTorch version.

---

## Expected Graph

The graph compares the training loss over epochs.

- **X-axis:** Epoch
- **Y-axis:** Loss

Typically:

- SGD shows a gradual decrease in loss.
- Adam decreases the loss more quickly and reaches a lower value within the same number of epochs.

---

## Project Structure

```
.
├── data/
│   └── MNIST/
├── optimizer_comparison.py
├── README.md
└── requirements.txt
```

---

## Concepts Covered

- PyTorch Neural Networks
- Multi-Layer Perceptron (MLP)
- MNIST Dataset
- DataLoader
- Forward Pass
- Backpropagation
- CrossEntropy Loss
- SGD Optimizer
- Adam Optimizer
- Training Loop
- Performance Measurement
- Loss Visualization with Matplotlib

---

## Applications

This project is useful for learning:

- Image classification using neural networks.
- Differences between optimization algorithms.
- Measuring training performance.
- Visualizing model convergence.
- PyTorch training workflow.

---

## Possible Improvements

- Add validation and test accuracy.
- Plot both loss and accuracy.
- Use GPU acceleration (CUDA).
- Compare additional optimizers such as RMSProp or AdamW.
- Save and reload trained models.
- Replace the MLP with a Convolutional Neural Network (CNN).
- Record average epoch loss instead of only the last batch loss.

---

## Author

Developed using **PyTorch** to demonstrate and compare the performance of **SGD** and **Adam** optimizers on the MNIST handwritten digit classification task.
