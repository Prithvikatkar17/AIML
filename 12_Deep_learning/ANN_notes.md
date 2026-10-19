# Complete Guide to Artificial Neural Networks (ANN) for Regression & Classification
> **From Scratch to PyTorch Implementation**  
> *A comprehensive, step-by-step conceptual and practical guide covering both Continuous Regression (`02_ANN_regretion.ipynb`) and Multi-Class/Binary Classification (`03_ANN_Classification.ipynb`).*

---

## Table of Contents
1. [Introduction to Deep Learning: Regression & Classification](#1-introduction-to-deep-learning-regression--classification)
   - What is Regression?
   - What is Classification? (Binary vs. Multi-Class)
   - Master Comparison Matrix: Regression vs. Classification
   - Why use an Artificial Neural Network (ANN)?
2. [Anatomy of a Neuron and an ANN](#2-anatomy-of-a-neuron-and-an-ann)
   - The Artificial Neuron (Perceptron & Mathematical Formulation)
   - Layers in an ANN (Input, Hidden, Output)
   - Hidden Layer Activations (ReLU, LeakyReLU, Tanh)
   - The Golden Rules for the Output Layer
   - Visualizing Regression vs. Classification Network Architectures
3. [End-to-End ML & DL Preprocessing Pipeline](#3-end-to-end-ml--dl-preprocessing-pipeline)
   - Dataset 1 (Regression): Combined Cycle Power Plant
   - Dataset 2 (Classification): Date Fruit Varieties
   - Exploratory Data Analysis & Handling Missing Values
   - Feature & Target Separation ($X$ and $y$)
   - Target Encoding for Classification (`LabelEncoder`)
   - Train-Test Split (Standard vs. Stratified Split)
   - Feature Scaling (`StandardScaler` — Why Scaling is Vital)
4. [Transitioning from NumPy/Pandas to PyTorch](#4-transitioning-from-numpypandas-to-pytorch)
   - PyTorch Tensors & Data Types (`float32` vs. `long`)
   - Target Reshaping & Data Type Rules (2D `(N, 1)` for Regression vs. 1D `(N,)` for Classification)
   - PyTorch `TensorDataset` and `DataLoader`
   - Batching, Shuffling, and Workers
5. [Designing ANN Architectures in PyTorch](#5-designing-ann-architectures-in-pytorch)
   - Subclassing `nn.Module`
   - Regression ANN Architecture (Power Plant Model)
   - Multi-Class Classification ANN Architecture (Date Fruit 7-Class Model)
   - Binary Classification ANN Architecture
   - The Raw Logits Rule: Why NOT to use Softmax/Sigmoid inside the PyTorch Model
   - Trainable Parameter Count Calculations
6. [Loss Functions & Optimization](#6-loss-functions--optimization)
   - Regression Losses: MSE (`nn.MSELoss`), MAE (`nn.L1Loss`), Huber Loss
   - Classification Losses: Cross-Entropy (`nn.CrossEntropyLoss`), Binary Cross-Entropy (`nn.BCEWithLogitsLoss`)
   - Mathematical Formulations
   - Optimization: Gradient Descent & Adam Optimizer (`optim.Adam`)
   - Learning Rate Dynamics ($\alpha$)
7. [The PyTorch Training & Validation Loop](#7-the-pytorch-training--validation-loop)
   - The 5 Golden Steps of Deep Learning
   - Training Mode vs. Evaluation Mode (`model.train()` vs. `model.eval()`)
   - Disabling Gradients during Validation (`torch.no_grad()`)
   - Complete Training & Validation Loop for Regression
   - Complete Training & Validation Loop for Classification (Tracking Accuracy)
   - Model Checkpointing (`best_model.pt`)
8. [Evaluation & Performance Metrics](#8-evaluation--performance-metrics)
   - Evaluating Regression Models: MSE, RMSE, MAE, $R^2$ Score, Residuals
   - Evaluating Classification Models: Accuracy, Logits to Predictions (`torch.max`), Confusion Matrix, Classification Report
   - Multi-Class Probability Extraction (`torch.softmax`)
   - Diagnostic Guide for Loss and Accuracy Learning Curves
9. [Top 10 Beginner Pitfalls & How to Avoid Them](#9-top-10-beginner-pitfalls--how-to-avoid-them)
10. [Full Working Code References](#10-full-working-code-references)
    - Full Pipeline 1: Regression (Combined Cycle Power Plant)
    - Full Pipeline 2: Multi-Class Classification (Date Fruit Varieties)
    - Full Pipeline 3: Binary Classification Template
11. [Quick Reference Cheatsheet](#11-quick-reference-cheatsheet)

---

## 1. Introduction to Deep Learning: Regression & Classification

Supervised machine learning algorithms fall primarily into two foundational tasks: **Regression** and **Classification**. While both learn mappings from input features $\mathbf{x}$ to a target $y$, the nature of the target dictates the network architecture, loss function, activation functions, and evaluation metrics.

```
                         Supervised Learning with ANNs
                                       │
                ┌──────────────────────┴──────────────────────┐
                ▼                                             ▼
        [ Regression ]                                [ Classification ]
    Predict continuous values                      Predict discrete categories
    e.g., Power Output (450.2 MW)                  e.g., Date Fruit Type (BERHI)
    • Output: 1 Linear neuron                      • Binary: 1 Logit neuron (BCE)
    • Loss: MSE / MAE                              • Multi-class: C Logit neurons (CrossEntropy)
```

### What is Regression?
In **regression**, the target variable $y \in \mathbb{R}$ is a **continuous real number**. The model approximates an underlying mathematical function that can output any numerical value along a continuous scale.
- *Examples*: Predicting electrical energy output ($MW$), house prices, stock values, temperatures, engine fuel efficiency.
- *Notebook Reference*: `02_ANN_regretion.ipynb` predicts net hourly electrical power output ($PE$) ranging between $420.26$ and $495.76\text{ MW}$.

### What is Classification?
In **classification**, the target variable $y$ is a **discrete categorical label**. The model acts as a decision boundary separator that assigns inputs into predefined classes or probability distributions over classes.

1. **Binary Classification**: Exactly 2 mutually exclusive classes ($y \in \{0, 1\}$).
   - *Examples*: Spam vs. Ham, Fraud vs. Legitimate, Disease Positive vs. Negative.
2. **Multi-Class Classification**: More than 2 mutually exclusive classes ($y \in \{0, 1, 2, \dots, C-1\}$).
   - *Examples*: Handwritten digits ($0-9$), Fruit varieties, Blood groups.
   - *Notebook Reference*: `03_ANN_Classification.ipynb` classifies date fruits into one of $7$ distinct varieties: `BERHI`, `DEGLET`, `DOKOL`, `IRAQI`, `ROTANA`, `SAFAVI`, and `SOGAY`.
3. **Multi-Label Classification**: Multiple non-exclusive tags per sample (e.g., an article can be tagged with both "AI" and "Finance").

---

### Master Comparison Matrix: Regression vs. Classification

| Key Attribute | ANN Regression | Binary Classification | Multi-Class Classification ($C$ classes) |
| :--- | :--- | :--- | :--- |
| **Target Variable ($y$)** | Continuous number ($y \in \mathbb{R}$) | Binary label ($y \in \{0, 1\}$) | Class index ($y \in \{0, 1, \dots, C-1\}$) |
| **Target PyTorch Tensor** | `torch.float32`, shape `(N, 1)` | `torch.float32`, shape `(N, 1)` | `torch.long` (int64), shape `(N,)` |
| **Output Layer Neurons** | **1 neuron** | **1 neuron** | **$C$ neurons** (one per class) |
| **Model Output** | Raw continuous value $\hat{y}$ | Raw logit $z \in (-\infty, +\infty)$ | Raw logits vector $\mathbf{z} \in \mathbb{R}^C$ |
| **Loss Function** | `nn.MSELoss()`, `nn.L1Loss()` | `nn.BCEWithLogitsLoss()` | `nn.CrossEntropyLoss()` |
| **Activation for Probabilities**| None | Sigmoid: $\sigma(z) = \frac{1}{1 + e^{-z}}$ | Softmax: $\frac{e^{z_i}}{\sum_j e^{z_j}}$ |
| **Prediction Decision Rule** | $\hat{y} = \text{model}(x)$ | $\hat{y} = 1$ if $\sigma(z) \ge 0.5$ else $0$ | $\hat{y} = \text{argmax}(\mathbf{z})$ |
| **Evaluation Metrics** | MSE, RMSE, MAE, $R^2$ Score | Accuracy, Precision, Recall, F1, ROC-AUC | Multi-class Accuracy, Confusion Matrix, Macro/Weighted F1 |

---

### Why use an Artificial Neural Network (ANN)?
Traditional algorithms (Linear Regression, Logistic Regression, Naive Bayes) often fail when features interact non-linearly. 

Under the **Universal Approximation Theorem**, a feedforward neural network with at least one non-linear hidden layer and sufficient neurons can approximate any continuous function or decision boundary, enabling it to model complex non-linear physical systems (like thermodynamics in power plants or biometric shape distributions in date fruits).

---

## 2. Anatomy of a Neuron and an ANN

### The Artificial Neuron (Perceptron & Mathematical Formulation)
Every neuron computes a weighted sum of its inputs, adds an adjustable bias, and applies an activation function:

```
Inputs (x)         Weights (w)       Linear Combination         Activation        Output
  x₁  ─────────────>  w₁  ─────\
  x₂  ─────────────>  w₂  ───────> [  z = Σ(wᵢ·xᵢ) + b  ]  ────> [  a = f(z)  ]  ───>  (ŷ)
  x₃  ─────────────>  w₃  ─────/
                                     Bias (b)
```

$$\mathbf{z} = \mathbf{w}^T \mathbf{x} + b = \sum_{i=1}^d w_i x_i + b$$
$$a = f(z)$$

- $\mathbf{x} = [x_1, x_2, \dots, x_d]^T$: Input feature vector.
- $\mathbf{w} = [w_1, w_2, \dots, w_d]^T$: Trainable weight parameters (feature importances).
- $b$: Trainable bias scalar (shifts the activation threshold).
- $f(\cdot)$: Non-linear activation function.

---

### Hidden Layer Activations
Without non-linear activations, stacking linear layers reduces mathematically to a single linear transformation: $W_2(W_1 \mathbf{x} + b_1) + b_2 = (W_2 W_1)\mathbf{x} + (W_2 b_1 + b_2) = W_{new}\mathbf{x} + b_{new}$.

Common hidden layer activations:
1. **ReLU (Rectified Linear Unit)** — *Standard Default*:
   $$f(z) = \max(0, z)$$
   - *Pros*: Extremely fast to compute; avoids vanishing gradients for positive inputs; sparsity benefits.
2. **Leaky ReLU**:
   $$f(z) = \max(\alpha z, z) \quad (\alpha \approx 0.01)$$
   - *Pros*: Solves the "Dying ReLU" issue by allowing a small gradient when $z < 0$.
3. **Tanh (Hyperbolic Tangent)**:
   $$f(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}$$
   - *Pros*: Zero-centered between $[-1, 1]$. Useful in recurrent architectures or normalized latent representations.

---

### The Golden Rules for the Output Layer

Understanding how to terminate your network is the single most critical architectural decision in deep learning:

```
Output Layer Selection Decision Tree:
                      What task are you solving?
                                  │
         ┌────────────────────────┼────────────────────────┐
         ▼                        ▼                        ▼
    Regression          Binary Classification   Multi-Class Classification
         │                        │                        │
     1 Neuron                 1 Neuron                 C Neurons
         │                        │                        │
    Linear / None        Raw Logit (No Sigmoid)   Raw Logits (No Softmax)
         │                        │                        │
    nn.MSELoss()          nn.BCEWithLogitsLoss()    nn.CrossEntropyLoss()
```

1. **Regression Output Rule**:
   - **Units**: 1 neuron (or $k$ neurons for multi-target regression).
   - **Activation**: **None (Linear / Identity)**: $f(z) = z$.
   - *Why*: The model must be able to output any real number (e.g., $480.48\text{ MW}$). Putting Sigmoid would restrict predictions to $(0, 1)$, and ReLU would prevent negative numbers!
2. **Binary Classification Output Rule**:
   - **Units**: 1 neuron.
   - **Activation**: **Raw Logit (No activation in the model)**.
   - *Why*: Pass the raw logit directly to `nn.BCEWithLogitsLoss()`, which fuses Sigmoid + BCE internally for numerical stability.
3. **Multi-Class Classification Output Rule**:
   - **Units**: $C$ neurons (where $C$ is the number of classes, e.g., 7 for Date Fruit).
   - **Activation**: **Raw Logits (No Softmax in the model)**.
   - *Why*: Pass the raw logits directly to `nn.CrossEntropyLoss()`, which computes Softmax + Log + Negative Log-Likelihood internally with numerical stability optimizations.

---

### Visualizing Regression vs. Classification Network Architectures

#### Regression Architecture (`02_ANN_regretion.ipynb`)
Input features: 4 (`AT`, `V`, `AP`, `RH`) $\to$ Output: 1 continuous value (`PE`).
```
 Input Layer (4)        Hidden Layer 1 (6)      Hidden Layer 2 (6)     Output Layer (1)
    [ AT ]  =======>          ( O )                   ( O )
    [  V ]  =======>          ( O )                   ( O )
    [ AP ]  =======>          ( O )  ==============>  ( O )  =======>   [ PE (ŷ) ]
    [ RH ]  =======>          ( O )                   ( O )             (Linear output)
                              ( O )                   ( O )
                              ( O )                   ( O )
```

#### Multi-Class Classification Architecture (`03_ANN_Classification.ipynb`)
Input features: 34 biometric traits $\to$ Output: 7 class logits (one for each date fruit variety).
```
 Input Layer (34)       Hidden Layer 1 (64)     Hidden Layer 2 (64)    Output Layer (7 Logits)
   [ AREA ] ======>           ( O )                   ( O )  =======>   [ Logit 0: BERHI  ]
   [ PERI ] ======>           ( O )                   ( O )  =======>   [ Logit 1: DEGLET ]
   [ MAJR ] ======>           ( O )  ==============>  ( O )  =======>   [ Logit 2: DOKOL  ]
   [ .... ] ======>           ( : )                   ( : )  =======>   [ Logit 3: IRAQI  ]
   [ DAUB ] ======>           ( O )                   ( O )  =======>   [ Logit 4: ROTANA ]
                              ( O )                   ( O )  =======>   [ Logit 5: SAFAVI ]
                                                             =======>   [ Logit 6: SOGAY  ]
```

---

## 3. End-to-End ML & DL Preprocessing Pipeline

### The Two Reference Datasets

| Property | Dataset 1: Power Plant (`powerplant_data.csv`) | Dataset 2: Date Fruit (`DateFruit_Dataset.csv`) |
| :--- | :--- | :--- |
| **Task Type** | Continuous Regression | Multi-Class Classification |
| **Samples ($N$)** | 9,568 rows | 898 rows |
| **Features ($d$)** | 4 thermodynamic variables (`AT`, `V`, `AP`, `RH`) | 34 morphological & color features (AREA, EQDIASQ, Kurtosis, etc.) |
| **Target ($y$)** | `PE` (Net Electrical Energy: $420.26 - 495.76\text{ MW}$) | `Class` (7 varieties: BERHI, DEGLET, DOKOL, IRAQI, ROTANA, SAFAVI, SOGAY) |

---

### Step 1: Loading & Inspecting Data

```python
import pandas as pd
import numpy as np

# Regression
df_reg = pd.read_csv("powerplant_data.csv")
print("Regression Data Shape:", df_reg.shape) # (9568, 5)

# Classification
df_clf = pd.read_csv("DateFruit_Dataset.csv")
print("Classification Data Shape:", df_clf.shape) # (898, 35)
print("Classification Classes:", df_clf["Class"].unique())
```

---

### Step 2: Feature and Target Separation

```python
# Regression: Target is continuous 'PE'
X_reg = df_reg.drop("PE", axis=1)
y_reg = df_reg["PE"]

# Classification: Target is categorical string 'Class'
X_clf = df_clf.drop("Class", axis=1)
y_clf = df_clf["Class"]
```

---

### Step 3: Target Encoding for Classification (`LabelEncoder`)

PyTorch neural networks cannot process raw string labels (`"BERHI"`, `"DEGLET"`). We must encode them into zero-indexed integers: $0, 1, 2, \dots, C-1$.

```python
from sklearn.preprocessing import LabelEncoder

# Encode string labels to integers [0, 1, 2, 3, 4, 5, 6]
le = LabelEncoder()
y_clf_encoded = le.fit_transform(y_clf)

# Inspect mapping
for idx, class_name in enumerate(le.classes_):
    print(f"Index {idx} -> {class_name}")
```

> **Why NOT One-Hot Encode targets in PyTorch?**  
> PyTorch's `nn.CrossEntropyLoss()` expects 1D integer class indices (e.g., `tensor([0, 3, 1])`), **not** one-hot encoded vectors (e.g., `tensor([[1,0,0], [0,0,1], ...])`). Label encoding directly provides what PyTorch requires.

---

### Step 4: Train-Test Split (Standard vs. Stratified)

Splitting data before feature scaling prevents **data leakage** (information from the test set leaking into the training pipeline).

```python
from sklearn.model_selection import train_test_split

# 1. Regression Split (Standard random split)
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

# 2. Classification Split (Stratified split is recommended!)
# 'stratify=y' ensures training and test sets maintain identical class proportions
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_clf, y_clf_encoded, test_size=0.2, random_state=42, stratify=y_clf_encoded
)
```

---

### Step 5: Feature Scaling (`StandardScaler`)

Neural network weights are updated via gradient descent:
$$w \leftarrow w - \alpha \frac{\partial \mathcal{L}}{\partial w}$$

If feature $A$ is on a scale of $1000$ (e.g., Ambient Pressure `AP`) and feature $B$ is on a scale of $10$ (e.g., Temperature `AT`), the loss surface becomes an elongated, eccentric valley. Gradients oscillate wildly, forcing you to use an extremely small learning rate and slowing convergence.

**Standardization (Z-score Normalization)** centers features to mean $\mu = 0$ and standard deviation $\sigma = 1$:
$$x_{scaled} = \frac{x - \mu}{\sigma}$$

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

# Fit ONLY on training data; transform both train and test
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)
```

> **CRITICAL RULE**: Never call `fit_transform()` on test data! Always call `fit_transform()` on `X_train` and only `transform()` on `X_test`. Using test data to calculate mean and variance violates real-world inference integrity.

---

## 4. Transitioning from NumPy/Pandas to PyTorch

### PyTorch Tensors & Data Types (`float32` vs. `long`)

PyTorch operations require data to be encapsulated in `torch.Tensor` structures with exact data types:

```python
import torch
from torch.utils.data import TensorDataset, DataLoader

# =======================================================
# REGRESSION TENSORS (Float32 for both X and y)
# =======================================================
X_train_reg_t = torch.tensor(X_train_scaled_r, dtype=torch.float32)
X_test_reg_t  = torch.tensor(X_test_scaled_r,  dtype=torch.float32)

# Regression target MUST be 2D (N, 1) float32
y_train_reg_t = torch.tensor(y_train_r.values, dtype=torch.float32).view(-1, 1)
y_test_reg_t  = torch.tensor(y_test_r.values,  dtype=torch.float32).view(-1, 1)

# =======================================================
# CLASSIFICATION TENSORS (Float32 for X, Long for y)
# =======================================================
X_train_clf_t = torch.tensor(X_train_scaled_c, dtype=torch.float32)
X_test_clf_t  = torch.tensor(X_test_scaled_c,  dtype=torch.float32)

# Classification target MUST be 1D (N,) torch.long (integer class index)
y_train_clf_t = torch.tensor(y_train_c, dtype=torch.long)
y_test_clf_t  = torch.tensor(y_test_c,  dtype=torch.long)
```

---

### Target Reshaping & Data Type Rules

| Task | Target Tensor `dtype` | Required Target Shape | Why? |
| :--- | :--- | :--- | :--- |
| **Regression** | `torch.float32` | **2D: `(N, 1)`** | Avoids PyTorch's silent broadcasting trap in `nn.MSELoss()`. |
| **Binary Classification** | `torch.float32` | **2D: `(N, 1)`** | Aligns with 1D output logits in `nn.BCEWithLogitsLoss()`. |
| **Multi-Class Classification**| `torch.long` (int64) | **1D: `(N,)`** | `nn.CrossEntropyLoss()` expects 1D tensor containing class indices $0 \dots C-1$. |

> **The Silent MSE Broadcasting Bug**:  
> If model predictions are shaped `(32, 1)` and targets are shaped `(32)` (1D array), PyTorch will NOT throw an error! Instead, it broadcasts them into a `(32, 32)` matrix and computes the average of 1,024 differences. Always use `.view(-1, 1)` for regression targets!

---

### PyTorch `TensorDataset` and `DataLoader`

Instead of feeding the entire dataset into memory at once, we batch samples using `DataLoader`:

```python
batch_size = 32

# 1. Wrap tensors in TensorDataset
train_ds = TensorDataset(X_train_t, y_train_t)
test_ds  = TensorDataset(X_test_t,  y_test_t)

# 2. Wrap dataset in DataLoader
# shuffle=True for training to prevent cyclical pattern memorization
# shuffle=False for test/validation because order does not affect evaluation
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)
```

---

## 5. Designing ANN Architectures in PyTorch

In PyTorch, custom neural networks subclass `torch.nn.Module`.

### 1. Regression ANN Architecture (`02_ANN_regretion.ipynb`)
```python
import torch.nn as nn

class RegressionANN(nn.Module):
    def __init__(self, in_features=4):
        super(RegressionANN, self).__init__()
        self.model = nn.Sequential(
            # Hidden Layer 1: 4 inputs -> 6 neurons
            nn.Linear(in_features, 6),
            nn.ReLU(),
            # Hidden Layer 2: 6 neurons -> 6 neurons
            nn.Linear(6, 6),
            nn.ReLU(),
            # Output Layer: 6 neurons -> 1 continuous value (No activation!)
            nn.Linear(6, 1)
        )
        
    def forward(self, x):
        return self.model(x)
```

---

### 2. Multi-Class Classification ANN Architecture (`03_ANN_Classification.ipynb`)
```python
class MultiClassANN(nn.Module):
    def __init__(self, in_features=34, num_classes=7):
        super(MultiClassANN, self).__init__()
        self.model = nn.Sequential(
            # Hidden Layer 1: 34 inputs -> 64 neurons
            nn.Linear(in_features, 64),
            nn.ReLU(),
            # Hidden Layer 2: 64 neurons -> 64 neurons
            nn.Linear(64, 64),
            nn.ReLU(),
            # Output Layer: 64 neurons -> 7 raw logits (No Softmax!)
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x):
        return self.model(x)
```

---

### 3. Binary Classification ANN Architecture
```python
class BinaryANN(nn.Module):
    def __init__(self, in_features):
        super(BinaryANN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(in_features, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            # Output Layer: 1 raw logit (No Sigmoid!)
            nn.Linear(16, 1)
        )
        
    def forward(self, x):
        return self.model(x)
```

---

### The Raw Logits Rule: Why NOT to use Softmax/Sigmoid inside the Model

```
Beginner Trap:
Model Output ───> [ nn.Softmax() ] ───> Logits Probabilities ───> [ nn.CrossEntropyLoss() ]
                                                                       ▲
                                            ERROR: CrossEntropyLoss applies Softmax AGAIN!

Correct PyTorch Practice:
Model Output ───> Raw Logits z ───────────────────────────────> [ nn.CrossEntropyLoss() ]
                  (Internally computes LogSoftmax + NLLLoss via Log-Sum-Exp trick)
```

- **Numerical Stability**: When probabilities $p$ approach $0$ or $1$, calculating $\log(p)$ directly can cause underflow ($-\infty$) or overflow (`NaN`). 
- PyTorch's `nn.CrossEntropyLoss()` and `nn.BCEWithLogitsLoss()` combine the activation with the loss using the mathematically stabilized **Log-Sum-Exp** trick.
- Only apply `torch.softmax(outputs, dim=1)` or `torch.sigmoid(outputs)` when you are doing **inference** to display probabilities to end users.

---

### Trainable Parameter Count Calculations

For any fully connected layer connecting $n_{in}$ inputs to $n_{out}$ neurons:
$$\text{Parameters} = (n_{in} \times n_{out}) + n_{out} \quad [\text{weights} + \text{biases}]$$

#### Power Plant Regression ANN:
- Layer 1 (`4 -> 6`): $(4 \times 6) + 6 = 24 + 6 = 30$
- Layer 2 (`6 -> 6`): $(6 \times 6) + 6 = 36 + 6 = 42$
- Layer 3 (`6 -> 1`): $(6 \times 1) + 1 = 6 + 1 = 7$
- **Total Parameters**: $30 + 42 + 7 = 79$

#### Date Fruit Classification ANN:
- Layer 1 (`34 -> 64`): $(34 \times 64) + 64 = 2176 + 64 = 2,240$
- Layer 2 (`64 -> 64`): $(64 \times 64) + 64 = 4096 + 64 = 4,160$
- Layer 3 (`64 -> 7`): $(64 \times 7) + 7 = 448 + 7 = 455$
- **Total Parameters**: $2240 + 4160 + 455 = 6,855$

---

## 6. Loss Functions & Optimization

### 1. Regression Loss Functions

#### Mean Squared Error (MSE) — *Primary Default*:
Penalizes larger errors quadratically:
$$\text{MSE} = \frac{1}{N} \sum_{i=1}^N (\hat{y}_i - y_i)^2$$
```python
criterion = nn.MSELoss()
```

#### Mean Absolute Error (MAE / L1 Loss):
Penalizes errors linearly; more robust to outliers:
$$\text{MAE} = \frac{1}{N} \sum_{i=1}^N |\hat{y}_i - y_i|$$
```python
criterion = nn.L1Loss()
```

---

### 2. Classification Loss Functions

#### Multi-Class: Cross-Entropy Loss (`nn.CrossEntropyLoss`)
Measures the divergence between the true one-hot distribution and predicted softmax probabilities:
$$\mathcal{L}_{CE} = - \sum_{c=1}^C y_c \log(p_c) \quad \text{where } p_c = \frac{e^{z_c}}{\sum_{j=1}^C e^{z_j}}$$
In PyTorch:
```python
criterion = nn.CrossEntropyLoss()
# Input: Model raw logits (batch_size, num_classes)
# Target: True class integer indices (batch_size) of dtype torch.long
```

#### Binary: Binary Cross-Entropy with Logits (`nn.BCEWithLogitsLoss`)
$$\mathcal{L}_{BCE} = - [y \log \sigma(z) + (1 - y) \log(1 - \sigma(z))]$$
In PyTorch:
```python
criterion = nn.BCEWithLogitsLoss()
# Input: Model raw logit (batch_size, 1)
# Target: Float tensor (batch_size, 1) with values 0.0 or 1.0
```

---

### 3. Optimization: Adam Optimizer

Gradient Descent updates model parameters $\mathbf{W}$ in the opposite direction of the gradient:
$$\mathbf{W} \leftarrow \mathbf{W} - \alpha \nabla_{\mathbf{W}} \mathcal{L}$$

**Adam (Adaptive Moment Estimation)** combines two powerful momentum concepts:
1. **First Moment (Momentum)**: Keeps a running exponentially decaying average of past gradients to accelerate through flat areas.
2. **Second Moment (RMSprop)**: Keeps a running average of squared gradients to scale learning rates per parameter, damping erratic oscillations.

```python
import torch.optim as optim

optimizer = optim.Adam(model.parameters(), lr=0.001)
```

---

## 7. The PyTorch Training & Validation Loop

### The 5 Golden Steps of Deep Learning

Inside every training batch `(xb, yb)`:
```
           ┌──────────────────────────────────────────────┐
           │              Training Batch (xb)             │
           └──────────────────────┬───────────────────────┘
                                  │
                                  ▼
           Step 1: optimizer.zero_grad()  (Reset accumulated gradients)
                                  │
                                  ▼
           Step 2: outputs = model(xb)    (Forward pass: compute predictions)
                                  │
                                  ▼
           Step 3: loss = criterion(...)  (Compute batch loss)
                                  │
                                  ▼
           Step 4: loss.backward()        (Backward pass: compute gradients via chain rule)
                                  │
                                  ▼
           Step 5: optimizer.step()       (Weight update: adjust weights via optimizer)
```

---

### Complete Training & Validation Loop for Regression (`02_ANN_regretion.ipynb`)

```python
epochs = 100
train_losses, val_losses = [], []
best_val_loss = float("inf")

for epoch in range(epochs):
    # ------------------ TRAINING PHASE ------------------
    model.train()
    running_loss = 0.0
    
    for xb, yb in train_loader:
        optimizer.zero_grad()            # 1. Reset gradients
        preds = model(xb)                # 2. Forward pass
        loss = criterion(preds, yb)      # 3. Compute loss (MSE)
        loss.backward()                  # 4. Backward pass
        optimizer.step()                 # 5. Update weights
        
        running_loss += loss.item()
        
    train_loss = running_loss / len(train_loader)
    train_losses.append(train_loss)
    
    # ------------------ VALIDATION PHASE ------------------
    model.eval()
    running_val_loss = 0.0
    
    with torch.no_grad():
        for xb, yb in test_loader:
            preds = model(xb)
            val_loss = criterion(preds, yb)
            running_val_loss += val_loss.item()
            
    val_loss = running_val_loss / len(test_loader)
    val_losses.append(val_loss)
    
    # Checkpointing: Save best weights
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "best_regression_model.pt")
        
    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"Epoch [{epoch+1:3d}/{epochs}] -> Train MSE: {train_loss:.2f} | Val MSE: {val_loss:.2f}")
```

---

### Complete Training & Validation Loop for Classification (`03_ANN_Classification.ipynb`)

In classification, we track **both Loss and Accuracy**:

```python
epochs = 100
train_losses, val_losses = [], []
best_val_acc = 0.0

for epoch in range(epochs):
    # ------------------ TRAINING PHASE ------------------
    model.train()
    running_loss = 0.0
    correct_train, total_train = 0, 0
    
    for xb, yb in train_loader:
        optimizer.zero_grad()
        outputs = model(xb)                     # Raw logits (batch_size, 7)
        loss = criterion(outputs, yb)           # CrossEntropyLoss
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        # Track training accuracy
        _, preds = torch.max(outputs, 1)        # Get index of highest logit
        correct_train += (preds == yb).sum().item()
        total_train += yb.size(0)
        
    train_loss = running_loss / len(train_loader)
    train_acc = (correct_train / total_train) * 100.0
    
    # ------------------ VALIDATION PHASE ------------------
    model.eval()
    running_val_loss = 0.0
    correct_val, total_val = 0, 0
    
    with torch.no_grad():
        for xb, yb in test_loader:
            outputs = model(xb)
            val_loss = criterion(outputs, yb)
            running_val_loss += val_loss.item()
            
            _, preds = torch.max(outputs, 1)
            correct_val += (preds == yb).sum().item()
            total_val += yb.size(0)
            
    val_loss = running_val_loss / len(test_loader)
    val_acc = (correct_val / total_val) * 100.0
    
    # Checkpoint based on highest validation accuracy
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "best_classification_model.pt")
        
    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"Epoch [{epoch+1:3d}/{epochs}] | Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")
```

---

## 8. Evaluation & Performance Metrics

### 1. Evaluating Regression Models

```python
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Load best checkpoint
model.load_state_dict(torch.load("best_regression_model.pt"))
model.eval()

with torch.no_grad():
    y_pred_tensor = model(X_test_reg_t)
    y_pred = y_pred_tensor.numpy()
    y_true = y_test_r.values

mse = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)

print(f"Test MSE  : {mse:.2f}")
print(f"Test RMSE : {rmse:.2f} MW")
print(f"Test MAE  : {mae:.2f} MW")
print(f"R² Score  : {r2:.4f}") # ~0.9311 (93.11% variance explained)
```

- **$R^2$ Score (Coefficient of Determination)**:
  $$R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$$
  Measures the proportion of target variance explained by inputs. $1.0$ is perfect; $0$ means no better than predicting the mean.

---

### 2. Evaluating Classification Models

```python
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load best checkpoint
model.load_state_dict(torch.load("best_classification_model.pt"))
model.eval()

with torch.no_grad():
    logits = model(X_test_clf_t)                 # (N, 7)
    _, predicted_classes = torch.max(logits, 1)  # Class indices (N,)
    probs = torch.softmax(logits, dim=1)        # Normalized probabilities (N, 7)

y_pred_np = predicted_classes.numpy()
y_true_np = y_test_c

# Accuracy
accuracy = (y_pred_np == y_true_np).mean() * 100.0
print(f"Test Accuracy: {accuracy:.2f}%") # Reaches ~93.33% on Date Fruit

# Detailed Classification Report (Precision, Recall, F1 per class)
print("\nClassification Report:")
print(classification_report(y_true_np, y_pred_np, target_names=le.classes_))
```

#### Extracting Top Prediction Probabilities:
```python
# Show sample predictions with probability confidence
sample_probs, sample_preds = torch.max(probs[:5], dim=1)
for i in range(5):
    class_name = le.classes_[sample_preds[i].item()]
    confidence = sample_probs[i].item() * 100
    print(f"Sample {i+1}: Predicted '{class_name}' with {confidence:.1f}% confidence")
```

---

### Diagnostic Guide for Loss and Accuracy Learning Curves

```
1. Ideal Fit:
   Loss                                 Accuracy
   │ \                                  │      /──────
   │  \──── Train                       │     /────── Val
   │   \─── Val                         │    /─────── Train
   └─────────── Epochs                  └─────────── Epochs

2. Overfitting (High Variance):
   Loss                                 Accuracy
   │ \                                  │      /────── Train
   │  \────── Train                     │     /
   │   \───────/\── Val (Climbs up!)    │    /──────── Val (Plateaus early)
   └─────────── Epochs                  └─────────── Epochs
   Fixes: Add Dropout (nn.Dropout(0.3)), add Weight Decay (L2), simplify architecture.

3. Underfitting (High Bias):
   Loss                                 Accuracy
   │ ────────── Train (Stays high)      │ ────────── Train (Stays low)
   │ ────────── Val   (Stays high)      │ ────────── Val   (Stays low)
   └─────────── Epochs                  └─────────── Epochs
   Fixes: Increase model capacity (more neurons/layers), train longer, tune learning rate.
```

---

## 9. Top 10 Beginner Pitfalls & How to Avoid Them

### 1. Target Shape Mismatch in Regression (`(N,)` vs. `(N, 1)`)
- **Problem**: Passing a 1D tensor for $y$ while the model outputs 2D `(N, 1)`.
- **Symptom**: `UserWarning: Using a target size... that is different to the input size...` and erratic MSE values.
- **Fix**: Always reshape: `y_tensor = torch.tensor(y.values, dtype=torch.float32).view(-1, 1)`.

### 2. Wrong Target DataType in Classification (`float32` instead of `long`)
- **Problem**: Passing float targets to `nn.CrossEntropyLoss()`.
- **Symptom**: `RuntimeError: Expected object of scalar type Long but got Float`.
- **Fix**: Use `y_tensor = torch.tensor(y, dtype=torch.long)`.

### 3. The Double Softmax Trap
- **Problem**: Adding `nn.Softmax()` to the model output AND using `nn.CrossEntropyLoss()`.
- **Symptom**: Extremely slow learning or stuck accuracy because gradients saturate through double softmax.
- **Fix**: Output **raw logits** from the model; let `nn.CrossEntropyLoss()` handle Softmax internally.

### 4. Activating the Output Layer in Regression
- **Problem**: Putting `nn.ReLU()` or `nn.Sigmoid()` after the final regression neuron.
- **Symptom**: Model cannot output negative values (ReLU) or outputs are trapped inside $(0, 1)$ (Sigmoid).
- **Fix**: Leave the regression output layer linear: `nn.Linear(hidden_dim, 1)`.

### 5. Forgetting `optimizer.zero_grad()`
- **Problem**: PyTorch accumulates gradients by default. If you omit `zero_grad()`, gradients from past batches add up indefinitely.
- **Symptom**: Model diverges; loss explodes to `NaN` or `inf`.
- **Fix**: Call `optimizer.zero_grad()` before `loss.backward()` in every training step.

### 6. Data Leakage with `StandardScaler`
- **Problem**: Calling `scaler.fit(df)` on the whole dataset or calling `scaler.fit_transform(X_test)`.
- **Fix**: Only fit on training data:
  ```python
  X_train_scaled = scaler.fit_transform(X_train)
  X_test_scaled  = scaler.transform(X_test)
  ```

### 7. Evaluating with Gradient Tracking Enabled
- **Problem**: Omitting `with torch.no_grad():` during validation and inference.
- **Symptom**: GPU/RAM out of memory (OOM) because PyTorch builds backward computation graphs during testing.
- **Fix**: Always evaluate inside `with torch.no_grad():` and set `model.eval()`.

### 8. Forgetting to Set `model.eval()` / `model.train()`
- **Problem**: Not toggling model mode when using Dropout or BatchNorm.
- **Symptom**: Inconsistent inference predictions or degraded validation scores.
- **Fix**: Call `model.train()` before the training loop and `model.eval()` before evaluation.

### 9. Omitting Class Stratification in Imbalanced Classification
- **Problem**: Using standard `train_test_split` without `stratify=y` on imbalanced classes.
- **Symptom**: Rare classes may have zero samples in the test set.
- **Fix**: `train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)`.

### 10. Using Only Accuracy on Imbalanced Datasets
- **Problem**: In a dataset with 95% Class A and 5% Class B, a dummy model predicting always "A" gets 95% accuracy.
- **Fix**: Use Precision, Recall, Macro-F1 score, and Confusion Matrices (`classification_report`).

---

## 10. Full Working Code References

### Pipeline 1: Regression (Power Plant Energy Prediction - `02_ANN_regretion.ipynb`)

```python
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error

# 1. Load Data
df = pd.read_csv("powerplant_data.csv")
X = df.drop("PE", axis=1)
y = df["PE"]

# 2. Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Scale Features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# 4. Prepare PyTorch Tensors & DataLoaders
X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_t = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)  # 2D (N, 1)

X_test_t  = torch.tensor(X_test_scaled,  dtype=torch.float32)
y_test_t  = torch.tensor(y_test.values,  dtype=torch.float32).view(-1, 1)

train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=32, shuffle=True)
test_loader  = DataLoader(TensorDataset(X_test_t,  y_test_t),  batch_size=32, shuffle=False)

# 5. Define Regression Architecture
class RegressionANN(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 6),
            nn.ReLU(),
            nn.Linear(6, 6),
            nn.ReLU(),
            nn.Linear(6, 1)  # Linear output
        )
    def forward(self, x):
        return self.net(x)

model = RegressionANN(in_features=X_train.shape[1])
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 6. Training Loop with Checkpointing
epochs = 100
best_val_loss = float("inf")

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for xb, yb in test_loader:
            preds = model(xb)
            val_loss += criterion(preds, yb).item()
    val_loss /= len(test_loader)
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "best_regression_model.pt")

# 7. Evaluate
model.load_state_dict(torch.load("best_regression_model.pt"))
model.eval()
with torch.no_grad():
    final_preds = model(X_test_t)
    r2 = r2_score(y_test, final_preds.numpy())
    mse = mean_squared_error(y_test, final_preds.numpy())

print(f"Final Test MSE : {mse:.4f}")
print(f"Final R² Score : {r2:.4f}")
```

---

### Pipeline 2: Multi-Class Classification (Date Fruit Varieties - `03_ANN_Classification.ipynb`)

```python
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score

# 1. Load Data
df = pd.read_csv("DateFruit_Dataset.csv")
X = df.drop("Class", axis=1)
y = df["Class"]

# 2. Encode Labels (String -> Integer 0..C-1)
le = LabelEncoder()
y_encoded = le.fit_transform(y)
num_classes = len(le.classes_)

# 3. Stratified Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# 4. Scale Features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# 5. Prepare PyTorch Tensors & DataLoaders
X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.long)          # 1D (N,) int64

X_test_t  = torch.tensor(X_test_scaled,  dtype=torch.float32)
y_test_t  = torch.tensor(y_test, dtype=torch.long)

train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=32, shuffle=True)
test_loader  = DataLoader(TensorDataset(X_test_t,  y_test_t),  batch_size=32, shuffle=False)

# 6. Define Multi-Class Architecture (Matches 03_ANN_Classification.ipynb)
class MultiClassANN(nn.Module):
    def __init__(self, in_features, num_classes):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)  # Raw logits (No Softmax!)
        )
    def forward(self, x):
        return self.model(x)

model = MultiClassANN(in_features=X_train.shape[1], num_classes=num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 7. Training Loop with Accuracy Tracking
epochs = 100
best_val_acc = 0.0

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        outputs = model(xb)
        loss = criterion(outputs, yb)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
    # Validation
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for xb, yb in test_loader:
            outputs = model(xb)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == yb).sum().item()
            total += yb.size(0)
            
    val_acc = (correct / total) * 100.0
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "best_classification_model.pt")

# 8. Evaluation
model.load_state_dict(torch.load("best_classification_model.pt"))
model.eval()
with torch.no_grad():
    test_outputs = model(X_test_t)
    _, test_preds = torch.max(test_outputs, 1)

print(f"Final Test Accuracy: {accuracy_score(y_test, test_preds.numpy()) * 100:.2f}%\n")
print(classification_report(y_test, test_preds.numpy(), target_names=le.classes_))
```

---

### Pipeline 3: Binary Classification Template

```python
import torch
import torch.nn as nn
import torch.optim as optim

# Architecture
class BinaryClassificationANN(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)  # 1 raw logit output (No Sigmoid!)
        )
    def forward(self, x):
        return self.net(x)

# Setup
model = BinaryClassificationANN(in_features=10)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Evaluating Predictions during Inference
model.eval()
with torch.no_grad():
    raw_logits = model(X_test_tensor)                     # Shape: (N, 1)
    probabilities = torch.sigmoid(raw_logits)              # Range: (0, 1)
    predicted_classes = (probabilities >= 0.5).int()       # 0 or 1
```

---

## 11. Quick Reference Cheatsheet

| Component | Regression | Binary Classification | Multi-Class Classification |
| :--- | :--- | :--- | :--- |
| **Output Layer Neurons** | `nn.Linear(H, 1)` | `nn.Linear(H, 1)` | `nn.Linear(H, C)` |
| **Output Activation** | None (Linear) | None (Raw Logit) | None (Raw Logits) |
| **Target Tensor Shape** | `(N, 1)` | `(N, 1)` | `(N,)` |
| **Target Tensor Type** | `torch.float32` | `torch.float32` | `torch.long` |
| **PyTorch Loss** | `nn.MSELoss()` | `nn.BCEWithLogitsLoss()` | `nn.CrossEntropyLoss()` |
| **Compute Class Predictions**| Direct output: `model(x)` | `(torch.sigmoid(logits) >= 0.5).int()` | `torch.max(logits, dim=1)[1]` |
| **Get Probabilities** | N/A | `torch.sigmoid(logits)` | `torch.softmax(logits, dim=1)` |
| **Evaluation Metrics** | MSE, RMSE, MAE, $R^2$ | Accuracy, ROC-AUC, F1 | Accuracy, Confusion Matrix, Report |

---
*Created as a comprehensive master study reference for `02_ANN_regretion.ipynb` and `03_ANN_Classification.ipynb` in Deep Learning.*
