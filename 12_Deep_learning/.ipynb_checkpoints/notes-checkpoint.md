# Complete Guide to Artificial Neural Networks (ANN) for Regression
> **From Scratch to PyTorch Implementation**  
> *A beginner-to-advanced, step-by-step conceptual and practical guide based on `02_ANN_regretion.ipynb`.*

---

## Table of Contents
1. [Introduction to Regression & Neural Networks](#1-introduction-to-regression--neural-networks)
   - What is Regression?
   - Regression vs. Classification in Deep Learning
   - Why use an Artificial Neural Network (ANN) for Regression?
2. [Anatomy of a Neuron and an ANN](#2-anatomy-of-a-neuron-and-an-ann)
   - The Biological Inspiration
   - The Artificial Neuron (Perceptron)
   - Mathematical Formulation ($z = \mathbf{w}^T \mathbf{x} + b$)
   - Layers in an ANN (Input, Hidden, Output)
   - Activation Functions (Why Non-Linearity Matters)
   - The Output Layer Rule for Regression
3. [End-to-End Machine Learning & Deep Learning Pipeline](#3-end-to-end-machine-learning--deep-learning-pipeline)
   - Dataset Overview (Combined Cycle Power Plant)
   - Exploratory Data Analysis & Handling Missing Values
   - Feature & Target Separation ($X$ and $y$)
   - Train-Test Split (Preventing Data Leakage)
   - Feature Scaling (Why Standardization is Crucial)
4. [Transitioning from NumPy/Pandas to PyTorch](#4-transitioning-from-numpypandas-to-pytorch)
   - Tensors vs. NumPy Arrays
   - Target Reshaping Pitfall (`.view(-1, 1)`)
   - PyTorch `TensorDataset` and `DataLoader`
   - Batching & Shuffling
5. [Designing the ANN Architecture in PyTorch](#5-designing-the-ann-architecture-in-pytorch)
   - Subclassing `nn.Module`
   - Defining Layers in `__init__`
   - Connecting Layers in `forward()`
   - Visualizing the Architecture
6. [Loss Functions & Optimization](#6-loss-functions--optimization)
   - Mean Squared Error (MSE) Loss
   - Other Regression Losses (MAE, RMSE, Huber)
   - Optimization: Gradient Descent & Adam Optimizer
   - The Role of Learning Rate ($\alpha$)
7. [The PyTorch Training & Validation Loop](#7-the-pytorch-training--validation-loop)
   - The 5 Golden Steps of Deep Learning Training
   - Training Mode vs. Evaluation Mode (`model.train()` vs `model.eval()`)
   - Disabling Gradients during Validation (`torch.no_grad()`)
   - Saving and Loading the Best Model Checkpoint
8. [Evaluation & Performance Metrics](#8-evaluation--performance-metrics)
   - Evaluating on Unseen Test Data
   - Mean Squared Error (MSE)
   - Coefficient of Determination ($R^2$ Score)
   - Plotting Loss Curves (Detecting Overfitting & Underfitting)
   - Comparing Actual vs. Predicted Values
9. [Common Beginner Pitfalls & How to Avoid Them](#9-common-beginner-pitfalls--how-to-avoid-them)
10. [Full Working Code Reference](#10-full-working-code-reference)

---

## 1. Introduction to Regression & Neural Networks

### What is Regression?
In Machine Learning, **regression** is a supervised learning task where the target output $y$ is a **continuous real number** (e.g., predicting temperature, house prices, stock prices, or electrical power output).

Contrast this with **classification**, where the goal is to predict discrete categories or labels (e.g., Spam vs. Not Spam, Cat vs. Dog).

### Regression vs. Classification in Deep Learning

| Attribute | ANN Regression | ANN Classification |
| :--- | :--- | :--- |
| **Target Variable ($y$)** | Continuous number ($\mathbb{R}$) | Discrete class label ($0, 1$ or classes $0 \dots C-1$) |
| **Output Layer Neurons** | **1 neuron** (for single-value prediction) | 1 neuron (Binary) or $C$ neurons (Multi-class) |
| **Output Activation** | **None (Linear / Identity)** | Sigmoid (Binary) or Softmax (Multi-class) |
| **Loss Function** | `nn.MSELoss()`, `nn.L1Loss()` | `nn.BCEWithLogitsLoss()`, `nn.CrossEntropyLoss()` |
| **Evaluation Metrics** | MSE, RMSE, MAE, $R^2$ Score | Accuracy, Precision, Recall, F1-Score, ROC-AUC |

### Why use an Artificial Neural Network (ANN) for Regression?
Traditional regression algorithms (like Ordinary Least Squares Linear Regression) assume linear relationships between features and the target. When datasets contain **complex, non-linear, multi-dimensional interactions**, linear models struggle.

An ANN acts as a **Universal Function Approximator**: with non-linear activation functions and hidden layers, it can learn virtually any continuous mapping from inputs to outputs.

---

## 2. Anatomy of a Neuron and an ANN

### The Artificial Neuron (Perceptron)
A single artificial neuron takes multiple inputs, multiplies each input by an adjustable **weight**, sums them up, adds a **bias**, and passes the result through an **activation function**.

```
Input Features            Weights           Summation & Bias         Activation           Output
   (x₁)  ------------->  (w₁)  -----\
   (x₂)  ------------->  (w₂)  ------> [  z = Σ(wᵢ·xᵢ) + b  ]  --->  [  a = σ(z)  ]  --->   (ŷ)
   (x₃)  ------------->  (w₃)  -----/
                                      Bias (b)
```

### Mathematical Formulation
For an input vector $\mathbf{x} = [x_1, x_2, \dots, x_d]^T$:

$$z = \mathbf{w}^T \mathbf{x} + b = \sum_{i=1}^d w_i x_i + b$$

Where:
- $\mathbf{w}$ = weights vector (determines the strength/importance of each input).
- $b$ = bias scalar (allows shifting the activation threshold up or down).
- $z$ = net linear combination.
- $a = f(z)$ = output after the activation function $f$.

### Layers in an ANN
An ANN (or Multi-Layer Perceptron - MLP) consists of:
1. **Input Layer**: Receives raw or preprocessed feature values (does not perform calculations).
2. **Hidden Layer(s)**: One or more layers of neurons that extract abstract patterns and non-linear representations.
3. **Output Layer**: Produces the final prediction $\hat{y}$.

```
  Input Layer          Hidden Layer 1         Hidden Layer 2         Output Layer
   (4 inputs)             (6 units)              (6 units)             (1 output)
     [ AT ]  =======>       ( O )                  ( O )       
     [  V ]  =======>       ( O )                  ( O )       
     [ AP ]  =======>       ( O )  =============>  ( O )  =======>     [ PE (ŷ) ]
     [ RH ]  =======>       ( O )                  ( O )       
                            ( O )                  ( O )       
                            ( O )                  ( O )       
```

### Activation Functions (Why Non-Linearity Matters)
If we do not apply an activation function, stacking multiple linear layers together is mathematically equivalent to just a single linear transformation:

$$W_2(W_1 x + b_1) + b_2 = (W_2 W_1)x + (W_2 b_1 + b_2) = W_{new} x + b_{new}$$

Activation functions introduce **curvature and non-linearity**, allowing the network to capture complex physical processes.

Common activation functions:
1. **ReLU (Rectified Linear Unit)**:
   $$f(z) = \max(0, z)$$
   - *Pros*: Fast to compute, prevents vanishing gradients for positive inputs, default choice for hidden layers.
2. **Sigmoid**:
   $$f(z) = \frac{1}{1 + e^{-z}}$$
   - Squeezes numbers between $(0, 1)$. Mainly used for binary probabilities, rarely in hidden layers today.
3. **Tanh (Hyperbolic Tangent)**:
   $$f(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}$$
   - Squeezes numbers between $(-1, 1)$. Zero-centered.

### The Output Layer Rule for Regression
> **Critical Rule**: In regression, the output neuron must **NOT** have an activation function (or have an identity/linear activation $f(z) = z$), unless the target is strictly bounded in a specific range like $[0, 1]$.
> 
> *Reason*: If you put a Sigmoid activation on the output layer, your network can never predict any value outside $(0, 1)$. For energy output like $480.48\text{ MW}$, the prediction would be completely ruined!

---

## 3. End-to-End Machine Learning & Deep Learning Pipeline

### Dataset Overview (Combined Cycle Power Plant)
In `02_ANN_regretion.ipynb`, the problem is predicting net hourly electrical energy output from a Combined Cycle Power Plant using thermodynamic variables:

- **Features ($X$)**:
  - `AT`: Ambient Temperature ($1.81^\circ\text{C} - 37.11^\circ\text{C}$)
  - `V`: Exhaust Vacuum ($25.36 - 81.56\text{ cm Hg}$)
  - `AP`: Ambient Pressure ($992.89 - 1033.30\text{ millibar}$)
  - `RH`: Relative Humidity ($25.56\% - 100.16\%$)
- **Target ($y$)**:
  - `PE`: Net hourly electrical energy output ($420.26 - 495.76\text{ MW}$)

### Step 1: Loading & Inspecting Data
```python
import pandas as pd
import numpy as np

# Load CSV dataset
df = pd.read_csv("powerplant_data.csv")

# Inspect shape and missing values
print("Dataset shape:", df.shape) # (9568 rows, 5 columns)
print("Missing values:\n", df.isnull().sum())
```

### Step 2: Feature and Target Separation
```python
X = df.drop("PE", axis=1)  # 4 features: AT, V, AP, RH
y = df["PE"]               # Target: PE
```

### Step 3: Train-Test Split
Always split data **before** scaling to prevent **data leakage** (where statistics from the test set leak into the training process).

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```
- **Training Set (80%)**: 7,654 samples.
- **Testing Set (20%)**: 1,914 samples.

### Step 4: Feature Scaling (StandardScaler)
Neural networks are extremely sensitive to feature scales. If `AP` is $\approx 1000$ and `AT` is $\approx 10$, the weights associated with `AP` will receive massive gradients compared to `AT`, causing erratic training or slow convergence.

**Standardization (Z-score Normalization)**:
$$x_{scaled} = \frac{x - \mu}{\sigma}$$

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

# 1. fit_transform() on train: calculates mean & std, then transforms
X_train_scaled = scaler.fit_transform(X_train)

# 2. transform() on test: uses the TRAIN mean & std (NEVER fit on test!)
X_test_scaled = scaler.transform(X_test)
```

> **Why `fit_transform` on train but only `transform` on test?**  
> If you call `fit_transform` on the test set, you calculate a new mean and variance specific to the test set. In real-world deployment, your model must handle incoming test points using the distribution parameters learned during training.

---

## 4. Transitioning from NumPy/Pandas to PyTorch

### Converting to Tensors & The Target Reshaping Rule
PyTorch requires inputs and targets to be PyTorch `Tensor` objects with `torch.float32` precision.

```python
import torch
from torch.utils.data import TensorDataset, DataLoader

# Convert features
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
X_test_tensor  = torch.tensor(X_test_scaled,  dtype=torch.float32)

# Convert targets (NOTE: .view(-1, 1) is MANDATORY!)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
y_test_tensor  = torch.tensor(y_test.values,  dtype=torch.float32).view(-1, 1)
```

> **WARNING: The 1D vs 2D Target Trap**  
> If your model output has shape `(batch_size, 1)` and your target has shape `(batch_size)` (1D array), PyTorch's `nn.MSELoss()` will silently **broadcast** the dimensions into a `(batch_size, batch_size)` matrix!  
> This calculates an unintended loss and triggers:  
> `UserWarning: Using a target size (torch.Size([N])) that is different to the input size (torch.Size([N, 1]))...`  
> Always use `.view(-1, 1)` or `.unsqueeze(1)` to ensure the target is 2D `(N, 1)`.

### PyTorch `TensorDataset` and `DataLoader`
Instead of passing all 7,654 rows at once into memory, we break the data into **mini-batches**:

```python
# 1. Wrap tensors in a TensorDataset
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset  = TensorDataset(X_test_tensor,  y_test_tensor)

# 2. Create DataLoaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)
```

- **`batch_size=32`**: The model processes 32 samples per update step.
- **`shuffle=True` (train)**: Randomizes sample order every epoch so the network does not memorize sequence patterns.
- **`shuffle=False` (test)**: No need to shuffle when evaluating.

---

## 5. Designing the ANN Architecture in PyTorch

In PyTorch, custom neural networks subclass `torch.nn.Module`.

```python
import torch.nn as nn

class ANN(nn.Module):
    def __init__(self, in_features=4):
        super(ANN, self).__init__()
        
        # Define sequential stack of layers
        self.model = nn.Sequential(
            # 1st Hidden Layer: 4 inputs -> 6 neurons
            nn.Linear(in_features, 6),
            nn.ReLU(),
            
            # 2nd Hidden Layer: 6 neurons -> 6 neurons
            nn.Linear(6, 6),
            nn.ReLU(),
            
            # Output Layer: 6 neurons -> 1 continuous prediction
            nn.Linear(6, 1)
            # Notice: No activation function here!
        )
        
    def forward(self, x):
        return self.model(x)

# Instantiate the model
model = ANN(in_features=4)
print(model)
```

### Parameter Count Breakdown
For a layer $l$ connecting $n_{in}$ inputs to $n_{out}$ neurons:
$$\text{Parameters} = (n_{in} \times n_{out}) + n_{out} \quad (\text{weights} + \text{biases})$$

- **Layer 1**: $(4 \times 6) + 6 = 30$
- **Layer 2**: $(6 \times 6) + 6 = 42$
- **Layer 3**: $(6 \times 1) + 1 = 7$
- **Total Trainable Parameters**: $30 + 42 + 7 = 79$

---

## 6. Loss Functions & Optimization

### 1. Loss Function: Mean Squared Error (MSE)
Measures the average squared difference between predicted values $\hat{y}_i$ and actual values $y_i$:

$$\text{MSE} = \frac{1}{N} \sum_{i=1}^N (\hat{y}_i - y_i)^2$$

In PyTorch:
```python
criterion = nn.MSELoss()
```

### 2. Optimizer: Adam (Adaptive Moment Estimation)
Gradient Descent updates parameters in the direction that minimizes the loss:
$$w \leftarrow w - \alpha \frac{\partial \mathcal{L}}{\partial w}$$

**Adam** combines:
- **Momentum**: Accelerates gradients in the relevant direction (smooths oscillations).
- **RMSprop**: Adapts learning rates individually for each parameter based on running historical gradients.

```python
import torch.optim as optim

optimizer = optim.Adam(model.parameters(), lr=0.001)
```

---

## 7. The PyTorch Training & Validation Loop

### The 5 Golden Steps of Deep Learning
In every training step over a batch `(xb, yb)`:
1. `optimizer.zero_grad()`: Reset gradients from the previous step (PyTorch accumulates gradients by default).
2. `outputs = model(xb)`: **Forward Pass** — compute predictions.
3. `loss = criterion(outputs, yb)`: Compute error.
4. `loss.backward()`: **Backward Pass** — calculate gradients via chain rule.
5. `optimizer.step()`: **Weight Update** — adjust model weights using gradients.

```
                  ┌──────────────────────────────────────────────┐
                  │              Training Batch (xb)             │
                  └──────────────────────┬───────────────────────┘
                                         │
                                         ▼
                                   [ Forward Pass ]
                                 outputs = model(xb)
                                         │
                                         ▼
                                  [ Compute Loss ]
                             loss = criterion(outputs, yb)
                                         │
                                         ▼
                                  [ Backward Pass ]
                                   loss.backward()
                                         │
                                         ▼
                                  [ Update Weights ]
                                  optimizer.step()
                                         │
                                         ▼
                               [ Clear Gradients ]
                               optimizer.zero_grad()
```

### Complete Training & Validation Code with Checkpointing
```python
epochs = 100
train_losses = []
val_losses = []
best_val_loss = float("inf")

for epoch in range(epochs):
    # ==========================
    # 1. TRAINING PHASE
    # ==========================
    model.train()  # Set model to training mode
    running_loss = 0.0
    
    for xb, yb in train_loader:
        optimizer.zero_grad()            # 1. Reset gradients
        outputs = model(xb)              # 2. Forward pass
        loss = criterion(outputs, yb)    # 3. Compute batch loss
        loss.backward()                  # 4. Backward pass (backpropagation)
        optimizer.step()                 # 5. Update weights
        
        running_loss += loss.item()      # Accumulate loss as float
        
    epoch_train_loss = running_loss / len(train_loader)
    train_losses.append(epoch_train_loss)
    
    # ==========================
    # 2. VALIDATION PHASE
    # ==========================
    model.eval()  # Set model to evaluation mode
    running_val_loss = 0.0
    
    with torch.no_grad():  # Disable gradient computation for speed & memory
        for xb, yb in test_loader:
            outputs = model(xb)
            val_loss = criterion(outputs, yb)
            running_val_loss += val_loss.item()
            
    epoch_val_loss = running_val_loss / len(test_loader)
    val_losses.append(epoch_val_loss)
    
    # Log progress every 10 epochs
    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"Epoch [{epoch+1:3d}/{epochs}] => Train Loss: {epoch_train_loss:.2f} | Val Loss: {epoch_val_loss:.2f}")
        
    # ==========================
    # 3. MODEL CHECKPOINTING
    # ==========================
    if epoch_val_loss < best_val_loss:
        best_val_loss = epoch_val_loss
        torch.save(model.state_dict(), "best_model.pt")
```

---

## 8. Evaluation & Performance Metrics

### Loading the Best Model
```python
# Restore the best weights discovered during training
model.load_state_dict(torch.load("best_model.pt"))
model.eval()
```

### Measuring Performance Metrics

```python
from sklearn.metrics import r2_score, mean_squared_error

with torch.no_grad():
    train_pred = model(X_train_tensor)
    test_pred  = model(X_test_tensor)

# Calculate Mean Squared Error
train_mse = mean_squared_error(y_train, train_pred.numpy())
test_mse  = mean_squared_error(y_test,  test_pred.numpy())
r2        = r2_score(y_test, test_pred.numpy())

print(f"Train MSE: {train_mse:.2f}")
print(f"Test MSE : {test_mse:.2f}")
print(f"R² Score : {r2:.4f}")
```

- **In our notebook experiment:**
  - $R^2 \approx 0.9311$ ($93.11\%$ of the variance in electrical output is explained by ambient variables).
  - Test MSE dropped from $\approx 200,000$ to $\approx 19.7\text{ MW}^2$.
  - Root Mean Squared Error (RMSE) $\approx \sqrt{19.7} \approx 4.44\text{ MW}$, meaning on average predictions are within $\pm 4.44\text{ MW}$ of actual power output!

### Visualizing Loss Curves
```python
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))
plt.plot(train_losses, label="Training Loss (MSE)", color="royalblue")
plt.plot(val_losses, label="Validation Loss (MSE)", color="darkorange", linestyle="--")
plt.xlabel("Epochs")
plt.ylabel("Loss (MSE)")
plt.title("Learning Curves: Training vs Validation Loss")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

#### Diagnostic Guide for Loss Curves:
1. **Good Fit**: Both training and validation losses smoothly decline and converge close together.
2. **Underfitting**: Both training and validation losses remain high or plateau early. (Solution: Increase model capacity, add more neurons/layers, train longer).
3. **Overfitting**: Training loss keeps dropping, but validation loss starts climbing back up. (Solution: Add dropout, weight decay / L2 regularization, or stop training earlier).

### Comparing Actual vs. Predicted Values
```python
results_df = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": test_pred.squeeze().numpy()
})
results_df["Difference (Error)"] = results_df["Predicted"] - results_df["Actual"]
print(results_df.head(10))
```

---

## 9. Common Beginner Pitfalls & How to Avoid Them

### 1. Shape Mismatch in Target (`(N,)` vs `(N, 1)`)
- **Problem**: Passing a 1D tensor for $y$ while model outputs 2D `(N, 1)`.
- **Symptom**: `UserWarning: Using a target size... that is different to the input size...` and weirdly huge MSE loss values.
- **Fix**: Use `y_tensor = torch.tensor(y.values, dtype=torch.float32).view(-1, 1)`.

### 2. Forgetting `optimizer.zero_grad()`
- **Problem**: PyTorch accumulates gradients across iterations.
- **Symptom**: Model diverges, loss explodes to `NaN` or `inf`.
- **Fix**: Call `optimizer.zero_grad()` at the start of every batch loop.

### 3. Activating the Output Layer
- **Problem**: Putting `nn.ReLU()` or `nn.Sigmoid()` after the final layer.
- **Symptom**: Inability to predict negative values (ReLU) or values $> 1$ (Sigmoid).
- **Fix**: Leave the output layer linear: `nn.Linear(hidden_dim, 1)`.

### 4. Data Leakage with Scaler
- **Problem**: Fitting the scaler on the entire dataset `scaler.fit(df)` before splitting, or calling `scaler.fit_transform(X_test)`.
- **Fix**: Only fit on training data:
  ```python
  X_train_scaled = scaler.fit_transform(X_train)
  X_test_scaled  = scaler.transform(X_test)
  ```

### 5. Evaluating with Gradient Tracking Enabled
- **Problem**: Running inference or validation without `with torch.no_grad():`.
- **Symptom**: High RAM/VRAM usage and slower evaluation.
- **Fix**: Wrap validation and inference blocks in `with torch.no_grad():` and call `model.eval()`.

---

## 10. Full Working Code Reference

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
y_train_t = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)

X_test_t  = torch.tensor(X_test_scaled,  dtype=torch.float32)
y_test_t  = torch.tensor(y_test.values,  dtype=torch.float32).view(-1, 1)

train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=32, shuffle=True)
test_loader  = DataLoader(TensorDataset(X_test_t,  y_test_t),  batch_size=32, shuffle=False)

# 5. Define ANN Model
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

# 6. Training Loop
epochs = 100
best_val_loss = float("inf")

for epoch in range(epochs):
    model.train()
    running_train_loss = 0.0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()
        running_train_loss += loss.item()
        
    # Validation
    model.eval()
    running_val_loss = 0.0
    with torch.no_grad():
        for xb, yb in test_loader:
            preds = model(xb)
            loss = criterion(preds, yb)
            running_val_loss += loss.item()
            
    val_loss = running_val_loss / len(test_loader)
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "best_model.pt")

# 7. Evaluate Best Model
model.load_state_dict(torch.load("best_model.pt"))
model.eval()
with torch.no_grad():
    final_preds = model(X_test_t)
    r2 = r2_score(y_test, final_preds.numpy())
    mse = mean_squared_error(y_test, final_preds.numpy())

print(f"Final Test MSE : {mse:.4f}")
print(f"Final R² Score : {r2:.4f}")
```

---
*Created as study reference for `02_ANN_regretion.ipynb` in Deep Learning.*
