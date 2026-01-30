"""
DETAILED EXPLANATION: How The CNN Training Works

This file walks through each step of training with detailed explanations.
Run this alongside train.py to understand what's happening.
"""

# ============================================================================
# PART 1: INITIALIZATION
# ============================================================================
"""
When train.py starts, it does this:

1. SET DEVICE
   device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   
   Why? To use GPU if available (50-100x faster than CPU)
   Output: "Using device: cuda" or "Using device: cpu"

2. LOAD DATA
   train_loader, val_loader, test_loader, num_classes = get_data_loaders(...)
   
   What happens:
   - CIFAR-10 dataset automatically downloads (~170 MB)
   - 50,000 training images split into:
     * 45,000 for training
     * 5,000 for validation
   - 10,000 test images kept separate
   - Images preprocessed: normalized to mean=0, std=1
   
   Why 3 datasets?
   - TRAIN: Used to update model weights
   - VALIDATION: Used to monitor performance, detect overfitting
   - TEST: Never seen during training, true final evaluation

3. CREATE MODEL
   model = SimpleCNN(num_classes=10)
   
   Architecture:
   Input (32×32×3)
      ↓
   Conv2d(3→32) + BatchNorm + ReLU
      ↓
   MaxPool(2×2)  [32×32 → 16×16]
      ↓
   Conv2d(32→64) + BatchNorm + ReLU
      ↓
   MaxPool(2×2)  [16×16 → 8×8]
      ↓
   Flatten  [8×8×64 = 4096 features]
      ↓
   Linear(4096→128) + ReLU + Dropout
      ↓
   Linear(128→10)  [10 classes]

4. SETUP TRAINING
   criterion = nn.CrossEntropyLoss()  # Loss function
   optimizer = optim.Adam(...)        # How to update weights
   
   Why CrossEntropyLoss?
   - Designed for multi-class classification
   - Combines log-softmax with negative log-likelihood
   - More numerically stable than alternatives
   
   Why Adam optimizer?
   - Adaptive learning rates per parameter
   - Momentum helps escape local minima
   - Works well with mini-batch training
   - Better than vanilla SGD for this task
"""

# ============================================================================
# PART 2: ONE EPOCH EXPLAINED
# ============================================================================
"""
An EPOCH = one complete pass through all training data

For each epoch:

STEP 1: TRAIN
--------
for images, labels in train_loader:  # Load batch of 32 images
    
    # Move to GPU/CPU
    images, labels = images.to(device), labels.to(device)
    
    # FORWARD PASS: Images → Model → Predictions
    outputs = model(images)  # Shape: [32, 10] (32 images, 10 classes)
    
    What forward pass does:
    Input image (32×32×3):
        → Conv layer learns 32 patterns (edges, colors)
        → MaxPool shrinks to 16×16
        → Conv learns 64 more complex patterns
        → MaxPool shrinks to 8×8
        → Flatten to vector
        → Fully connected layers combine patterns for classification
        → Output: confidence scores for each of 10 classes
    
    Example output: [2.1, -0.5, 0.8, 1.2, ..., -1.3]
    Interpretation: slightly confident it's class 0 (airplane)
    
    # CALCULATE LOSS: How wrong are we?
    loss = criterion(outputs, labels)  # Single number
    
    What loss does:
    - Compares predicted classes with true labels
    - Assigns a penalty for being wrong
    - Larger penalty for being more wrong
    
    Example:
    - Predicted: airplane (correct)   → loss ≈ 0.1
    - Predicted: dog (wrong)          → loss ≈ 2.3
    - Predicted: truck (very wrong)   → loss ≈ 5.8
    
    # BACKWARD PASS: Compute gradients
    optimizer.zero_grad()  # Clear old gradients (important!)
    loss.backward()        # Compute ∂loss/∂weight for each weight
    
    What backward does:
    - Uses chain rule (backpropagation)
    - Computes how much each weight contributed to the loss
    - Gradient tells us which direction to adjust weights
    
    Example gradient: ∂loss/∂w = -0.5
    Interpretation: increasing this weight by 0.01 would decrease loss by 0.005
    
    # UPDATE WEIGHTS: Gradient descent step
    optimizer.step()  # Update weights using gradients
    
    What step does:
    For each weight w:
        new_w = old_w - learning_rate * gradient
    
    This reduces the loss for this batch
    
    total_loss += loss.item()  # Accumulate for epoch average

average_train_loss = total_loss / num_batches

Output: "Train Loss: 2.1234"
Interpretation: On average, predictions were this wrong

STEP 2: VALIDATE
--------
model.eval()  # Disable dropout, use batch statistics

for images, labels in val_loader:
    with torch.no_grad():  # Don't compute gradients (faster, uses less memory)
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Calculate accuracy
        _, predicted = torch.max(outputs, 1)  # Get class with highest confidence
        accuracy = (predicted == labels).sum() / total
        
        total_loss += loss.item()

val_loss = total_loss / num_batches
val_accuracy = correct / total * 100

Output: 
    "Val Loss: 1.8976"
    "Val Accuracy: 45.32%"

Interpretation:
- Validation loss shows if model generalizes well
- If val_loss >> train_loss: overfitting (model memorized training)
- If val_accuracy increasing: model learning
- If val_accuracy plateaus: may need more training or different architecture
"""

# ============================================================================
# PART 3: WHAT'S ACTUALLY HAPPENING UNDER THE HOOD
# ============================================================================
"""
Inside a Convolutional Layer:

Input: 32×32×3 image

    Each filter is a 3×3×3 grid of weights:
    
    Filter 1 (detects horizontal edges):
    [  0.2  -0.5   0.2 ]
    [  0.2  -0.5   0.2 ]  (this is repeated for 3 color channels)
    [  0.2  -0.5   0.2 ]
    
    Sliding window operation:
    
    Position (0,0):
    Input region × Filter weights → Sum → Activate with ReLU
    
    Position (0,1):
    (slide right, repeat)
    
    Position (1,0):
    (slide down, repeat)
    
    Result: 32×32 feature map showing where edges are detected
    
With 32 filters → 32 feature maps (32 channels)

After MaxPool (2×2):
    16×16×32  (resolution halved, patterns preserved)

Inside a Fully Connected Layer:

Input: 4096 features
Weight matrix: 4096 × 128

For each output:
    output = sum(input * weight) + bias
    
    With 4096 inputs and 128 outputs:
    128,000+ individual computations
    
    These are learned weights that map visual patterns to class features

Why Batch Normalization?

Without BatchNorm:
    Layer 1 outputs: Mean=50, Std=30  (Large scale)
    Layer 2 inputs:  Large scale causes gradient issues
    → Training is slow, unstable

With BatchNorm:
    Each layer normalized to: Mean=0, Std=1
    → Gradients stable
    → Can use higher learning rate
    → Trains faster and more stable
"""

# ============================================================================
# PART 4: MONITORING TRAINING
# ============================================================================
"""
After each epoch, you should see:

Epoch 1/10
  Train Loss: 2.3021
  Val Loss: 2.1234
  Val Accuracy: 18.45%

What this means:

1. FIRST EPOCH
   - Train Loss: 2.3 means average prediction is very wrong
     (Random guessing gives ~2.3 for 10 classes)
   - Val Accuracy: 18% is below 10% random, learning!

2. MIDDLE EPOCHS
   - Train Loss: 1.2 → drops further
   - Val Loss: 1.0 → network learning generalizable patterns
   - Val Accuracy: 65% → meaningful improvement

3. LATER EPOCHS
   - Train Loss: 0.5 → model memorizing training data
   - Val Loss: 0.8 → still reasonable generalization
   - Val Accuracy: 78% → good final performance
   
4. OVERFITTING DETECTION
   - If train_loss << val_loss (0.1 vs 1.5)
   - And accuracy gap is large (95% train vs 65% val)
   - Then model is overfitting
   - Solution: use more dropout, less complex model, or more data

5. UNDERFITTING DETECTION
   - Both train and val accuracy stay low (< 50%)
   - Loss not decreasing
   - Solution: train longer, use more capacity, simpler model
"""

# ============================================================================
# PART 5: FINAL TESTING
# ============================================================================
"""
After 10 epochs:

model.load_state_dict(torch.load('best_model.pth'))

This loads the weights from the epoch with best validation accuracy
(Not the final epoch, which might have overfitted)

Then evaluate on test set:

test_accuracy = 78.34%

This is the TRUE measure of model performance
(Because test data was never seen during training or validation)

If test_accuracy ≈ val_accuracy: Good, model generalizes well
If test_accuracy << val_accuracy: Problem, test set might be different
"""

# ============================================================================
# PART 6: UNDERSTANDING THE NUMBERS
# ============================================================================
"""
Example Training Run:

Epoch 1/10
  Train Loss: 2.2967  ← Random guessing
  Val Loss: 2.1845
  Val Accuracy: 20.34%  ← Barely better than random (10%)

Epoch 2/10
  Train Loss: 1.8234
  Val Loss: 1.7123
  Val Accuracy: 38.45%

Epoch 5/10
  Train Loss: 0.9234
  Val Loss: 1.0456
  Val Accuracy: 64.23%

Epoch 10/10
  Train Loss: 0.4123
  Val Loss: 0.9876
  Val Accuracy: 78.45%

Test Accuracy: 78.34%

Analysis:
- Training loss decreases continuously ✓
- Validation loss decreases then stabilizes ✓
- Validation accuracy increases ✓
- Test accuracy close to val accuracy ✓
- Model learned well with minimal overfitting ✓
"""

# ============================================================================
# PART 7: KEY CONCEPTS SUMMARY
# ============================================================================
"""
GRADIENT DESCENT
  direction = -gradient  # Negative because we want to reduce loss
  weight = weight + learning_rate * direction
  
  Intuition: downhill on a loss landscape

BACKPROPAGATION
  Computes gradients for all weights efficiently
  Chain rule: ∂L/∂w1 = ∂L/∂output × ∂output/∂w1
  
BATCH TRAINING
  Why not single sample?
  - Noisy gradients from single sample
  - Can't parallelize well
  - Batches are more stable

DATA AUGMENTATION
  Why RandomCrop and RandomFlip?
  - Artificial variation simulates real-world variety
  - Prevents overfitting
  - Increases effective training data

TRAIN/VAL/TEST
  Train: Update weights
  Val: Monitor generalization, save best model
  Test: Final performance estimate
  
  Why separate?
  - Can't evaluate on data used for optimization
  - Validation detects overfitting
  - Test measures true generalization
"""

# ============================================================================
# CONCLUSION
# ============================================================================
"""
Training process:
1. Initialize model with random weights
2. For each epoch:
   a) For each batch:
      - Predict on batch
      - Calculate loss
      - Backpropagate gradients
      - Update weights
   b) Evaluate on validation set
   c) Save if best so far
3. Load best model
4. Evaluate on test set

Result: A model that can classify new images with ~78% accuracy!

This process is the foundation of deep learning.
Same concepts apply to:
- Image segmentation
- Object detection
- Natural language processing
- Audio processing
- And more...
"""
