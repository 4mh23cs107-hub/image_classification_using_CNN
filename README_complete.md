# Image Classification using CNN

A beginner-friendly implementation of a Convolutional Neural Network (CNN) for image classification on the CIFAR-10 dataset. This project focuses on understanding **data loading**, **model training**, and **basic evaluation**.

## Project Structure

```
├── model.py              # CNN model architecture definition
├── data_loader.py        # Data loading and preprocessing
├── train.py              # Main training script
├── inference.py          # Model inference and prediction utilities
├── requirements.txt      # Python dependencies
└── README.md             # Documentation
```

## Features

✅ **Simple CNN Architecture** - Easy to understand convolutional layers with batch normalization  
✅ **CIFAR-10 Dataset** - Automatically downloads 60,000 labeled 32x32 images  
✅ **Data Augmentation** - Random cropping and flipping for better generalization  
✅ **Train/Validation Split** - 90/10 split with separate test evaluation  
✅ **Progress Tracking** - Visual progress bars and detailed epoch statistics  
✅ **Model Checkpointing** - Automatically saves the best model during training  
✅ **Training Visualization** - Plots loss curves and accuracy trends  
✅ **Inference Module** - Tools for making predictions on new images  

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the Model

```bash
python train.py
```

That's it! The script will:
- Download CIFAR-10 dataset automatically
- Train the CNN for 10 epochs
- Save the best model and training plots
- Report final test accuracy

**Expected Results:**
- Validation Accuracy: ~75-80%
- Test Accuracy: ~75-80%
- Training Time: 2-5 minutes

### 3. Make Predictions

```python
from inference import load_model, visualize_predictions
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = load_model('best_model.pth', device=device)
visualize_predictions(model, 'path/to/image.jpg', device=device)
```

---

## Detailed Documentation

### Project Architecture

The CNN model processes images through two convolutional blocks:

```
Input (3×32×32)
    ↓
[Conv2d (32 filters) → BatchNorm → ReLU → MaxPool]
    ↓
[Conv2d (64 filters) → BatchNorm → ReLU → MaxPool]
    ↓
Flatten (4096 features)
    ↓
[Linear (128 units) → ReLU → Dropout]
    ↓
Linear (10 classes)
```

**Key Components:**
- **Convolutional Layers**: Extract visual features (edges, textures, shapes)
- **Batch Normalization**: Stabilize training and speed up learning
- **Max Pooling**: Reduce spatial dimensions while keeping important information
- **Dropout**: Prevent overfitting by randomly deactivating neurons
- **Fully Connected Layers**: Map learned features to class predictions

### Data Loading (`data_loader.py`)

The data loading pipeline handles:

1. **Dataset**: CIFAR-10 contains:
   - 60,000 32×32 RGB images
   - 10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
   - Pre-split: 50,000 training + 10,000 test

2. **Preprocessing**:
   ```python
   transforms.Compose([
       transforms.RandomCrop(32, padding=4),      # Data augmentation
       transforms.RandomHorizontalFlip(),          # Data augmentation
       transforms.ToTensor(),                      # Convert to tensor
       transforms.Normalize(mean, std)             # Normalization
   ])
   ```

3. **Data Split**:
   - Training: 45,000 samples (with augmentation)
   - Validation: 5,000 samples
   - Test: 10,000 samples

4. **Batching**: Loads images in batches for efficient GPU processing

### Training Loop (`train.py`)

The training process repeats for each epoch:

```
For each batch of images:
    1. Forward Pass: images → model → predictions
    2. Calculate Loss: how wrong are the predictions?
    3. Backward Pass: compute gradients using backpropagation
    4. Update Weights: adjust model parameters using Adam optimizer
```

**Metrics**:
- **Loss**: Measures prediction error (lower is better)
- **Accuracy**: Percentage of correct predictions (higher is better)

### Model Inference (`inference.py`)

After training, use the inference module to:
- Load the trained model
- Predict classes for new images
- Display confidence scores
- Visualize results

---

## Understanding Each Component

### 1. Data Loading - Why It Matters

```python
# Bad approach: load all data into memory
images = load_all_images()  # ~1GB for CIFAR-10!

# Good approach: load in batches
for batch in data_loader:
    # Process batch, then move to next
```

Benefits:
- **Memory Efficient**: Only keep one batch in memory
- **Fast**: Parallel loading while GPU processes previous batch
- **Flexible**: Easy to add augmentation

### 2. Model Architecture - Why This Structure?

```
Input: 32×32 image → too big to connect directly to output
Convolution: Learn patterns at different spatial locations
Pooling: Reduce size, keep important features
Flattening: Convert 2D features to 1D vector
Fully Connected: Make final classification decision
```

**Why 2 conv blocks?**
- 1st block: Detects low-level features (edges, colors)
- 2nd block: Detects high-level features (shapes, objects)

### 3. Training - Why These Choices?

```python
# Loss function: CrossEntropyLoss
# - Specifically designed for multi-class classification
# - Combines LogSoftmax and NLLLoss for numerical stability

# Optimizer: Adam
# - Adapts learning rate per parameter
# - Works well with noisy data
# - Better than vanilla gradient descent for this task

# Learning rate: 0.001
# - Small enough to find good solutions
# - Large enough to train in reasonable time
```

---

## Hyperparameter Tuning

Adjust these in `train.py`:

| Hyperparameter | Current | Range | Effect |
|---|---|---|---|
| `NUM_EPOCHS` | 10 | 5-50 | More epochs = higher accuracy but more time |
| `BATCH_SIZE` | 32 | 8-128 | Larger batch = faster training but more memory |
| `LEARNING_RATE` | 0.001 | 0.0001-0.01 | Higher = faster but unstable; Lower = slower but stable |

**Experiment Tips**:
1. **Low Accuracy?**
   - Increase `NUM_EPOCHS` to 20
   - Decrease `LEARNING_RATE` to 0.0005
   
2. **Out of Memory?**
   - Reduce `BATCH_SIZE` to 16
   - Use CPU instead of GPU

3. **Training Too Slow?**
   - Increase `BATCH_SIZE` to 64
   - Reduce `NUM_EPOCHS`

---

## Output Files Generated

After training:

1. **best_model.pth** (≈2 MB)
   - Contains all learned model weights
   - Can be loaded for inference or fine-tuning
   
2. **training_results.png**
   - Left plot: Training vs Validation Loss (should decrease)
   - Right plot: Validation Accuracy (should increase)

---

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'torch'"

**Solution**: Upgrade pip and install PyTorch (pinned)
```bash
pip install --upgrade pip && pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cpu
```

### Issue: Accuracy stuck below 50%

**Diagnosis**: Model isn't learning properly

**Solutions**:
1. Check if data is loading correctly
2. Increase `NUM_EPOCHS` to 20
3. Decrease `LEARNING_RATE` to 0.0005
4. Verify GPU is being used (check device output)

### Issue: "CUDA out of memory"

**Solutions**:
```python
# Option 1: Reduce batch size
BATCH_SIZE = 16  # instead of 32

# Option 2: Use CPU
device = torch.device('cpu')

# Option 3: Clear cache
torch.cuda.empty_cache()
```

### Issue: Training is very slow (> 30 min for 10 epochs)

**Solutions**:
1. **Verify GPU usage**: Should see NVIDIA GPU in output
2. **Increase batch size**: `BATCH_SIZE = 64`
3. **Reduce precision**: Use `torch.float16` (advanced)

---

## Learning Concepts Explained

### 1. Convolutional Neural Networks
- **Convolution**: Sliding window operation that preserves spatial relationships
- **Receptive Field**: Area of input that affects one output neuron
- **Feature Maps**: Outputs of convolution layers showing detected patterns

### 2. Training Fundamentals
- **Gradient Descent**: Iteratively move parameters in direction of steepest loss decrease
- **Backpropagation**: Efficiently compute gradients using chain rule
- **Epochs & Batches**: Epoch = full pass through data; Batch = subset processed together

### 3. Overfitting & Generalization
- **Overfitting**: Model memorizes training data but performs poorly on new data
- **Dropout**: Randomly disable neurons to prevent co-adaptation
- **Data Augmentation**: Create variations of training data to improve generalization

### 4. Evaluation Metrics
- **Accuracy**: Simple metric - percentage correct
- **Loss**: Detailed metric - shows how confident wrong predictions were
- **Train vs Val/Test**: Detects overfitting when train accuracy >> test accuracy

---

## Expanding Your Knowledge

### Challenge 1: Improve Accuracy
Try to achieve > 85% test accuracy by:
- Tuning hyperparameters
- Adding more convolutional layers
- Using deeper architecture (ResNet style)

### Challenge 2: Different Dataset
Replace CIFAR-10 with:
- **MNIST**: Handwritten digits (simpler, trains faster)
- **Fashion-MNIST**: Clothing items (similar size to CIFAR-10)
- **STL-10**: Similar to CIFAR-10 but unlabeled data available

### Challenge 3: Visualization
Visualize:
- Learned filters in first convolutional layer
- Feature maps at different layers
- Misclassified examples

### Challenge 4: Deployment
- Save model for production
- Create a simple web interface
- Export to mobile using ONNX

---

## References

- **PyTorch Tutorials**: https://pytorch.org/tutorials/
- **CIFAR-10 Paper**: https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf
- **Stanford CS231N**: https://cs231n.github.io/
- **Deep Learning Book**: https://www.deeplearningbook.org/

---

## License

This project is released under the MIT License. Feel free to use for learning and teaching purposes.

---

**Happy Learning!** 🚀

Questions? Experiment with the code - that's the best way to learn!
