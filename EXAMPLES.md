"""
USAGE EXAMPLES - Practical Code Snippets

This file shows how to use the CNN project for different tasks.
Copy and paste these examples to try different operations.
"""

# ============================================================================
# EXAMPLE 1: BASIC TRAINING
# ============================================================================

"""
Simplest way to train the model:

Just run in your terminal:
    python train.py

That's it! The script handles everything:
1. Downloads CIFAR-10
2. Trains for 10 epochs
3. Saves the best model
4. Shows training plots
5. Reports final accuracy
"""

# ============================================================================
# EXAMPLE 2: MODIFY TRAINING PARAMETERS
# ============================================================================

"""
To change training settings, edit train.py and modify:

    # Hyperparameters
    NUM_EPOCHS = 20        # Train for 20 epochs instead of 10
    BATCH_SIZE = 64        # Use larger batches (faster)
    LEARNING_RATE = 0.0005 # Lower learning rate (more stable)
    NUM_WORKERS = 0        # Keep at 0 on Windows

Then run:
    python train.py

Expected: Better accuracy due to longer training
"""

# ============================================================================
# EXAMPLE 3: MAKE PREDICTIONS ON A SINGLE IMAGE
# ============================================================================

# Copy this into a Python script or Jupyter notebook:

from inference import load_model, visualize_predictions, predict_image
import torch

# Setup device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load the trained model
model = load_model('best_model.pth', device=device)
print("Model loaded!")

# Method 1: Just get the prediction (without visualization)
predicted_class, confidence, probabilities = predict_image(
    model, 
    'path/to/your/image.jpg',  # Replace with your image path
    device=device
)

print(f"Predicted class: {predicted_class}")
print(f"Confidence: {confidence:.2f}%")
print(f"\nAll probabilities:")
from inference import CLASS_NAMES
for name, prob in zip(CLASS_NAMES, probabilities):
    print(f"  {name}: {prob*100:.2f}%")

# Method 2: Visualize with bar chart
visualize_predictions(model, 'path/to/your/image.jpg', device=device)

# ============================================================================
# EXAMPLE 4: BATCH PREDICTION ON MULTIPLE IMAGES
# ============================================================================

# To predict on multiple images:

from inference import load_model, predict_image, CLASS_NAMES
import torch
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = load_model('best_model.pth', device=device)

# List all images in a folder
image_folder = 'my_images'
for filename in os.listdir(image_folder):
    if filename.endswith(('.jpg', '.png', '.jpeg')):
        image_path = os.path.join(image_folder, filename)
        
        predicted_class, confidence, _ = predict_image(
            model, image_path, device=device
        )
        
        print(f"{filename}: {predicted_class} ({confidence:.2f}%)")

# ============================================================================
# EXAMPLE 5: INSPECT MODEL ARCHITECTURE
# ============================================================================

# See what the model looks like:

from model import SimpleCNN
import torch

model = SimpleCNN(num_classes=10)

# Print architecture
print(model)

# Print number of parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nTotal parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

# Print each layer
print("\n\nDetailed layer information:")
for name, module in model.named_modules():
    if name:  # Skip the parent module
        print(f"{name}: {module}")

# ============================================================================
# EXAMPLE 6: TEST ON CIFAR-10 TEST SET
# ============================================================================

# To evaluate model on the full test set:

import torch
from data_loader import get_data_loaders
from model import SimpleCNN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load data
_, _, test_loader, _ = get_data_loaders(batch_size=32)

# Load model
model = SimpleCNN(num_classes=10).to(device)
model.load_state_dict(torch.load('best_model.pth'))
model.eval()

# Evaluate
correct = 0
total = 0
class_correct = [0] * 10
class_total = [0] * 10

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Per-class accuracy
        for i in range(10):
            class_total[i] += (labels == i).sum().item()
            class_correct[i] += ((predicted == labels) & (labels == i)).sum().item()

overall_accuracy = 100 * correct / total
print(f"Overall Test Accuracy: {overall_accuracy:.2f}%")

print("\nPer-class accuracy:")
from inference import CLASS_NAMES
for i, class_name in enumerate(CLASS_NAMES):
    if class_total[i] > 0:
        acc = 100 * class_correct[i] / class_total[i]
        print(f"  {class_name}: {acc:.2f}%")

# ============================================================================
# EXAMPLE 7: MODIFY THE ARCHITECTURE
# ============================================================================

# To try a different architecture, edit model.py:
# Replace the forward method with:

"""
def forward(self, x):
    # First convolutional block
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu1(x)
    x = self.pool1(x)
    
    # Second convolutional block
    x = self.conv2(x)
    x = self.bn2(x)
    x = self.relu2(x)
    x = self.pool2(x)
    
    # ADD A THIRD BLOCK HERE:
    self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
    self.bn3 = nn.BatchNorm2d(128)
    self.relu3 = nn.ReLU()
    self.pool3 = nn.MaxPool2d(2, 2)
    
    x = self.conv3(x)
    x = self.bn3(x)
    x = self.relu3(x)
    x = self.pool3(x)
    
    # Flatten (now 128 * 4 * 4 = 2048 features)
    x = x.view(x.size(0), -1)
    
    # Fully connected
    x = self.fc1(x)
    x = self.relu1(x)
    x = self.dropout(x)
    x = self.fc2(x)
    
    return x
"""

# Then run: python train.py
# The model will train with the new architecture

# ============================================================================
# EXAMPLE 8: CHANGE OPTIMIZER
# ============================================================================

# In train.py, change the optimizer:

# From:
# optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# To SGD with momentum:
# optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# Or try different learning rates:
# optimizer = optim.Adam(model.parameters(), lr=0.0001)
# optimizer = optim.Adam(model.parameters(), lr=0.01)

# Then run: python train.py
# Compare how different optimizers affect training

# ============================================================================
# EXAMPLE 9: USE GPU ACCELERATION
# ============================================================================

# Ensure GPU is being used:

import torch

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")

# When running train.py, you'll see at the start:
# "Using device: cuda" = GPU is being used ✓
# "Using device: cpu" = Only CPU is being used (slow)

# If on Windows and CUDA not detected, upgrade pip and install PyTorch-CUDA:
# pip install --upgrade pip && pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# ============================================================================
# EXAMPLE 10: SAVE AND LOAD MODELS
# ============================================================================

# Save model:
import torch
from model import SimpleCNN

model = SimpleCNN(num_classes=10)
torch.save(model.state_dict(), 'my_model.pth')

# Later, load model:
model = SimpleCNN(num_classes=10)
model.load_state_dict(torch.load('my_model.pth'))

# Use loaded model:
model.eval()
with torch.no_grad():
    # Make predictions
    output = model(image_tensor)

# ============================================================================
# EXAMPLE 11: COMPARE ARCHITECTURES
# ============================================================================

# To experiment with different architectures:

import torch
from model import SimpleCNN

# Create models with different sizes
models = {
    'small': SimpleCNN(num_classes=10),  # Current model
    'medium': 'Edit model.py to add more filters',
    'large': 'Edit model.py to add more layers'
}

for name, model in models.items():
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{name}: {total_params:,} parameters")

# Train each and compare:
# - Which converges faster?
# - Which achieves best accuracy?
# - Which is fastest to train?
# - Which generalizes best (val vs test)?

# ============================================================================
# EXAMPLE 12: VISUALIZE TRAINING PROGRESS
# ============================================================================

# After training, view the generated plot:

import matplotlib.pyplot as plt
from PIL import Image

# Display the training results image
img = Image.open('training_results.png')
plt.figure(figsize=(12, 5))
plt.imshow(img)
plt.axis('off')
plt.title('Training Results')
plt.tight_layout()
plt.show()

# Or use Python to re-create the plots with your own data:
# See lines in train.py: plot_results(train_losses, val_losses, val_accuracies)

# ============================================================================
# EXAMPLE 13: DEBUG LOW ACCURACY
# ============================================================================

# If getting low accuracy, check data loading:

from data_loader import get_data_loaders
import matplotlib.pyplot as plt
import torchvision

train_loader, val_loader, test_loader, _ = get_data_loaders(batch_size=32)

# Get one batch
images, labels = next(iter(train_loader))

print(f"Batch shape: {images.shape}")  # Should be [32, 3, 32, 32]
print(f"Labels shape: {labels.shape}")  # Should be [32]
print(f"Min pixel value: {images.min()}")  # Should be around -2
print(f"Max pixel value: {images.max()}")  # Should be around +2

# Visualize some images
def imshow(img):
    img = img / 2 + 0.5  # Undo normalization
    plt.imshow(img.permute(1, 2, 0))

plt.figure(figsize=(12, 3))
for i in range(8):
    plt.subplot(1, 8, i+1)
    imshow(images[i])
    from inference import CLASS_NAMES
    plt.title(CLASS_NAMES[labels[i]])
    plt.axis('off')

plt.tight_layout()
plt.show()

# ============================================================================
# EXAMPLE 14: TRANSFER LEARNING (ADVANCED)
# ============================================================================

# Use a pre-trained model from torchvision:

import torch
import torchvision.models as models

# Load pre-trained ResNet50
pretrained_model = models.resnet50(pretrained=True)

# Freeze all layers except the last
for param in pretrained_model.parameters():
    param.requires_grad = False

# Replace the last layer for CIFAR-10 (10 classes)
pretrained_model.fc = torch.nn.Linear(2048, 10)

# Now train with:
# optimizer = torch.optim.Adam(pretrained_model.fc.parameters(), lr=0.001)

# This usually achieves 90%+ accuracy on CIFAR-10!

# ============================================================================
# EXAMPLE 15: USE DIFFERENT LOSS FUNCTIONS
# ============================================================================

# In train.py, change the loss function:

# From:
# criterion = nn.CrossEntropyLoss()

# To try:
# criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # Label smoothing
# criterion = nn.FocalLoss()  # For imbalanced data
# criterion = nn.LabelSmoothingCrossEntropy()  # Custom loss

# Different losses can improve accuracy for specific problems

"""
SUMMARY OF EXAMPLES:
- Example 1-2: Basic training
- Example 3-4: Make predictions
- Example 5-6: Model inspection and evaluation
- Example 7-9: Architecture and optimization modifications
- Example 10-11: Model persistence and comparison
- Example 12-13: Visualization and debugging
- Example 14-15: Advanced techniques

Start with examples 1-4, then experiment with 7-9 to learn!
"""
