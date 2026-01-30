"""
QUICK START GUIDE - CNN Image Classification

This guide will get you training your first CNN in 5 minutes!
"""

# ============================================================================
# STEP 1: Install Dependencies (2 minutes)
# ============================================================================
"""
Open your terminal/command prompt and run:

    pip install -r requirements.txt

Or if that doesn't work, install PyTorch separately:

    pip install torch torchvision
    
Verify installation:

    python -c "import torch; print(torch.__version__)"
"""

# ============================================================================
# STEP 2: Train the Model (3 minutes)
# ============================================================================
"""
Simply run:

    python train.py

What happens:
    1. CIFAR-10 dataset downloads automatically (~170 MB)
    2. Model trains for 10 epochs
    3. Progress shows with loss and accuracy
    4. Best model saved as 'best_model.pth'
    5. Training plots saved as 'training_results.png'

Expected output:
    Epoch 1/10
      Train Loss: 2.1234
      Val Loss: 1.8976
      Val Accuracy: 32.45%
    
    ... (more epochs)
    
    Test Accuracy: 78.34%
"""

# ============================================================================
# STEP 3: Understand What's Happening
# ============================================================================
"""
The training process has these main components:

1. DATA LOADING (data_loader.py)
   - Loads 50,000 training images from CIFAR-10
   - Splits into 45,000 train + 5,000 validation
   - Applies data augmentation (crops, flips)
   - Normalizes using dataset statistics
   
2. MODEL (model.py)
   - SimpleCNN with 2 convolutional blocks
   - 32 and 64 filters respectively
   - Batch normalization for stability
   - Dropout to prevent overfitting
   
3. TRAINING (train.py)
   - For each epoch:
     a) Feed batch of images to model
     b) Compare predictions with actual labels
     c) Calculate loss (error)
     d) Update model weights to reduce loss
   - Validate on held-out data
   - Save best performing model

4. EVALUATION
   - Check accuracy on test set
   - Visualize training curves
"""

# ============================================================================
# STEP 4: File Guide
# ============================================================================
"""
model.py
  └─ Defines SimpleCNN class
     - 2 Conv blocks: 3 → 32 → 64 filters
     - Fully connected layers for classification
     - With Batch Norm and Dropout

data_loader.py
  └─ get_data_loaders() function
     - Downloads CIFAR-10
     - Creates train/val/test splits
     - Applies augmentation and normalization

train.py
  └─ Main training script
     - train_epoch(): One epoch of training
     - validate(): Evaluate on validation set
     - test(): Evaluate on test set
     - plot_results(): Visualize training curves

inference.py
  └─ Make predictions on new images
     - load_model(): Load trained weights
     - predict_image(): Classify single image
     - visualize_predictions(): Show results with confidence

requirements.txt
  └─ Python package dependencies
"""

# ============================================================================
# STEP 5: Make Your First Prediction
# ============================================================================
"""
After training, use this code:

    from inference import load_model, visualize_predictions
    import torch
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model('best_model.pth', device=device)
    
    # Predict on an image
    visualize_predictions(model, 'path/to/image.jpg', device=device)

The image should be in one of these classes:
    0: airplane
    1: automobile  
    2: bird
    3: cat
    4: deer
    5: dog
    6: frog
    7: horse
    8: ship
    9: truck
"""

# ============================================================================
# STEP 6: Experiment & Learn
# ============================================================================
"""
Try these modifications:

1. Train longer:
   In train.py, change:
       NUM_EPOCHS = 10
   to:
       NUM_EPOCHS = 20

2. Change batch size:
   NUM_BATCH_SIZE = 32  →  NUM_BATCH_SIZE = 64

3. Adjust learning rate:
   LEARNING_RATE = 0.001  →  LEARNING_RATE = 0.0005

4. Add more layers to model:
   Open model.py and add another Conv block

5. Use a different optimizer:
   In train.py, change:
       optimizer = optim.Adam(...)
   to:
       optimizer = optim.SGD(model.parameters(), lr=0.01)
"""

# ============================================================================
# STEP 7: Common Issues
# ============================================================================
"""
Problem: ImportError: No module named 'torch'
Solution: pip install torch torchvision

Problem: CUDA out of memory
Solution: Reduce BATCH_SIZE from 32 to 16

Problem: Training is very slow
Solution: 
  - Check if using GPU (should say "cuda" not "cpu")
  - Increase BATCH_SIZE to 64
  - Reduce NUM_EPOCHS

Problem: Low accuracy (< 50%)
Solution:
  - Train for more epochs (NUM_EPOCHS = 20)
  - Lower learning rate (LEARNING_RATE = 0.0005)
  - Check data is loading (add print statements)

Problem: Can't find best_model.pth
Solution: Make sure train.py completed successfully
          Check for error messages in training output
"""

# ============================================================================
# STEP 8: Learning Resources
# ============================================================================
"""
To understand the concepts better:

1. Convolutional Networks:
   https://cs231n.github.io/convolutional-networks/

2. PyTorch Tutorials:
   https://pytorch.org/tutorials/

3. CIFAR-10 Dataset:
   https://www.cs.toronto.edu/~kriz/cifar.html

4. Deep Learning Basics:
   https://www.deeplearningbook.org/

5. Interactive CNN Visualization:
   https://poloclub.github.io/cnn-explainer/
"""

# ============================================================================
# SUMMARY
# ============================================================================
"""
You now have a complete CNN image classification pipeline that teaches:

✓ Data Loading & Preprocessing
  - How to load image datasets efficiently
  - Data augmentation for better generalization
  - Normalization importance

✓ Model Architecture Design
  - Convolutional layers and their purpose
  - Batch normalization for training stability
  - Pooling and fully connected layers

✓ Training Process
  - Forward pass through network
  - Loss calculation
  - Backward pass (backpropagation)
  - Weight updates with optimization

✓ Model Evaluation
  - Train/Validation/Test splits
  - Accuracy metrics
  - Overfitting detection

✓ Inference
  - Loading trained models
  - Making predictions
  - Confidence scores

Next steps:
1. Run 'python train.py' to see it in action
2. Modify hyperparameters and observe effects
3. Experiment with architecture changes
4. Try on different datasets
5. Deploy the model
"""
