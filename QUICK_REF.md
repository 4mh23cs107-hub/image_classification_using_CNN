# Quick Reference Card

## 🚀 Getting Started

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train the model
python train.py

# 3. Wait for it to complete (~3-5 minutes)
```

## 📊 Expected Output

```
Using device: cuda                 # GPU (fast) or cpu (slow)
Loading CIFAR-10 dataset...
Dataset loaded successfully!

Epoch 1/10
Training: 100%|████████████| 1406/1406 [00:10<00:00, 130.05it/s]
  Train Loss: 2.2156
  Val Loss: 2.0234
  Val Accuracy: 25.34%

... (more epochs)

Epoch 10/10
  Train Loss: 0.3456
  Val Loss: 0.8765
  Val Accuracy: 78.90%

Test Accuracy: 78.34%
```

## 📁 Files to Know

| File | What it does | When to use |
|------|-------------|-----------|
| `train.py` | Trains the model | Run this first |
| `model.py` | CNN architecture | Modify to change model |
| `data_loader.py` | Loads CIFAR-10 | Change dataset here |
| `inference.py` | Makes predictions | Use to classify images |
| `requirements.txt` | Dependencies | Run `pip install -r` |

## 🔧 Common Commands

```python
# Install dependencies
pip install -r requirements.txt

# Train for longer
# Edit train.py and change: NUM_EPOCHS = 20
# Then: python train.py

# Use CPU instead of GPU (slower but always works)
# Edit train.py and change: 
#   device = torch.device('cpu')

# Try a different learning rate
# Edit train.py and change: LEARNING_RATE = 0.0001
# Then: python train.py

# Make predictions
from inference import load_model, visualize_predictions
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = load_model('best_model.pth', device=device)
visualize_predictions(model, 'image.jpg', device=device)
```

## 📈 Hyperparameters to Tweak

| Parameter | Current | Try This | Effect |
|-----------|---------|----------|--------|
| `NUM_EPOCHS` | 10 | 20 | Better accuracy |
| `BATCH_SIZE` | 32 | 64 | Faster training |
| `LEARNING_RATE` | 0.001 | 0.0005 | More stable |

## ❓ Quick Troubleshooting

| Problem | Fix |
|---------|-----|
| `No module named torch` | `pip install torch torchvision` |
| Accuracy too low | Increase `NUM_EPOCHS` to 20 |
| Out of memory | Reduce `BATCH_SIZE` to 16 |
| Training slow | Check if GPU is being used |
| Model crashes | Reduce `BATCH_SIZE` to 8 |

## 📚 Read These Files

1. **QUICKSTART.md** - 5 minute tutorial
2. **EXAMPLES.md** - Code snippets to copy
3. **TRAINING_EXPLAINED.md** - How training works

## 🎯 Success Metrics

After training, you should see:

- ✅ Loss decreases (2.3 → 0.3)
- ✅ Accuracy increases (10% → 78%)
- ✅ Test accuracy ≈ validation accuracy
- ✅ Model saved as `best_model.pth`
- ✅ Plot saved as `training_results.png`

## 💾 Files Created During Training

```
best_model.pth              # Trained model (load with torch.load)
training_results.png        # Loss and accuracy plots
data/                       # CIFAR-10 dataset (cached after first run)
```

## 🔑 Key Ideas

- **CNN**: Network learns visual features automatically
- **Training**: Repeatedly show examples and adjust weights
- **Validation**: Check if model generalizes to new data
- **Test**: Final evaluation on unseen data

## ⚡ Performance Tips

- GPU 10x faster than CPU
- Larger `BATCH_SIZE` trains faster (if memory allows)
- More `NUM_EPOCHS` gives better accuracy
- Smaller `LEARNING_RATE` more stable

## 🎓 Learning Path

| Time | What to Do |
|------|-----------|
| 5 min | Run `python train.py` |
| 10 min | Read `QUICKSTART.md` |
| 30 min | Read `TRAINING_EXPLAINED.md` |
| 1 hour | Try examples in `EXAMPLES.md` |
| 2 hours | Modify `model.py` and retrain |
| 3+ hours | Try different datasets/architectures |

## 📞 Need Help?

1. Read `PROJECT_SUMMARY.md` for full overview
2. Check `README_complete.md` for comprehensive docs
3. See `EXAMPLES.md` for code snippets
4. Read `TRAINING_EXPLAINED.md` for deep understanding

## ✅ Next Steps

```bash
# Step 1
python train.py

# Step 2 (after training completes)
# Open EXAMPLES.md and try the code snippets

# Step 3
# Read TRAINING_EXPLAINED.md to understand what happened

# Step 4
# Modify model.py to add more layers and retrain

# Step 5
# Try EXAMPLES.md: "Change Dataset" section
```

---

**Remember**: Start with `python train.py` and everything else follows! 🚀
