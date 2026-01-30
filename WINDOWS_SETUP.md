# 🚀 Windows Setup Guide - Fix DLL Error

## Problem
You're getting this error:
```
OSError: [WinError 1114] A dynamic link library (DLL) initialization routine failed
Microsoft Visual C++ Redistributable is not installed
```

## Solution - 4 Easy Steps

### Step 1: Install Microsoft Visual C++ Redistributable
1. Download from: https://aka.ms/vs/17/release/vc_redist.x64.exe
2. Run the installer
3. Click "Install" and wait for completion
4. Click "Close"
5. **RESTART YOUR COMPUTER**

### Step 2: Clean up and reinstall PyTorch

Open PowerShell in this folder and run:

```powershell
# Activate the virtual environment
.\env\Scripts\Activate.ps1

# Remove old PyTorch installation
pip uninstall torch torchvision -y

# Upgrade pip (helps find compatible wheels)
pip install --upgrade pip

# Install CPU-only PyTorch (Windows compatible, pinned to tested versions)
pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cpu

# Install other dependencies from requirements
pip install -r requirements.txt
```

### Step 3: Verify Installation

Run this command:
```powershell
python -c "import torch; print(f'PyTorch {torch.__version__} installed!'); print(f'Device: CPU')"
```

You should see:
```
PyTorch 2.10.0+cpu installed!
Device: CPU
```

### Step 4: Train Your Model

```powershell
python train.py
```

---

## Troubleshooting

### Still getting DLL error after reinstalling Visual C++?

Try this:
1. Delete the `.venv` folder completely
2. Create a new virtual environment: `python -m venv env`
3. Activate it: `.\env\Scripts\Activate.ps1`
4. Upgrade pip and install PyTorch (pinned): `pip install --upgrade pip && pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cpu && pip install -r requirements.txt`

### Having trouble with PowerShell activation?

Use Command Prompt instead:
```cmd
env\Scripts\activate.bat
python train.py
```

### Still not working?

Try installing from conda instead:
```powershell
conda install pytorch torchvision pytorch-cuda=12.4 -c pytorch -c nvidia
```

---

## Quick Reference

| Task | Command |
|------|---------|
| Activate environment (PowerShell) | `.\env\Scripts\Activate.ps1` |
| Activate environment (CMD) | `env\Scripts\activate.bat` |
| Install packages | `pip install --upgrade pip && pip install -r requirements.txt` |
| Train model | `python train.py` |
| Check PyTorch | `python -c "import torch; print(torch.__version__)"` |

---

## Expected Output When Training Starts

```
Using device: cpu
Loading CIFAR-10 dataset...
Dataset loaded successfully!
  - Training batches: 1406
  - Validation batches: 156
  - Test batches: 313

Model initialized:
  - Architecture: SimpleCNN
  - Total parameters: 67,010

Starting training...

Epoch 1/10
Training: 100%|██████████| 1406/1406 [00:45<00:00, 31.24it/s]
  Train Loss: 2.1234
  Val Loss: 1.9876
  Val Accuracy: 32.45%

... (more epochs)

Epoch 10/10
  Train Loss: 0.3456
  Val Loss: 0.8765
  Val Accuracy: 78.90%

Test Accuracy: 78.34%
```

---

## Windows-Specific Notes

- 🐢 CPU training is slower (~2-5 min per epoch) but still works well for learning
- 📦 PyTorch CPU edition is fully functional for this project
- 💾 Total dataset download: ~170 MB (cached after first run)
- ⏱️ Total training time: 10-20 minutes for 10 epochs on CPU

---

## Need More Help?

1. Read `QUICK_REF.md` for quick reference
2. Read `QUICKSTART.md` for detailed setup
3. Read `PROJECT_SUMMARY.md` for complete overview
4. Check `TRAINING_EXPLAINED.md` for technical details

Good luck! 🚀
