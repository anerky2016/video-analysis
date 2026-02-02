# Installation Guide for MMPose Swimmer Analysis

This guide will help you set up the MMPose-based swimmer posture analysis system.

## Prerequisites

- Python 3.7 or higher
- pip package manager
- (Optional) CUDA-capable GPU for faster processing

## Installation Steps

### 1. Create Virtual Environment

```bash
cd /Users/diz/git/video-analysis
python3 -m venv venv
source venv/bin/activate
```

### 2. Install PyTorch

Choose the appropriate PyTorch version for your system:

**For CPU only (macOS, Linux, Windows):**
```bash
pip install torch torchvision torchaudio
```

**For GPU (Linux/Windows with CUDA 11.8):**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**For GPU (Linux/Windows with CUDA 12.1):**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Visit https://pytorch.org/get-started/locally/ for other configurations.

### 3. Install OpenMIM

OpenMIM is a package manager for OpenMMLab projects:

```bash
pip install -U openmim
```

### 4. Install MMCV

```bash
mim install mmcv
```

This may take a few minutes as it builds or downloads the appropriate version.

### 5. Install MMDetection

```bash
mim install mmdet
```

### 6. Install MMPose

```bash
mim install mmpose
```

### 7. Install Additional Dependencies

```bash
pip install opencv-python numpy
```

Or install everything from requirements.txt:
```bash
pip install -r requirements.txt
```

### 8. Verify Installation

```bash
python -c "import mmpose; print('MMPose version:', mmpose.__version__)"
python -c "import mmcv; print('MMCV version:', mmcv.__version__)"
python -c "import mmdet; print('MMDet version:', mmdet.__version__)"
python -c "import torch; print('PyTorch version:', torch.__version__)"
python -c "import cv2; print('OpenCV version:', cv2.__version__)"
```

## Quick Test

Run the analyzer on a test video:

```bash
python analyze_swimmer_mmpose.py your_video.mp4
```

The first run will automatically download the default model (~200MB).

## Troubleshooting

### Issue: "No module named 'mmcv'"

**Solution:**
```bash
pip uninstall mmcv mmcv-full
mim install mmcv
```

### Issue: "ImportError: cannot import name 'MODELS'"

**Solution:** Update to compatible versions:
```bash
pip install mmpose>=1.3.0 mmdet>=3.0.0 mmcv>=2.0.0
```

### Issue: CUDA out of memory

**Solution:** Use CPU mode:
```bash
python analyze_swimmer_mmpose.py video.mp4 -d cpu
```

### Issue: Model download fails

**Solution:** Download manually:
```bash
mim download mmpose --config td-hm_hrnet-w48_8xb32-210e_coco-256x192 --dest checkpoints/
```

Then use:
```bash
python analyze_swimmer_mmpose.py video.mp4 -c checkpoints/hrnet_w48_coco_256x192.pth
```

### Issue: Slow performance on CPU

**Solution:** Use a lighter model:
```bash
python analyze_swimmer_mmpose.py video.mp4 -m td-hm_mobilenetv2_8xb64-210e_coco-256x192
```

## Platform-Specific Notes

### macOS

- Apple Silicon (M1/M2/M3): PyTorch has MPS acceleration support
- Intel Macs: Use CPU mode
- You may need to install Xcode Command Line Tools: `xcode-select --install`

### Linux

- Works best with NVIDIA GPU and CUDA
- Make sure you have compatible CUDA drivers installed
- Check CUDA version: `nvidia-smi`

### Windows

- Install Visual C++ Build Tools if you encounter compilation errors
- CUDA support requires NVIDIA drivers and CUDA Toolkit
- Use PowerShell or Command Prompt (not Git Bash) for installation

## Recommended Setup for Different Use Cases

### For Development/Testing (CPU)
```bash
pip install torch torchvision torchaudio
mim install mmcv mmpose mmdet
pip install opencv-python numpy
```

### For Production/Fast Processing (GPU)
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
mim install mmcv mmpose mmdet
pip install opencv-python numpy
```

### For Lightweight/Edge Devices
```bash
# Use the same installation but specify lightweight models
python analyze_swimmer_mmpose.py video.mp4 -m td-hm_mobilenetv2_8xb64-210e_coco-256x192
```

## Next Steps

After installation, check out:
- `README_MMPOSE.md` - Full documentation
- `analyze_swimmer_mmpose.py` - Main analysis script
- `swimmer_pose_mmpose.py` - Core detector class for custom integration

## Getting Help

If you encounter issues:
1. Check MMPose documentation: https://mmpose.readthedocs.io/
2. Verify package versions match requirements
3. Try with a lightweight model first
4. Test with CPU mode before GPU mode
