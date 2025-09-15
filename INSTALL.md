# Installation Guide for Robotic Vision Pipeline

## Quick Setup

### 1. Create Virtual Environment (Recommended)
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate
```

### 2. Install Dependencies

#### Option A: Full Installation (Recommended for Development)
```bash
pip install -r requirements.txt
```

#### Option B: Minimal Installation (Basic Functionality Only)
```bash
pip install -r requirements-minimal.txt
```

### 3. Download SAM Checkpoint
```bash
# Download SAM ViT-B model (375MB)
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth

# Or download SAM ViT-H model (2.5GB) for better performance
# wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

### 4. Verify Installation
```bash
# Run unit tests to verify everything works
cd tests
python test_object_detection_only.py
python test_object_classification.py
```

## System Requirements

### Minimum Requirements
- Python 3.8+
- 8GB RAM
- 2GB free disk space
- CPU: Intel i5 or equivalent

### Recommended Requirements
- Python 3.9+
- 16GB RAM
- 5GB free disk space
- GPU: NVIDIA RTX 3060 or equivalent (for faster inference)
- CUDA 11.8+ (if using GPU)

## GPU Support (Optional)

### Install CUDA version of PyTorch
```bash
# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## Troubleshooting

### Common Issues

1. **SAM Installation Issues**
   ```bash
   pip install git+https://github.com/facebookresearch/segment-anything.git
   ```

2. **OpenCV Issues**
   ```bash
   pip uninstall opencv-python opencv-contrib-python
   pip install opencv-python==4.8.1.78
   ```

3. **PyTorch Version Conflicts**
   ```bash
   pip uninstall torch torchvision torchaudio
   pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2
   ```

4. **Memory Issues**
   - Use ViT-B model instead of ViT-H
   - Reduce image resolution in tests
   - Close other applications

### Platform-Specific Notes

#### Windows
- Ensure Visual Studio Build Tools are installed
- Use PowerShell or Command Prompt
- May need to install Microsoft Visual C++ Redistributable

#### Linux
- Install system dependencies:
  ```bash
  sudo apt update
  sudo apt install python3-pip python3-venv
  sudo apt install libgl1-mesa-glx libglib2.0-0
  ```

#### macOS
- Install Xcode Command Line Tools:
  ```bash
  xcode-select --install
  ```
- Consider using Homebrew for Python installation

## Development Setup

### Additional Development Tools
```bash
pip install black flake8 pytest-xdist jupyter
```

### Pre-commit Hooks (Optional)
```bash
pip install pre-commit
pre-commit install
```

## Verification Commands

```bash
# Check Python version
python --version

# Check PyTorch installation
python -c "import torch; print(f'PyTorch: {torch.__version__}')"

# Check CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Check OpenCV installation
python -c "import cv2; print(f'OpenCV: {cv2.__version__}')"

# Check SAM installation
python -c "from segment_anything import sam_model_registry; print('SAM installed successfully')"
```

## Getting Help

- Check the README.md for usage examples
- Run unit tests to identify specific issues
- Check GitHub issues for known problems
- Ensure all file paths are correct for your system