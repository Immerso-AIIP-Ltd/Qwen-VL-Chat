#!/bin/bash
# Script to upgrade PyTorch to version 2.6+ with CUDA 12.1 support (compatible with CUDA 12.8)

echo "============================================================"
echo "Upgrading PyTorch for CUDA 12.8"
echo "============================================================"
echo ""

# Activate virtual environment
source qwen-venv/bin/activate

# Check current PyTorch version
echo "Current PyTorch version:"
python -c "import torch; print(f'  PyTorch: {torch.__version__}'); print(f'  CUDA available: {torch.cuda.is_available()}')" 2>/dev/null || echo "  PyTorch not installed or error checking version"
echo ""

# Check CUDA version
echo "CUDA version:"
nvcc --version | grep "release" || echo "  CUDA compiler not found"
echo ""

# Install PyTorch 2.6+ with CUDA 12.1 support (compatible with CUDA 12.8)
echo "Installing PyTorch 2.6+ with CUDA 12.1 support..."
echo "This may take a few minutes..."
echo ""

pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo ""
echo "============================================================"
echo "Verifying installation..."
echo "============================================================"

python -c "
import torch
print(f'✓ PyTorch version: {torch.__version__}')
print(f'✓ CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'✓ CUDA version: {torch.version.cuda}')
    print(f'✓ GPU device: {torch.cuda.get_device_name(0)}')
else:
    print('⚠ CUDA not available')
"

echo ""
echo "============================================================"
echo "Installation complete!"
echo "============================================================"
echo ""
echo "You can now run: python testqwen.py"
