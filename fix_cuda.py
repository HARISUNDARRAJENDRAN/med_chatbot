import os
import torch
import sys

# Force CPU mode
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["USE_CPU"] = "1"
os.environ["FORCE_CPU"] = "1"
os.environ["TORCH_DEVICE"] = "cpu"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# Disable CUDA in PyTorch
if hasattr(torch, 'cuda'):
    torch.cuda.is_available = lambda: False

print("✅ CUDA disabled successfully. Your application will now use CPU only.")
print("📋 Environment variables set:")
print(f"  - CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")
print(f"  - TORCH_DEVICE: {os.environ['TORCH_DEVICE']}")
print(f"  - PyTorch CUDA available: {torch.cuda.is_available() if hasattr(torch, 'cuda') else False}")

# Create a .no_cuda file that your application can check
with open(".no_cuda", "w") as f:
    f.write("CPU_ONLY=1")

print("✅ Created .no_cuda file for your application to detect")