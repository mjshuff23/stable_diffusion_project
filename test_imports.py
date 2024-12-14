import torch
import diffusers
import transformers
import accelerate

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Diffusers version: {diffusers.__version__}")
print(f"Transformers version: {transformers.__version__}")