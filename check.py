import torch
print(torch.cuda.is_available())  # Should print True if GPU is accessible
print(torch.version.cuda)         # Verify CUDA version
print(torch.backends.cudnn.enabled)  # Verify cuDNN is enabled

torch.cuda.memory_summary()