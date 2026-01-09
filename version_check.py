''' Python, Pytorch, CUDA, Device 등 정보 확인 '''

import torch
import sys

# python version
print(f"Python: {sys.version}")

# PyTorch & CUDA version
print(f"PyTorch: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")

# GPU 사용 할 수 있는 지 & 사용 할 수 있다면 GPU information 출력
if torch.cuda.is_available():
    print(f"Device: {torch.cuda.get_device_name(0)}")