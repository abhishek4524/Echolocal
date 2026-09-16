import sys
import torch

def test_gpu():
    print("Python Version:", sys.version)
    print("PyTorch Version:", torch.__version__)
    
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_available}")
    
    if cuda_available:
        device_count = torch.cuda.device_count()
        print(f"Device Count: {device_count}")
        for i in range(device_count):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("WARNING: NVIDIA GPU is NOT detected by PyTorch.")

if __name__ == "__main__":
    test_gpu()
