import torch
import sys
import os

print("Hello from Slurm!")
print(f"Python version: {sys.version}")
print(f"CWD: {os.getcwd()}")

if torch.cuda.is_available():
    print(f"CUDA is available. Device count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"Device {i}: {torch.cuda.get_device_name(i)}")
else:
    print("CUDA is NOT available.")
