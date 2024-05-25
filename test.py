import torch

# Check if CUDA is available
cuda_available = torch.cuda.is_available()
print(f"CUDA Available: {cuda_available}")

if cuda_available:
    # Number of GPUs
    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs: {num_gpus}")
    
    for i in range(num_gpus):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"  - Total Memory: {torch.cuda.get_device_properties(i).total_memory / (1024 ** 3):.2f} GB")
        print(f"  - CUDA Capability: {torch.cuda.get_device_properties(i).major}.{torch.cuda.get_device_properties(i).minor}")

# CPU Information
print("\nCPU Information:")
print(f"Number of CPUs: {torch.get_num_threads()}")

# Memory Information
if cuda_available:
    total_memory = torch.cuda.memory_allocated()
    cached_memory = torch.cuda.memory_reserved()
    print(f"\nMemory Information (if CUDA is available):")
    print(f"  - Allocated Memory: {total_memory / (1024 ** 3):.2f} GB")
    print(f"  - Cached Memory: {cached_memory / (1024 ** 3):.2f} GB")