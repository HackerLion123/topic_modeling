import torch


def get_device(verbose=True):
    """
    Automatically select the best available device across platforms.
    Priority: CUDA > MPS > ROCm > CPU
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        device_name = torch.cuda.get_device_name(0)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        device_name = "Apple Silicon GPU (MPS)"
    elif torch.version.hip is not None and torch.cuda.is_available():
        # ROCm uses "cuda" device but is AMD ROCm underneath
        device = torch.device("cuda")
        device_name = "AMD GPU (ROCm)"
    else:
        device = torch.device("cpu")
        device_name = "CPU"
    if verbose:
        print(f"Using device: {device_name}")
        if device.type == "cuda":
            print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        elif device.type == "mps":
            print(f"  macOS: {torch.backends.mps.is_macos13_or_newer()}")
    return device

