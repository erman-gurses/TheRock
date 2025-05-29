#!/usr/bin/env python3

import torch

print("Torch version:", torch.__version__)

if torch.version.hip is None:
    raise RuntimeError("HIP backend not detected (torch.version.hip is None)")

if not torch.cuda.is_available():
    raise RuntimeError("ROCm GPU is not available (torch.cuda.is_available() == False)")

print("Test passed ROCm is available and torch is functional.")
