
# │ 1. Install & configure numba-cuda (ensures PTX ≤ 8.4 / CUDA 12.4)         │


# (Replaces built-in CUDA target with the NVIDIA-maintained one)
# !pip install -q --system numba-cuda==0.4.0  # Google Colab maintainers recommend v0.4.0 for compatibility :contentReference[oaicite:0]{index=0}

# Enable the JIT‐link patch so that Numba emits PTX your driver accepts
from numba import config
config.CUDA_ENABLE_PYNVJITLINK = 1          # activates the pinned NVVM linker :contentReference[oaicite:1]{index=1}

# Verify we have CUDA available at the expected driver version

import numpy as np
from numba import cuda

@cuda.jit
def vector_add(a, b, out):
    idx = cuda.grid(1)
    if idx < a.size:
        out[idx] = a[idx] + b[idx]

# Host-side data
n = 1_000_000
a = np.random.rand(n).astype(np.float32)
b = np.random.rand(n).astype(np.float32)
out = np.empty_like(a)

# Transfer to GPU
d_a = cuda.to_device(a)
d_b = cuda.to_device(b)
d_out = cuda.to_device(out)

# Launch configuration
threads_per_block = 256
blocks_per_grid = (n + threads_per_block - 1) // threads_per_block

# Kernel launch
vector_add[blocks_per_grid, threads_per_block](d_a, d_b, d_out)

# Copy result back
d_out.copy_to_host(out)

# Verify a few elements
print("First 5 elements of a + b:", out[:5])
