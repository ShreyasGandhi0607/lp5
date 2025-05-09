
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

n = 1_000_000
a = np.random.rand(n).astype(np.float32)
b = np.random.rand(n).astype(np.float32)
out = np.zeros_like(a)

d_a = cuda.to_device(a)
d_b = cuda.to_device(b)
d_out = cuda.to_device(out)

threads_per_block = 256
blocks_per_grid = (n + threads_per_block - 1) // threads_per_block
vector_add[blocks_per_grid, threads_per_block](d_a, d_b, d_out)

d_out.copy_to_host(out)
print("Vector add sample:", out[:5])

@cuda.jit
def matmul(A, B, C):
    row, col = cuda.grid(2)
    if row < C.shape[0] and col < C.shape[1]:
        tmp = 0.0
        for k in range(A.shape[1]):
            tmp += A[row, k] * B[k, col]
        C[row, col] = tmp

N = 512
A = np.random.rand(N, N).astype(np.float32)
B = np.random.rand(N, N).astype(np.float32)
C = np.zeros((N, N), dtype=np.float32)

d_A = cuda.to_device(A)
d_B = cuda.to_device(B)
d_C = cuda.to_device(C)

threads_per_block = (16, 16)
blocks_per_grid_x = (N + threads_per_block[0] - 1) // threads_per_block[0]
blocks_per_grid_y = (N + threads_per_block[1] - 1) // threads_per_block[1]
matmul[(blocks_per_grid_x, blocks_per_grid_y), threads_per_block](d_A, d_B, d_C)

C = d_C.copy_to_host()
print("Matmul C[0:2,0:2]:\n", C[:2, :2])
