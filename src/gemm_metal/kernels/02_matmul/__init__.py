from naive import matmul_naive
from reference import matmul_ref
from tiled import matmul_tiled_16, matmul_tiled_32

MATMUL_KERNELS = {
    "mlx": matmul_ref,
    "naive": matmul_naive,
    "tiled_16": matmul_tiled_16,
    "tiled_32": matmul_tiled_32,
}

__all__ = ["MATMUL_KERNELS", "matmul_naive", "matmul_ref", "matmul_tiled_16", "matmul_tiled_32"]
