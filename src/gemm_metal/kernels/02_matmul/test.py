from __future__ import annotations

import mlx.core as mx
from naive import matmul_naive
from reference import matmul_ref
from tiled import matmul_tiled_16, matmul_tiled_32

from gemm_metal.harness import assert_close

KERNELS = {
    "naive": matmul_naive,
    "tiled_16": matmul_tiled_16,
    "tiled_32": matmul_tiled_32,
}


def randn(shape: tuple[int, int], seed: int) -> mx.array:
    mx.random.seed(seed)
    value = mx.random.normal(shape, dtype=mx.float32)
    mx.eval(value)
    return value


if __name__ == "__main__":
    cases = [
        (1, 1, 1),
        (16, 16, 16),
        (31, 29, 17),
        (128, 64, 256),
        (256, 256, 256),
    ]

    for index, (M, N, K) in enumerate(cases):
        A = randn((M, K), seed=index * 2)
        B = randn((K, N), seed=index * 2 + 1)
        expected = matmul_ref(A, B, M, N, K)

        for name, fn in KERNELS.items():
            actual = fn(A, B, M, N, K)
            assert_close(actual, expected, rtol=1e-4, atol=1e-3)
            print(f"{name} correct at shape ({M}, {K}) x ({K}, {N})")
