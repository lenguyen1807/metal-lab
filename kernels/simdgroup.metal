#include <metal_stdlib>
#include <metal_simdgroup_matrix>

using namespace metal;

struct GemmShape {
  uint M, N, K;
};

// One SIMD group owns one 8x8 output tile. This is the smallest matrix
// primitive and a useful baseline before adding a larger register tile.
kernel void simdgroup_8x8(device const float* a [[buffer(0)]],
                          device const float* b [[buffer(1)]],
                          device float* c [[buffer(2)]],
                          constant GemmShape& p [[buffer(3)]],
                          uint2 group [[threadgroup_position_in_grid]]) {
  const uint row = group.y * 8;
  const uint col = group.x * 8;
  simdgroup_float8x8 acc = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
  for (uint k = 0; k < p.K; k += 8) {
    simdgroup_float8x8 ta, tb, next;
    simdgroup_load(ta, a + row * p.K + k, p.K);
    simdgroup_load(tb, b + k * p.N + col, p.N);
    simdgroup_multiply_accumulate(next, ta, tb, acc);
    acc = next;
  }
  simdgroup_store(acc, c + row * p.N + col, p.N);
}

// Four SIMD groups cover a 32x32 tile. Each group keeps four 8x8 output
// fragments in registers, reusing two A and two B fragments per K step.
kernel void simdgroup_32x32(device const float* a [[buffer(0)]],
                            device const float* b [[buffer(1)]],
                            device float* c [[buffer(2)]],
                            constant GemmShape& p [[buffer(3)]],
                            uint2 group [[threadgroup_position_in_grid]],
                            uint sg [[simdgroup_index_in_threadgroup]]) {
  const uint row = group.y * 32 + (sg / 2) * 16;
  const uint col = group.x * 32 + (sg % 2) * 16;
  simdgroup_float8x8 c00 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
  simdgroup_float8x8 c01 = c00, c10 = c00, c11 = c00;
  for (uint k = 0; k < p.K; k += 8) {
    simdgroup_float8x8 a0, a1, b0, b1, next;
    simdgroup_load(a0, a + row * p.K + k, p.K);
    simdgroup_load(a1, a + (row + 8) * p.K + k, p.K);
    simdgroup_load(b0, b + k * p.N + col, p.N);
    simdgroup_load(b1, b + k * p.N + col + 8, p.N);
    simdgroup_multiply_accumulate(next, a0, b0, c00);
    c00 = next;
    simdgroup_multiply_accumulate(next, a0, b1, c01);
    c01 = next;
    simdgroup_multiply_accumulate(next, a1, b0, c10);
    c10 = next;
    simdgroup_multiply_accumulate(next, a1, b1, c11);
    c11 = next;
  }
  simdgroup_store(c00, c + row * p.N + col, p.N);
  simdgroup_store(c01, c + row * p.N + col + 8, p.N);
  simdgroup_store(c10, c + (row + 8) * p.N + col, p.N);
  simdgroup_store(c11, c + (row + 8) * p.N + col + 8, p.N);
}

// Stage a 32x32 A tile and a 32x32 B tile in threadgroup memory. The padding
// changes each row's bank alignment; the barriers protect tile reuse.
kernel void simdgroup_32x32_shared(device const float* a [[buffer(0)]],
                                   device const float* b [[buffer(1)]],
                                   device float* c [[buffer(2)]],
                                   constant GemmShape& p [[buffer(3)]],
                                   uint2 group [[threadgroup_position_in_grid]],
                                   uint sg [[simdgroup_index_in_threadgroup]],
                                   uint tid [[thread_index_in_threadgroup]]) {
  threadgroup float As[32 * 33];
  threadgroup float Bs[32 * 33];
  const uint block_row = group.y * 32;
  const uint block_col = group.x * 32;
  const uint subrow = (sg / 2) * 16;
  const uint subcol = (sg % 2) * 16;
  simdgroup_float8x8 c00 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
  simdgroup_float8x8 c01 = c00, c10 = c00, c11 = c00;

  for (uint kb = 0; kb < p.K; kb += 32) {
    for (uint i = tid; i < 32 * 32; i += 128) {
      const uint r = i / 32;
      const uint col = i % 32;
      As[r * 33 + col] = a[(block_row + r) * p.K + kb + col];
      Bs[r * 33 + col] = b[(kb + r) * p.N + block_col + col];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint k = 0; k < 32; k += 8) {
      simdgroup_float8x8 a0, a1, b0, b1, next;
      simdgroup_load(a0, As + subrow * 33 + k, 33);
      simdgroup_load(a1, As + (subrow + 8) * 33 + k, 33);
      simdgroup_load(b0, Bs + k * 33 + subcol, 33);
      simdgroup_load(b1, Bs + k * 33 + subcol + 8, 33);
      simdgroup_multiply_accumulate(next, a0, b0, c00);
      c00 = next;
      simdgroup_multiply_accumulate(next, a0, b1, c01);
      c01 = next;
      simdgroup_multiply_accumulate(next, a1, b0, c10);
      c10 = next;
      simdgroup_multiply_accumulate(next, a1, b1, c11);
      c11 = next;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  const uint row = block_row + subrow;
  const uint col = block_col + subcol;
  simdgroup_store(c00, c + row * p.N + col, p.N);
  simdgroup_store(c01, c + row * p.N + col + 8, p.N);
  simdgroup_store(c10, c + (row + 8) * p.N + col, p.N);
  simdgroup_store(c11, c + (row + 8) * p.N + col + 8, p.N);
}
