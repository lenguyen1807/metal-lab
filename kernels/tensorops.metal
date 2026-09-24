#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;
using namespace mpp::tensor_ops;

struct GemmShape {
  uint M, N, K;
};

// Packed row-major matrices have innermost-first extents (K,M), (N,K), (N,M).
// Metal 4 TensorOps use the GPU neural accelerators on M5.
uint compact_even_bits(uint value) {
  value &= 0x55555555u;
  value = (value ^ (value >> 1)) & 0x33333333u;
  value = (value ^ (value >> 2)) & 0x0f0f0f0fu;
  value = (value ^ (value >> 4)) & 0x00ff00ffu;
  return (value ^ (value >> 8)) & 0x0000ffffu;
}

template <int TM, int TN, int SG, bool Morton>
kernel void matmul_tensorops(device float* a [[buffer(0)]],
                             device float* b [[buffer(1)]],
                             device float* c [[buffer(2)]],
                             constant GemmShape& p [[buffer(3)]],
                             uint2 group [[threadgroup_position_in_grid]]) {
  auto A = tensor<device float, dextents<int, 2>, tensor_inline>(
      a, dextents<int, 2>((int)p.K, (int)p.M));
  auto B = tensor<device float, dextents<int, 2>, tensor_inline>(
      b, dextents<int, 2>((int)p.N, (int)p.K));
  auto C = tensor<device float, dextents<int, 2>, tensor_inline>(
      c, dextents<int, 2>((int)p.N, (int)p.M));

  constexpr auto desc = matmul2d_descriptor(TM, TN, dynamic_length_v<int>);
  matmul2d<desc, execution_simdgroups<SG>> op;
  if constexpr (Morton) {
    const uint tiles_x = (p.N + TN - 1) / TN;
    const uint tiles_y = (p.M + TM - 1) / TM;
    const uint index = group.x;
    if (tiles_x == tiles_y && (tiles_x & (tiles_x - 1)) == 0) {
      group = uint2(compact_even_bits(index), compact_even_bits(index >> 1));
    } else {
      group = uint2(index % tiles_x, index / tiles_x);
    }
  }
  const int row = (int)group.y * TM;
  const int col = (int)group.x * TN;

  if (row + TM <= (int)p.M && col + TN <= (int)p.N) {
    auto ta = A.slice<dynamic_extent, TM>(0, row);
    auto tb = B.slice<TN, dynamic_extent>(col, 0);
    auto tc = C.slice<TN, TM>(col, row);
    op.run(ta, tb, tc);
  } else {
    auto ta = A.slice(0, row);
    auto tb = B.slice(col, 0);
    auto tc = C.slice(col, row);
    op.run(ta, tb, tc);
  }
}

template [[host_name("tensorops_64x64")]]
kernel void matmul_tensorops<64, 64, 4, false>(device float* a [[buffer(0)]],
    device float* b [[buffer(1)]], device float* c [[buffer(2)]],
    constant GemmShape& p [[buffer(3)]],
    uint2 group [[threadgroup_position_in_grid]]);

template [[host_name("tensorops_64x128")]]
kernel void matmul_tensorops<64, 128, 4, false>(device float* a [[buffer(0)]],
    device float* b [[buffer(1)]], device float* c [[buffer(2)]],
    constant GemmShape& p [[buffer(3)]],
    uint2 group [[threadgroup_position_in_grid]]);

template [[host_name("tensorops_128x64")]]
kernel void matmul_tensorops<128, 64, 4, false>(device float* a [[buffer(0)]],
    device float* b [[buffer(1)]], device float* c [[buffer(2)]],
    constant GemmShape& p [[buffer(3)]],
    uint2 group [[threadgroup_position_in_grid]]);

template [[host_name("tensorops_128x128")]]
kernel void matmul_tensorops<128, 128, 4, false>(device float* a [[buffer(0)]],
    device float* b [[buffer(1)]], device float* c [[buffer(2)]],
    constant GemmShape& p [[buffer(3)]],
    uint2 group [[threadgroup_position_in_grid]]);

template [[host_name("tensorops_64x64_morton")]]
kernel void matmul_tensorops<64, 64, 4, true>(device float* a [[buffer(0)]],
    device float* b [[buffer(1)]], device float* c [[buffer(2)]],
    constant GemmShape& p [[buffer(3)]],
    uint2 group [[threadgroup_position_in_grid]]);

template [[host_name("tensorops_32x32_sg1")]]
kernel void matmul_tensorops<32, 32, 1, false>(device float* a [[buffer(0)]],
    device float* b [[buffer(1)]], device float* c [[buffer(2)]],
    constant GemmShape& p [[buffer(3)]],
    uint2 group [[threadgroup_position_in_grid]]);

template [[host_name("tensorops_32x32_sg1_morton")]]
kernel void matmul_tensorops<32, 32, 1, true>(device float* a [[buffer(0)]],
    device float* b [[buffer(1)]], device float* c [[buffer(2)]],
    constant GemmShape& p [[buffer(3)]],
    uint2 group [[threadgroup_position_in_grid]]);

template [[host_name("tensorops_32x64_sg1")]]
kernel void matmul_tensorops<32, 64, 1, false>(device float* a [[buffer(0)]],
    device float* b [[buffer(1)]], device float* c [[buffer(2)]],
    constant GemmShape& p [[buffer(3)]],
    uint2 group [[threadgroup_position_in_grid]]);

template [[host_name("tensorops_64x32_sg1")]]
kernel void matmul_tensorops<64, 32, 1, false>(device float* a [[buffer(0)]],
    device float* b [[buffer(1)]], device float* c [[buffer(2)]],
    constant GemmShape& p [[buffer(3)]],
    uint2 group [[threadgroup_position_in_grid]]);
