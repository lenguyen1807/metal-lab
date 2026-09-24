#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;
using namespace mpp::tensor_ops;

struct GemmShape {
  uint M, N, K;
};

// The host enables each variant only for its aligned dimensions.
uint compact_even_bits(uint value) {
  value &= 0x55555555u;
  value = (value ^ (value >> 1)) & 0x33333333u;
  value = (value ^ (value >> 2)) & 0x0f0f0f0fu;
  value = (value ^ (value >> 4)) & 0x00ff00ffu;
  return (value ^ (value >> 8)) & 0x0000ffffu;
}

template <int TM, int TN, int BK, bool Morton = false>
kernel void matmul_tensorops_sync(
    device float* a [[buffer(0)]], device float* b [[buffer(1)]],
    device float* c [[buffer(2)]], constant GemmShape& p [[buffer(3)]],
    uint2 group [[threadgroup_position_in_grid]]) {
  if constexpr (Morton) {
    const uint tiles_x = p.N / TN;
    const uint tiles_y = p.M / TM;
    const uint index = group.x;
    if (tiles_x == tiles_y && (tiles_x & (tiles_x - 1)) == 0) {
      group = uint2(compact_even_bits(index), compact_even_bits(index >> 1));
    } else {
      group = uint2(index % tiles_x, index / tiles_x);
    }
  }
  auto A = tensor<device float, dextents<int, 2>, tensor_inline>(
      a, dextents<int, 2>((int)p.K, (int)p.M));
  auto B = tensor<device float, dextents<int, 2>, tensor_inline>(
      b, dextents<int, 2>((int)p.N, (int)p.K));
  auto C = tensor<device float, dextents<int, 2>, tensor_inline>(
      c, dextents<int, 2>((int)p.N, (int)p.M));
  const int row = (int)group.y * TM;
  const int col = (int)group.x * TN;
  constexpr auto desc = matmul2d_descriptor(
      TM, TN, BK, false, false, false,
      matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, execution_simdgroups<4>> op;
  auto first_a = A.slice<BK, TM>(0, row);
  auto first_b = B.slice<TN, BK>(col, 0);
  auto acc = op.template get_destination_cooperative_tensor<
      decltype(first_a), decltype(first_b), float>();

  for (uint16_t i = 0; i < acc.get_capacity(); ++i) {
    if (acc.is_valid_element(i)) acc[i] = 0.0f;
  }
  for (int k = 0; k < (int)p.K; k += BK) {
    threadgroup_barrier(mem_flags::mem_none);
    auto ta = A.slice<BK, TM>(k, row);
    auto tb = B.slice<TN, BK>(col, k);
    op.run(ta, tb, acc);
  }
  auto tc = C.slice<TN, TM>(col, row);
  acc.store(tc);
}

template [[host_name("tensorops_sync_k128")]]
kernel void matmul_tensorops_sync<64, 64, 128>(device float* a [[buffer(0)]],
    device float* b [[buffer(1)]], device float* c [[buffer(2)]],
    constant GemmShape& p [[buffer(3)]],
    uint2 group [[threadgroup_position_in_grid]]);

template [[host_name("tensorops_sync_k256")]]
kernel void matmul_tensorops_sync<64, 64, 256>(device float* a [[buffer(0)]],
    device float* b [[buffer(1)]], device float* c [[buffer(2)]],
    constant GemmShape& p [[buffer(3)]],
    uint2 group [[threadgroup_position_in_grid]]);

template [[host_name("tensorops_sync_k512")]]
kernel void matmul_tensorops_sync<64, 64, 512>(device float* a [[buffer(0)]],
    device float* b [[buffer(1)]], device float* c [[buffer(2)]],
    constant GemmShape& p [[buffer(3)]],
    uint2 group [[threadgroup_position_in_grid]]);

template [[host_name("tensorops_sync_k1024")]]
kernel void matmul_tensorops_sync<64, 64, 1024>(device float* a [[buffer(0)]],
    device float* b [[buffer(1)]], device float* c [[buffer(2)]],
    constant GemmShape& p [[buffer(3)]],
    uint2 group [[threadgroup_position_in_grid]]);

template [[host_name("tensorops_sync_64x128_k512")]]
kernel void matmul_tensorops_sync<64, 128, 512>(device float* a [[buffer(0)]],
    device float* b [[buffer(1)]], device float* c [[buffer(2)]],
    constant GemmShape& p [[buffer(3)]],
    uint2 group [[threadgroup_position_in_grid]]);

template [[host_name("tensorops_sync_32x64_k512")]]
kernel void matmul_tensorops_sync<32, 64, 512>(device float* a [[buffer(0)]],
    device float* b [[buffer(1)]], device float* c [[buffer(2)]],
    constant GemmShape& p [[buffer(3)]],
    uint2 group [[threadgroup_position_in_grid]]);

template [[host_name("tensorops_sync_64x128_k256")]]
kernel void matmul_tensorops_sync<64, 128, 256>(device float* a [[buffer(0)]],
    device float* b [[buffer(1)]], device float* c [[buffer(2)]],
    constant GemmShape& p [[buffer(3)]],
    uint2 group [[threadgroup_position_in_grid]]);

template [[host_name("tensorops_sync_k256_morton")]]
kernel void matmul_tensorops_sync<64, 64, 256, true>(device float* a [[buffer(0)]],
    device float* b [[buffer(1)]], device float* c [[buffer(2)]],
    constant GemmShape& p [[buffer(3)]],
    uint2 group [[threadgroup_position_in_grid]]);
