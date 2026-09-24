#include <metal_stdlib>

using namespace metal;

struct GemmShape
{
  uint M;
  uint N;
  uint K;
};
