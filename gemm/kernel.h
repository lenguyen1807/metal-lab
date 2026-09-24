#pragma once

#include <string>
#include <vector>

#include "Foundation/NSSharedPtr.hpp"
#include "Metal/MTLComputePipeline.hpp"
#include "Metal/MTLDevice.hpp"
#include "Metal/MTLLibrary.hpp"
#include "Metal/MTLTypes.hpp"
#include "gemm/params.h"

namespace MTL {
class CommandBuffer;
class ComputeCommandEncoder;
}
class DeviceMatrix;

struct DispatchPlan
{
  MTL::Size threadgroups;
  MTL::Size threads_per_threadgroup;
};

struct KernelSpec
{
  const char* name;
  const char* function;
  DispatchPlan (*dispatch)(const GemmShape&, const MTL::ComputePipelineState&);
  bool (*supports)(const GemmShape&);
  void (*encode)(MTL::ComputeCommandEncoder&,
                 const DeviceMatrix&,
                 const DeviceMatrix&,
                 DeviceMatrix&,
                 const GemmShape&,
                 const DispatchPlan&);
};

// Each native kernel owns launch geometry, shape constraints, and bindings.
// The benchmark only asks a prepared kernel to encode its work.
const std::vector<KernelSpec>& kernel_specs();

class Kernel
{
public:
  Kernel(KernelSpec spec, MTL::Device* device);

  const char* name() const { return spec_.name; }
  bool supports(const GemmShape& shape) const;
  void encode(MTL::CommandBuffer* command_buffer,
              const DeviceMatrix& A,
              const DeviceMatrix& B,
              DeviceMatrix& C) const;

private:
  KernelSpec spec_;
  NS::SharedPtr<MTL::Library> library_;
  NS::SharedPtr<MTL::Function> function_;
  NS::SharedPtr<MTL::ComputePipelineState> pipeline_;
};
