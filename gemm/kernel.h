#pragma once

#include "Foundation/NSAutoreleasePool.hpp"
#include "Foundation/NSSharedPtr.hpp"
#include "Metal/MTLComputePipeline.hpp"
#include "Metal/MTLDevice.hpp"
#include "Metal/MTLLibrary.hpp"
#include "gemm/utils.h"

struct KernelConfig
{
  // The dimensions of the threadgroup (block) in threads.
  size_t block_width = 32;
  size_t block_height = 32;

  // The dimensions of the output tile processed by a single threadgroup.
  // This is crucial for calculating the grid size correctly.
  size_t tile_width = 1;
  size_t tile_height = 1;
};

class Kernel
{
public:
  Kernel(const std::string& kernel_name, MTL::Device* device)
  {
    if (device == nullptr) {
      throw std::runtime_error("Device for kernel cannot be empty");
    }
    auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());

    // setup kernel configuration
    if (kernel_name == "naive") {
      config_ = {.block_width = 32,
                 .block_height = 32,
                 .tile_width = 1,
                 .tile_height = 1};
    } else {
      throw std::runtime_error("Unknown kernel name for configuration: "
                               + kernel_name);
    }

    NS::Error* error = nullptr;

    // read kernel source
    std::string path = std::string(KERNEL_PATH) + kernel_name + ".metal";
    std::string src = read_file(path);
    auto metal_src =
        NS::String::string(src.c_str(), NS::StringEncoding::UTF8StringEncoding);

    // compile library
    library_ = NS::TransferPtr(device->newLibrary(metal_src, nullptr, &error));

    // check errors
    if (error != nullptr || !library_) {
      const char* msg = error
          ? error->localizedDescription()->utf8String()
          : "Metal returned no library";
      throw std::runtime_error("Cannot create library because: " + std::string(msg));
    }

    // create function
    auto str = NS::String::string(("matmul_" + kernel_name).c_str(),
                                  NS::ASCIIStringEncoding);
    func_ = NS::TransferPtr(library_->newFunction(str));
    if (!func_) {
      throw std::runtime_error("Cannot find Metal function matmul_" + kernel_name);
    }

    // create pipeline for function
    pipeline_ = NS::TransferPtr(device->newComputePipelineState(func_.get(), &error));

    if (error != nullptr || !pipeline_) {
      const char* msg = error
          ? error->localizedDescription()->utf8String()
          : "Metal returned no pipeline";
      throw std::runtime_error("Cannot create library pipeline, error: "
                               + std::string(msg));
    }

  }

  MTL::Library* library() const { return library_.get(); }
  MTL::Function* function() const { return func_.get(); }
  MTL::ComputePipelineState* pipeline() const { return pipeline_.get(); }
  const KernelConfig& config() const { return config_; }

private:
  NS::SharedPtr<MTL::Library> library_;
  NS::SharedPtr<MTL::Function> func_;
  NS::SharedPtr<MTL::ComputePipelineState> pipeline_;
  KernelConfig config_;
};
