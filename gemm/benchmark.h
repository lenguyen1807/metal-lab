#pragma once

#include <memory>
#include <string>

#include "Metal/MTLTypes.hpp"

class DeviceMatrix;
class Kernel;
class MPSGemm;
struct MetalContext;

class BenchmarkMgr
{
public:
  BenchmarkMgr();
  ~BenchmarkMgr();

  void run_benchmark_suite(const std::string& kernel_name, bool smoke = false);

private:
  double start_kernel(const DeviceMatrix& A,
                      const DeviceMatrix& B,
                      DeviceMatrix& C,
                      Kernel* kernel,
                      MPSGemm* mps,
                      MTL::Size grid_size,
                      MTL::Size block_size);

  double run_multiples(const DeviceMatrix& A,
                       const DeviceMatrix& B,
                       DeviceMatrix& C,
                       Kernel* kernel,
                       MPSGemm* mps,
                       MTL::Size grid_size,
                       MTL::Size block_size);

  std::unique_ptr<MetalContext> ctx_;
};
