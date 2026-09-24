#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

class DeviceMatrix;
class Kernel;
class MPSGemm;
struct MetalContext;

struct BenchmarkOptions
{
  bool smoke = false;
  bool test_only = false;
  size_t iterations = 10;
  std::vector<std::string> kernels;
};

class BenchmarkMgr
{
public:
  BenchmarkMgr();
  ~BenchmarkMgr();

  void run(const BenchmarkOptions& options);

private:
  double start_kernel(const DeviceMatrix& A,
                      const DeviceMatrix& B,
                      DeviceMatrix& C,
                      Kernel* kernel,
                      MPSGemm* mps);

  double run_multiples(const DeviceMatrix& A,
                       const DeviceMatrix& B,
                       DeviceMatrix& C,
                       Kernel* kernel,
                       MPSGemm* mps,
                       size_t iterations);

  std::unique_ptr<MetalContext> ctx_;
};
