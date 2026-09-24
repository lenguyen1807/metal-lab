#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "Foundation/NSAutoreleasePool.hpp"
#include "Metal/MTLCommandBuffer.hpp"
#include "Metal/MTLComputeCommandEncoder.hpp"
#include "gemm/benchmark.h"
#include "gemm/kernel.h"
#include "gemm/matrix.h"
#include "gemm/metal_mgr.h"
#include "gemm/mps_gemm.h"
#include "gemm/params.h"
#include "gemm/utils.h"

BenchmarkMgr::BenchmarkMgr() : ctx_(std::make_unique<MetalContext>()) {}
BenchmarkMgr::~BenchmarkMgr() = default;

void BenchmarkMgr::run_benchmark_suite(const std::string& kernel_name,
                                       bool smoke)
{
  if (kernel_name != "naive" && kernel_name != "mps") {
    throw std::invalid_argument("Unknown kernel: " + kernel_name);
  }

  std::unique_ptr<Kernel> kernel;
  if (kernel_name == "naive") {
    kernel = std::make_unique<Kernel>(kernel_name, ctx_->device.get());
  }

  CSVWriter writer(kernel_name + (smoke ? "_smoke.csv" : ".csv"));
  writer << "M" << "N" << "K" << "Time (ms)" << "GFLOPS"
         << "Max abs error" << endrow;

  const std::vector<GemmShape> smoke_shapes = {
      {1, 1, 1}, {7, 13, 5}, {17, 31, 9}, {32, 32, 32},
      {65, 37, 33}, {257, 263, 255}};
  const auto& shapes = smoke ? smoke_shapes : BENCHMARK_SHAPES;

  std::cout << "Running " << kernel_name << (smoke ? " smoke" : " benchmark")
            << " on " << ctx_->device->name()->utf8String() << "\n";

  for (const auto& shape : shapes) {
    const size_t M = shape.M, N = shape.N, K = shape.K;
    HostMatrix A = HostMatrix::random(0.f, 1.f, M, K);
    HostMatrix B = HostMatrix::random(0.f, 1.f, K, N);
    DeviceMatrix d_A(ctx_->device.get(), M, K);
    DeviceMatrix d_B(ctx_->device.get(), K, N);
    DeviceMatrix d_C(ctx_->device.get(), M, N);
    copy(A, d_A);
    copy(B, d_B);

    std::unique_ptr<MPSGemm> mps;
    MTL::Size block = MTL::Size::Make(1, 1, 1);
    MTL::Size grid = MTL::Size::Make(1, 1, 1);
    if (kernel) {
      const auto& config = kernel->config();
      block = MTL::Size::Make(config.block_width, config.block_height, 1);
      grid = MTL::Size::Make(
          (N + config.tile_width * config.block_width - 1)
              / (config.tile_width * config.block_width),
          (M + config.tile_height * config.block_height - 1)
              / (config.tile_height * config.block_height),
          1);
    } else {
      mps = std::make_unique<MPSGemm>(ctx_->device.get(), d_A, d_B, d_C);
    }

    // Both paths use the same preallocated, packed row-major FP32 buffers.
    start_kernel(d_A, d_B, d_C, kernel.get(), mps.get(), grid, block);

    double max_abs_error = std::numeric_limits<double>::quiet_NaN();
    if (smoke || (M <= 1024 && N <= 1024 && K <= 1024)) {
      HostMatrix actual(M, N);
      HostMatrix expected(M, N);
      copy(d_C, actual);
      matmul_cpu(A, B, expected);
      const auto comparison = compare(actual, expected);
      max_abs_error = comparison.max_abs_error;
      if (comparison.mismatches != 0) {
        throw std::runtime_error(
            "Incorrect " + kernel_name + " result for (" + std::to_string(M)
            + "," + std::to_string(N) + "," + std::to_string(K)
            + "): mismatches=" + std::to_string(comparison.mismatches)
            + ", max_abs_error=" + std::to_string(max_abs_error));
      }
    }

    const double time_ms = run_multiples(
        d_A, d_B, d_C, kernel.get(), mps.get(), grid, block);
    const double gflops = matmul_time_to_gflops(M, N, K, time_ms);
    std::cout << "(" << M << ", " << N << ", " << K << "): " << time_ms
              << " ms, " << gflops << " GFLOP/s, max abs error "
              << max_abs_error << "\n";
    writer << M << N << K << time_ms << gflops << max_abs_error << endrow;
  }
}

double BenchmarkMgr::start_kernel(const DeviceMatrix& A,
                                  const DeviceMatrix& B,
                                  DeviceMatrix& C,
                                  Kernel* kernel,
                                  MPSGemm* mps,
                                  MTL::Size grid_size,
                                  MTL::Size block_size)
{
  if ((kernel == nullptr) == (mps == nullptr)) {
    throw std::invalid_argument("Select exactly one GEMM implementation");
  }

  auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
  auto command_buffer = NS::RetainPtr(ctx_->cmd_queue->commandBuffer());
  if (!command_buffer) {
    throw std::runtime_error("Cannot create Metal command buffer");
  }

  if (mps) {
    mps->encode(command_buffer.get());
  } else {
    MatmulParams params{static_cast<uint32_t>(C.rows),
                        static_cast<uint32_t>(C.cols),
                        static_cast<uint32_t>(A.cols),
                        1.f,
                        0.f,
                        static_cast<uint32_t>(block_size.width),
                        static_cast<uint32_t>(block_size.height)};
    auto encoder = NS::RetainPtr(command_buffer->computeCommandEncoder());
    if (!encoder) {
      throw std::runtime_error("Cannot create Metal compute encoder");
    }
    encoder->setComputePipelineState(kernel->pipeline());
    encoder->setBuffer(A.data(), 0, 0);
    encoder->setBuffer(B.data(), 0, 1);
    encoder->setBuffer(C.data(), 0, 2);
    encoder->setBytes(&params, sizeof(params), 3);
    encoder->dispatchThreadgroups(grid_size, block_size);
    encoder->endEncoding();
  }

  command_buffer->commit();
  command_buffer->waitUntilCompleted();
  if (command_buffer->status() != MTL::CommandBufferStatusCompleted) {
    const auto* error = command_buffer->error();
    const std::string message = error
        ? error->localizedDescription()->utf8String()
        : "unknown Metal error";
    throw std::runtime_error("Metal GEMM failed: " + message);
  }

  const double elapsed_ms =
      (command_buffer->GPUEndTime() - command_buffer->GPUStartTime()) * 1000.0;
  if (elapsed_ms <= 0.0) {
    throw std::runtime_error("GPU command-buffer timestamps unavailable");
  }
  return elapsed_ms;
}

double BenchmarkMgr::run_multiples(const DeviceMatrix& A,
                                   const DeviceMatrix& B,
                                   DeviceMatrix& C,
                                   Kernel* kernel,
                                   MPSGemm* mps,
                                   MTL::Size grid_size,
                                   MTL::Size block_size)
{
  double total_ms = 0.0;
  for (int i = 0; i < BENCHMARK_TIME; ++i) {
    total_ms += start_kernel(A, B, C, kernel, mps, grid_size, block_size);
  }
  return total_ms / BENCHMARK_TIME;
}
