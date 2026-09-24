#include "gemm/benchmark.h"

#include <algorithm>
#include <cctype>
#include <iomanip>
#include <iostream>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "Foundation/NSAutoreleasePool.hpp"
#include "Metal/MTLCommandBuffer.hpp"
#include "gemm/kernel.h"
#include "gemm/matrix.h"
#include "gemm/metal_mgr.h"
#include "gemm/mps_gemm.h"
#include "gemm/params.h"
#include "gemm/utils.h"

namespace {

const std::vector<GemmShape> smoke_shapes = {
    {1, 1, 1}, {7, 13, 5}, {17, 31, 9},
    {32, 32, 32}, {65, 37, 33}, {257, 263, 255}};

std::string device_tag(const char* name)
{
  std::string tag;
  for (const unsigned char ch : std::string(name)) {
    if (std::isalnum(ch)) {
      tag.push_back(static_cast<char>(std::tolower(ch)));
    } else if (!tag.empty() && tag.back() != '_') {
      tag.push_back('_');
    }
  }
  if (!tag.empty() && tag.back() == '_') {
    tag.pop_back();
  }
  return tag;
}

std::string number(double value, int precision)
{
  std::ostringstream out;
  out << std::fixed << std::setprecision(precision) << value;
  return out.str();
}

void print_result(const GemmShape& shape,
                  const std::string& kernel,
                  const std::string& status,
                  std::optional<double> median_ms,
                  std::optional<double> baseline_ms,
                  CSVWriter& writer)
{
  const std::string ms = median_ms ? number(*median_ms, 4) : "-";
  const std::string gflops = median_ms
      ? number(matmul_time_to_gflops(shape.M, shape.N, shape.K, *median_ms), 2)
      : "-";
  // For the same shape, throughput ratio equals MPS time / variant time.
  const std::string relative = median_ms && baseline_ms
      ? number(*baseline_ms / *median_ms, 2) + "x"
      : "-";
  std::cout << std::left << std::setw(12) << kernel
            << std::right << std::setw(12) << ms
            << std::setw(13) << gflops
            << std::setw(11) << relative
            << std::setw(14) << status << '\n';

  writer << shape.M << shape.N << shape.K << kernel << status
         << (median_ms ? number(*median_ms, 6) : "")
         << (median_ms ? number(matmul_time_to_gflops(
                shape.M, shape.N, shape.K, *median_ms), 3) : "")
         << (median_ms && baseline_ms
             ? number(*baseline_ms / *median_ms, 4) : "")
         << endrow;
}

}  // namespace

BenchmarkMgr::BenchmarkMgr() : ctx_(std::make_unique<MetalContext>()) {}
BenchmarkMgr::~BenchmarkMgr() = default;

void BenchmarkMgr::run(const BenchmarkOptions& options)
{
  if (options.iterations == 0) {
    throw std::invalid_argument("Iterations must be positive");
  }
  std::vector<std::unique_ptr<Kernel>> native;
  for (const auto& spec : kernel_specs()) {
    if (options.kernels.empty()
        || std::find(options.kernels.begin(), options.kernels.end(), spec.name)
            != options.kernels.end()) {
      native.push_back(std::make_unique<Kernel>(spec, ctx_->device.get()));
    }
  }
  for (const auto& requested : options.kernels) {
    if (std::none_of(native.begin(), native.end(), [&](const auto& kernel) {
          return requested == kernel->name();
        })) {
      throw std::invalid_argument("Unknown custom kernel: " + requested);
    }
  }

  const std::string gpu_name = ctx_->device->name()->utf8String();
  std::cout << "GPU: " << gpu_name
            << " | baseline: MPSMatrixMultiplication"
            << " | FP32 row-major NN\n";
  std::unique_ptr<CSVWriter> writer;
  std::string filename;
  if (!options.test_only) {
    filename = std::string("bench_")
        + (options.smoke ? "smoke_" : "")
        + device_tag(gpu_name.c_str()) + ".csv";
    writer = std::make_unique<CSVWriter>(filename);
    *writer << "M" << "N" << "K" << "kernel" << "status"
            << "median_ms" << "gflops" << "vs_mps" << endrow;
  }

  const auto& shapes = options.smoke || options.test_only
      ? smoke_shapes : BENCHMARK_SHAPES;
  for (const GemmShape& shape : shapes) {
    const size_t M = shape.M, N = shape.N, K = shape.K;
    HostMatrix A = HostMatrix::random(0.f, 1.f, M, K);
    HostMatrix B = HostMatrix::random(0.f, 1.f, K, N);
    DeviceMatrix d_A(ctx_->device.get(), M, K);
    DeviceMatrix d_B(ctx_->device.get(), K, N);
    copy(A, d_A);
    copy(B, d_B);

    DeviceMatrix d_baseline(ctx_->device.get(), M, N);
    MPSGemm baseline(ctx_->device.get(), d_A, d_B, d_baseline);
    start_kernel(d_A, d_B, d_baseline, nullptr, &baseline);
    struct PreparedKernel
    {
      Kernel* kernel;
      std::unique_ptr<DeviceMatrix> output;
    };
    std::vector<PreparedKernel> prepared;
    for (const auto& kernel : native) {
      if (!kernel->supports(shape)) {
        prepared.push_back({kernel.get(), nullptr});
        continue;
      }
      auto output = std::make_unique<DeviceMatrix>(ctx_->device.get(), M, N);
      start_kernel(d_A, d_B, *output, kernel.get(), nullptr);
      const Comparison comparison = compare(*output, d_baseline);
      if (comparison.mismatches != 0) {
        throw std::runtime_error(
            std::string(kernel->name()) + " disagrees with MPS at "
            + std::to_string(M) + "x" + std::to_string(N) + "x"
            + std::to_string(K) + ": mismatches="
            + std::to_string(comparison.mismatches) + ", max_abs_error="
            + std::to_string(comparison.max_abs_error));
      }
      prepared.push_back({kernel.get(), std::move(output)});
    }
    if (options.test_only) {
      std::cout << "ok  " << M << "x" << N << "x" << K << '\n';
      continue;
    }

    // Every variant is now warmed and checked. Timing begins only here.
    const double baseline_ms = run_multiples(
        d_A, d_B, d_baseline, nullptr, &baseline, options.iterations);
    std::cout << "\n### " << M << "x" << N << "x" << K << "\n"
              << std::left << std::setw(12) << "Kernel"
              << std::right << std::setw(12) << "median ms"
              << std::setw(13) << "GFLOP/s"
              << std::setw(11) << "vs MPS"
              << std::setw(14) << "status" << '\n';
    print_result(shape, "mps", "baseline", baseline_ms, baseline_ms, *writer);
    for (const auto& item : prepared) {
      if (!item.output) {
        print_result(shape, item.kernel->name(), "unsupported",
                     std::nullopt, baseline_ms, *writer);
        continue;
      }
      const double median_ms = run_multiples(
          d_A, d_B, *item.output, item.kernel, nullptr, options.iterations);
      print_result(shape, item.kernel->name(), "ok", median_ms,
                   baseline_ms, *writer);
    }
  }

  if (writer) {
    std::cout << "\nCSV: " << OUTPUTS_PATH << filename << '\n';
  }
}

double BenchmarkMgr::start_kernel(const DeviceMatrix& A,
                                  const DeviceMatrix& B,
                                  DeviceMatrix& C,
                                  Kernel* kernel,
                                  MPSGemm* mps)
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
    kernel->encode(command_buffer.get(), A, B, C);
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
                                   size_t iterations)
{
  std::vector<double> samples;
  samples.reserve(iterations);
  for (size_t i = 0; i < iterations; ++i) {
    samples.push_back(start_kernel(A, B, C, kernel, mps));
  }
  std::sort(samples.begin(), samples.end());
  const size_t middle = samples.size() / 2;
  return samples.size() % 2
      ? samples[middle]
      : (samples[middle - 1] + samples[middle]) / 2.0;
}
