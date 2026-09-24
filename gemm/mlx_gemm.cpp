#include "gemm/mlx_gemm.h"

#include "gemm/matrix.h"

namespace mx = mlx::core;

namespace {

mx::Shape shape_of(const HostMatrix& matrix) {
    return {static_cast<mx::ShapeElem>(matrix.rows), static_cast<mx::ShapeElem>(matrix.cols)};
}

}  // namespace

MLXGemm::MLXGemm(const HostMatrix& A, const HostMatrix& B)
    : gpu_(mx::default_stream(mx::Device::gpu)),
      left_(mx::copy(mx::array(A.data(), shape_of(A), mx::float32), gpu_)),
      right_(mx::copy(mx::array(B.data(), shape_of(B), mx::float32), gpu_)) {
    left_.eval();
    right_.eval();
    mx::synchronize(gpu_);
}

mx::array MLXGemm::evaluate() const {
    auto result = mx::matmul(left_, right_, gpu_);
    result.eval();
    mx::synchronize(gpu_);
    return result;
}

void MLXGemm::run() const { (void)evaluate(); }

std::vector<float> MLXGemm::output() const {
    auto host = mx::copy(evaluate(), mx::Device::cpu);
    host.eval();
    mx::synchronize(mx::default_stream(mx::Device::cpu));
    const float* data = host.data<float>();
    return {data, data + host.size()};
}
