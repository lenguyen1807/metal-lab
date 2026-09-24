#include "gemm/mlx_gemm.h"

#include <limits>
#include <optional>
#include <stdexcept>

#include "gemm/matrix.h"
#include "mlx/mlx.h"

namespace mx = mlx::core;

namespace {

mx::Shape shape_of(const HostMatrix& matrix)
{
  if (matrix.rows > static_cast<size_t>(std::numeric_limits<mx::ShapeElem>::max())
      || matrix.cols > static_cast<size_t>(std::numeric_limits<mx::ShapeElem>::max())) {
    throw std::invalid_argument("Matrix dimension exceeds MLX shape limit");
  }
  return {static_cast<mx::ShapeElem>(matrix.rows),
          static_cast<mx::ShapeElem>(matrix.cols)};
}

}  // namespace

struct MLXGemm::Impl
{
  mx::Stream gpu = mx::default_stream(mx::Device::gpu);
  mx::array left;
  mx::array right;
  std::optional<mx::array> result;

  Impl(const HostMatrix& A, const HostMatrix& B)
      : left(mx::copy(mx::array(A.data(), shape_of(A), mx::float32), gpu))
      , right(mx::copy(mx::array(B.data(), shape_of(B), mx::float32), gpu))
  {
    left.eval();
    right.eval();
    mx::synchronize(gpu);
  }
};

MLXGemm::MLXGemm(const HostMatrix& A, const HostMatrix& B)
{
  if (A.cols != B.rows) {
    throw std::invalid_argument("Invalid MLX GEMM matrix dimensions");
  }
  impl_ = std::make_unique<Impl>(A, B);
}

MLXGemm::~MLXGemm() = default;

void MLXGemm::run()
{
  impl_->result = mx::matmul(impl_->left, impl_->right, impl_->gpu);
  impl_->result->eval();
  mx::synchronize(impl_->gpu);
}

std::vector<float> MLXGemm::output() const
{
  if (!impl_->result) {
    throw std::logic_error("MLX GEMM has not run");
  }
  auto host = mx::copy(*impl_->result, mx::Device::cpu);
  host.eval();
  mx::synchronize(mx::default_stream(mx::Device::cpu));
  const float* data = host.data<float>();
  if (!data) {
    throw std::runtime_error("MLX GEMM output is unavailable");
  }
  return {data, data + host.size()};
}
