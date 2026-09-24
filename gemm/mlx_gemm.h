#pragma once

#include <vector>

#include "mlx/mlx.h"

class HostMatrix;

class MLXGemm {
   public:
    MLXGemm(const HostMatrix& A, const HostMatrix& B);

    void run() const;
    std::vector<float> output() const;

   private:
    mlx::core::array evaluate() const;

    mlx::core::Stream gpu_;
    mlx::core::array left_;
    mlx::core::array right_;
};
