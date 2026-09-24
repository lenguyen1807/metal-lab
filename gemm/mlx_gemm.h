#pragma once

#include <memory>
#include <vector>

class HostMatrix;

// Keep MLX types and its lazy evaluation contract inside the adapter.
class MLXGemm
{
public:
  MLXGemm(const HostMatrix& A, const HostMatrix& B);
  ~MLXGemm();

  MLXGemm(const MLXGemm&) = delete;
  MLXGemm& operator=(const MLXGemm&) = delete;

  void run();
  std::vector<float> output() const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
