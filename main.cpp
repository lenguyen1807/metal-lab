#include <exception>
#include <iostream>
#include <string>

#include "gemm/benchmark.h"

int main(int argc, char* argv[])
{
  if (argc < 2 || argc > 3 || (argc == 3 && std::string(argv[2]) != "--smoke")) {
    std::cerr << "Usage: gemm <naive|mps> [--smoke]\n";
    return 2;
  }

  try {
    BenchmarkMgr mgr;
    mgr.run_benchmark_suite(argv[1], argc == 3);
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
