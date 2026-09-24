#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>

#include "gemm/benchmark.h"
#include "gemm/kernel.h"

namespace {

void usage()
{
  std::cerr << "Usage:\n"
            << "  gemm bench [--smoke] [--kernel GROUP] [--function NAME] [--iterations N] [--output FILE.csv]\n"
            << "  gemm list\n";
}

}  // namespace

int main(int argc, char* argv[])
{
  if (argc < 2) {
    usage();
    return 2;
  }

  const std::string command = argv[1];
  if (command == "list" && argc == 2) {
    std::cout << "mps (baseline)\n";
    std::cout << "mlx (baseline)\n";
    for (const auto& spec : kernel_specs()) {
      std::cout << spec.function << " (group: " << spec.name << ")\n";
    }
    return 0;
  }
  if (command != "bench") {
    usage();
    return 2;
  }

  try {
    BenchmarkOptions options;
    for (int i = 2; i < argc; ++i) {
      const std::string arg = argv[i];
      if (arg == "--smoke") {
        options.smoke = true;
      } else if (arg == "--kernel" && i + 1 < argc) {
        options.kernels.emplace_back(argv[++i]);
      } else if (arg == "--function" && i + 1 < argc) {
        options.functions.emplace_back(argv[++i]);
      } else if (arg == "--output" && i + 1 < argc) {
        options.output = argv[++i];
        if (options.output.empty()) {
          throw std::invalid_argument("Output must be a CSV filename without a path");
        }
      } else if (arg == "--iterations" && i + 1 < argc) {
        size_t consumed = 0;
        const std::string value = argv[++i];
        if (value.empty() || value.front() == '-') {
          throw std::invalid_argument("Iterations must be a positive integer");
        }
        options.iterations = std::stoul(value, &consumed);
        if (consumed != value.size() || options.iterations == 0) {
          throw std::invalid_argument("Iterations must be a positive integer");
        }
      } else {
        usage();
        return 2;
      }
    }
    BenchmarkMgr mgr;
    mgr.run(options);
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
