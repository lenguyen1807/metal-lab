cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DMLX_DIR="$(brew --prefix mlx)/share/cmake/MLX"
cmake --build build -j
./build/bin/gemm bench --smoke --iterations 5
