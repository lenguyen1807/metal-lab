#include "gemm/kernel.h"

#include <stdexcept>
#include <string>

#include "Foundation/NSAutoreleasePool.hpp"
#include "Foundation/NSError.hpp"
#include "Metal/MTLCommandBuffer.hpp"
#include "Metal/MTLComputeCommandEncoder.hpp"
#include "gemm/matrix.h"
#include "gemm/utils.h"

bool all_shapes(const GemmShape&) {
    return true;
}

namespace naive_registry {

template <size_t BLOCK_SIZE>
DispatchPlan naive_dispatch(const GemmShape& shape, const MTL::ComputePipelineState&) {
    constexpr size_t threads_x = BLOCK_SIZE;
    constexpr size_t threads_y = BLOCK_SIZE;
    return {MTL::Size::Make((shape.N + threads_x - 1) / threads_x,
                            (shape.M + threads_y - 1) / threads_y, 1),
            MTL::Size::Make(threads_x, threads_y, 1)};
}

void encode(MTL::ComputeCommandEncoder& encoder, const DeviceMatrix& A, const DeviceMatrix& B,
            DeviceMatrix& C, const GemmShape& shape, const DispatchPlan& plan) {
    encoder.setBuffer(A.data(), 0, 0);
    encoder.setBuffer(B.data(), 0, 1);
    encoder.setBuffer(C.data(), 0, 2);
    encoder.setBytes(&shape, sizeof(shape), 3);
    encoder.dispatchThreadgroups(plan.threadgroups, plan.threads_per_threadgroup);
}
};  // namespace naive_registry

namespace tensorops_registry {

template <size_t TM, size_t TN, size_t SG = 4, bool Morton = false>
DispatchPlan dispatch(const GemmShape& shape, const MTL::ComputePipelineState& pipeline) {
    const size_t tiles_x = (shape.N + TN - 1) / TN;
    const size_t tiles_y = (shape.M + TM - 1) / TM;
    return {MTL::Size::Make(Morton ? tiles_x * tiles_y : tiles_x, Morton ? 1 : tiles_y, 1),
            MTL::Size::Make(pipeline.threadExecutionWidth() * SG, 1, 1)};
}

template <size_t TM, size_t TN, size_t BK>
bool supports_aligned(const GemmShape& shape) {
    return shape.M > 0 && shape.N > 0 && shape.K > 0 && shape.M % TM == 0 && shape.N % TN == 0 &&
           shape.K % BK == 0;
}

}  // namespace tensorops_registry

namespace simdgroup_registry {

template <size_t Tile, size_t Groups>
DispatchPlan dispatch(const GemmShape& shape, const MTL::ComputePipelineState& pipeline) {
    return {MTL::Size::Make(shape.N / Tile, shape.M / Tile, 1),
            MTL::Size::Make(pipeline.threadExecutionWidth() * Groups, 1, 1)};
}

template <size_t Tile>
bool supports(const GemmShape& shape) {
    return shape.M > 0 && shape.N > 0 && shape.K > 0 && shape.M % Tile == 0 &&
           shape.N % Tile == 0 && shape.K % 8 == 0;
}

bool supports_shared(const GemmShape& shape) {
    return supports<32>(shape) && shape.K % 32 == 0;
}

}  // namespace simdgroup_registry

namespace coarsening_registry {
template <size_t BM, size_t BN, size_t BK, size_t TM>
DispatchPlan dispatch_1D(const GemmShape& shape, const MTL::ComputePipelineState& pipeline) {
    return {MTL::Size::Make((shape.N + BN - 1) / BN, (shape.M + BM - 1) / BM, 1),
            MTL::Size::Make((BM * BN) / TM, 1, 1)};
}

template <size_t BM, size_t BN, size_t BK, size_t TM, size_t TN>
DispatchPlan dispatch_2D(const GemmShape& shape, const MTL::ComputePipelineState& pipeline) {
    return {MTL::Size::Make((shape.N + BN - 1) / BN, (shape.M + BM - 1) / BM, 1),
            MTL::Size::Make((BM * BN) / (TM * TN), 1, 1)};
}
}  // namespace coarsening_registry

const std::vector<KernelSpec>& kernel_specs() {
    static const std::vector<KernelSpec> specs = {
        {"naive", "matmul_naive", naive_registry::naive_dispatch<32>, all_shapes,
         naive_registry::encode},
        // tiling
        {"tiling", "matmul_tiling_mnk16", naive_registry::naive_dispatch<16>, all_shapes,
         naive_registry::encode},
        {"tiling", "matmul_tiling_mnk32", naive_registry::naive_dispatch<32>, all_shapes,
         naive_registry::encode},
        {"tiling", "matmul_tiling_mn32k16", naive_registry::naive_dispatch<32>, all_shapes,
         naive_registry::encode},
        // coarsening
        {"1D_coarsening", "matmul_1D_coarsening_mn64k8",
         coarsening_registry::dispatch_1D<64, 64, 8, 8>, all_shapes, naive_registry::encode},
        {"2D_coarsening", "matmul_2D_coarsening_mn128k8",
         coarsening_registry::dispatch_2D<128, 128, 8, 8, 8>, all_shapes, naive_registry::encode},
        // tensor op
        {"tensorops", "tensorops_64x64", tensorops_registry::dispatch<64, 64>, all_shapes,
         naive_registry::encode},
        {"tensorops", "tensorops_64x128", tensorops_registry::dispatch<64, 128>, all_shapes,
         naive_registry::encode},
        {"tensorops", "tensorops_128x64", tensorops_registry::dispatch<128, 64>, all_shapes,
         naive_registry::encode},
        {"tensorops", "tensorops_128x128", tensorops_registry::dispatch<128, 128>, all_shapes,
         naive_registry::encode},
        {"tensorops", "tensorops_64x64_morton", tensorops_registry::dispatch<64, 64, 4, true>,
         all_shapes, naive_registry::encode},
        {"tensorops", "tensorops_32x32_sg1", tensorops_registry::dispatch<32, 32, 1>, all_shapes,
         naive_registry::encode},
        {"tensorops", "tensorops_32x32_sg1_morton", tensorops_registry::dispatch<32, 32, 1, true>,
         all_shapes, naive_registry::encode},
        {"tensorops", "tensorops_32x64_sg1", tensorops_registry::dispatch<32, 64, 1>, all_shapes,
         naive_registry::encode},
        {"tensorops", "tensorops_64x32_sg1", tensorops_registry::dispatch<64, 32, 1>, all_shapes,
         naive_registry::encode},
        // simdgroup
        {"simdgroup", "simdgroup_8x8", simdgroup_registry::dispatch<8, 1>,
         simdgroup_registry::supports<8>, naive_registry::encode},
        {"simdgroup", "simdgroup_32x32", simdgroup_registry::dispatch<32, 4>,
         simdgroup_registry::supports<32>, naive_registry::encode},
        {"simdgroup", "simdgroup_32x32_shared", simdgroup_registry::dispatch<32, 4>,
         simdgroup_registry::supports_shared, naive_registry::encode},
        // tensor op sync
        {"tensorops_sync", "tensorops_sync_k128", tensorops_registry::dispatch<64, 64>,
         tensorops_registry::supports_aligned<64, 64, 128>, naive_registry::encode},
        {"tensorops_sync", "tensorops_sync_k256", tensorops_registry::dispatch<64, 64>,
         tensorops_registry::supports_aligned<64, 64, 256>, naive_registry::encode},
        {"tensorops_sync", "tensorops_sync_k512", tensorops_registry::dispatch<64, 64>,
         tensorops_registry::supports_aligned<64, 64, 512>, naive_registry::encode},
        {"tensorops_sync", "tensorops_sync_k1024", tensorops_registry::dispatch<64, 64>,
         tensorops_registry::supports_aligned<64, 64, 1024>, naive_registry::encode},
        {"tensorops_sync", "tensorops_sync_64x128_k512", tensorops_registry::dispatch<64, 128>,
         tensorops_registry::supports_aligned<64, 128, 512>, naive_registry::encode},
        {"tensorops_sync", "tensorops_sync_32x64_k512", tensorops_registry::dispatch<32, 64>,
         tensorops_registry::supports_aligned<32, 64, 512>, naive_registry::encode},
        {"tensorops_sync", "tensorops_sync_64x128_k256", tensorops_registry::dispatch<64, 128>,
         tensorops_registry::supports_aligned<64, 128, 256>, naive_registry::encode},
        {"tensorops_sync", "tensorops_sync_k256_morton",
         tensorops_registry::dispatch<64, 64, 4, true>,
         tensorops_registry::supports_aligned<64, 64, 256>, naive_registry::encode},
    };
    return specs;
}

Kernel::Kernel(KernelSpec spec, MTL::Device* device) : spec_(spec) {
    if (device == nullptr || spec_.dispatch == nullptr || spec_.encode == nullptr) {
        throw std::invalid_argument("Kernel requires a device and launch callbacks");
    }
    auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
    NS::Error* error = nullptr;
    const std::string source = read_file(std::string(KERNEL_PATH) + spec_.name + ".metal");
    auto* metal_source = NS::String::string(source.c_str(), NS::StringEncoding::UTF8StringEncoding);
    auto options = NS::TransferPtr(MTL::CompileOptions::alloc()->init());
    if (std::string(spec_.name).starts_with("tensorops")) {
        options->setLanguageVersion(MTL::LanguageVersion4_0);
    }
    library_ = NS::TransferPtr(device->newLibrary(metal_source, options.get(), &error));
    if (!library_) {
        const char* message =
            error ? error->localizedDescription()->utf8String() : "Metal returned no library";
        throw std::runtime_error("Cannot compile " + std::string(spec_.name) + ": " + message);
    }

    auto* function_name =
        NS::String::string(spec_.function, NS::StringEncoding::UTF8StringEncoding);
    function_ = NS::TransferPtr(library_->newFunction(function_name));
    if (!function_) {
        throw std::runtime_error("Cannot find shader function " + std::string(spec_.function));
    }
    error = nullptr;
    pipeline_ = NS::TransferPtr(device->newComputePipelineState(function_.get(), &error));
    if (!pipeline_) {
        const char* message =
            error ? error->localizedDescription()->utf8String() : "Metal returned no pipeline";
        throw std::runtime_error("Cannot create pipeline for " + std::string(spec_.name) + ": " +
                                 message);
    }
}

bool Kernel::supports(const GemmShape& shape) const {
    return spec_.supports == nullptr || spec_.supports(shape);
}

void Kernel::encode(MTL::CommandBuffer* command_buffer, const DeviceMatrix& A,
                    const DeviceMatrix& B, DeviceMatrix& C) const {
    const GemmShape shape{static_cast<uint32_t>(C.rows), static_cast<uint32_t>(C.cols),
                          static_cast<uint32_t>(A.cols)};
    if (!supports(shape) || A.cols != B.rows || A.rows != C.rows || B.cols != C.cols) {
        throw std::invalid_argument("Unsupported GEMM shape for " + std::string(spec_.name));
    }
    const DispatchPlan plan = spec_.dispatch(shape, *pipeline_.get());
    const auto& threads = plan.threads_per_threadgroup;
    if (threads.width == 0 || threads.height == 0 || threads.depth == 0 ||
        threads.width * threads.height * threads.depth >
            pipeline_->maxTotalThreadsPerThreadgroup()) {
        throw std::runtime_error("Invalid threadgroup size for " + std::string(spec_.name));
    }

    auto encoder = NS::RetainPtr(command_buffer->computeCommandEncoder());
    if (!encoder) {
        throw std::runtime_error("Cannot create compute encoder for " + std::string(spec_.name));
    }
    encoder->setComputePipelineState(pipeline_.get());
    spec_.encode(*encoder.get(), A, B, C, shape, plan);
    encoder->endEncoding();
}
