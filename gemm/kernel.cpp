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

template<size_t BLOCK_SIZE>
DispatchPlan naive_dispatch(const GemmShape& shape, const MTL::ComputePipelineState&) {
    constexpr size_t threads_x = BLOCK_SIZE;
    constexpr size_t threads_y = BLOCK_SIZE;
    return {MTL::Size::Make((shape.N + threads_x - 1) / threads_x,
                            (shape.M + threads_y - 1) / threads_y, 1),
            MTL::Size::Make(threads_x, threads_y, 1)};
}

void encode_naive(MTL::ComputeCommandEncoder& encoder, const DeviceMatrix& A, const DeviceMatrix& B,
                  DeviceMatrix& C, const GemmShape& shape, const DispatchPlan& plan) {
    const auto& threads = plan.threads_per_threadgroup;
    MatmulParams params{shape.M,
                        shape.N,
                        shape.K,
                        1.f,
                        0.f,
                        static_cast<uint32_t>(threads.width),
                        static_cast<uint32_t>(threads.height)};
    encoder.setBuffer(A.data(), 0, 0);
    encoder.setBuffer(B.data(), 0, 1);
    encoder.setBuffer(C.data(), 0, 2);
    encoder.setBytes(&params, sizeof(params), 3);
    encoder.dispatchThreadgroups(plan.threadgroups, plan.threads_per_threadgroup);
}
};

const std::vector<KernelSpec>& kernel_specs() {
    static const std::vector<KernelSpec> specs = {
        {"naive", "matmul_naive", naive_registry::naive_dispatch<32>, all_shapes, naive_registry::encode_naive},
        {"tiling", "matmul_tiling_mnk16", naive_registry::naive_dispatch<16>, all_shapes, naive_registry::encode_naive},
        {"tiling", "matmul_tiling_mnk32", naive_registry::naive_dispatch<32>, all_shapes, naive_registry::encode_naive},
        {"tiling", "matmul_tiling_mn32k16", naive_registry::naive_dispatch<32>, all_shapes, naive_registry::encode_naive}
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
    library_ = NS::TransferPtr(device->newLibrary(metal_source, nullptr, &error));
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
