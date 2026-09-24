#pragma once

#include <memory>

namespace MTL {
class CommandBuffer;
class Device;
}  // namespace MTL

class DeviceMatrix;

// Keep Objective-C and MPS types out of the C++ benchmark harness.
class MPSGemm {
   public:
    MPSGemm(MTL::Device* device, const DeviceMatrix& A, const DeviceMatrix& B, DeviceMatrix& C);
    ~MPSGemm();

    MPSGemm(const MPSGemm&) = delete;
    MPSGemm& operator=(const MPSGemm&) = delete;

    void encode(MTL::CommandBuffer* command_buffer) const;

   private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
