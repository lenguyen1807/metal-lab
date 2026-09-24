#pragma once

#include <type_traits>

#include "Foundation/NSSharedPtr.hpp"
#include "Metal/MTLCommandQueue.hpp"
#include "Metal/MTLDevice.hpp"

struct MetalContext {
    NS::SharedPtr<MTL::Device> device;
    NS::SharedPtr<MTL::CommandQueue> cmd_queue;

    MetalContext();
};
