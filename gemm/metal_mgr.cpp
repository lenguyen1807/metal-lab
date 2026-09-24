#include "metal_mgr.h"

#include <stdexcept>

MetalContext::MetalContext() {
    device = NS::TransferPtr(MTL::CreateSystemDefaultDevice());
    if (!device) {
        throw std::runtime_error("Cannot create Metal device");
    }

    cmd_queue = NS::TransferPtr(device->newCommandQueue());
    if (!cmd_queue) {
        throw std::runtime_error("Cannot create Metal command queue");
    }
}
