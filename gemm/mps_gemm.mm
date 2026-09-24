#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#include <stdexcept>

#include "gemm/matrix.h"
#include "gemm/mps_gemm.h"

struct MPSGemm::Impl
{
  MPSMatrixMultiplication* multiplication;
  MPSMatrix* left;
  MPSMatrix* right;
  MPSMatrix* result;
};

MPSGemm::MPSGemm(MTL::Device* device,
                 const DeviceMatrix& A,
                 const DeviceMatrix& B,
                 DeviceMatrix& C)
    : impl_(std::make_unique<Impl>())
{
  if (device == nullptr || A.cols != B.rows || A.rows != C.rows
      || B.cols != C.cols) {
    throw std::invalid_argument("Invalid MPS GEMM device or matrix dimensions");
  }

  @autoreleasepool {
    id<MTLDevice> objc_device = (__bridge id<MTLDevice>)device;
    id<MTLBuffer> a_buffer = (__bridge id<MTLBuffer>)A.data();
    id<MTLBuffer> b_buffer = (__bridge id<MTLBuffer>)B.data();
    id<MTLBuffer> c_buffer = (__bridge id<MTLBuffer>)C.data();

    // Packed row-major FP32 matches DeviceMatrix and the handwritten kernels.
    MPSMatrixDescriptor* a_desc =
        [MPSMatrixDescriptor matrixDescriptorWithRows:A.rows
                                              columns:A.cols
                                             rowBytes:A.cols * sizeof(float)
                                             dataType:MPSDataTypeFloat32];
    MPSMatrixDescriptor* b_desc =
        [MPSMatrixDescriptor matrixDescriptorWithRows:B.rows
                                              columns:B.cols
                                             rowBytes:B.cols * sizeof(float)
                                             dataType:MPSDataTypeFloat32];
    MPSMatrixDescriptor* c_desc =
        [MPSMatrixDescriptor matrixDescriptorWithRows:C.rows
                                              columns:C.cols
                                             rowBytes:C.cols * sizeof(float)
                                             dataType:MPSDataTypeFloat32];

    impl_->left = [[MPSMatrix alloc] initWithBuffer:a_buffer descriptor:a_desc];
    impl_->right = [[MPSMatrix alloc] initWithBuffer:b_buffer descriptor:b_desc];
    impl_->result = [[MPSMatrix alloc] initWithBuffer:c_buffer descriptor:c_desc];
    impl_->multiplication =
        [[MPSMatrixMultiplication alloc] initWithDevice:objc_device
                                            resultRows:C.rows
                                         resultColumns:C.cols
                                        interiorColumns:A.cols];
    if (!impl_->left || !impl_->right || !impl_->result
        || !impl_->multiplication) {
      throw std::runtime_error("Cannot create MPS matrix multiplication");
    }
  }
}

MPSGemm::~MPSGemm() = default;

void MPSGemm::encode(MTL::CommandBuffer* command_buffer) const
{
  if (command_buffer == nullptr) {
    throw std::invalid_argument("MPS command buffer is null");
  }
  id<MTLCommandBuffer> objc_command_buffer =
      (__bridge id<MTLCommandBuffer>)command_buffer;
  [impl_->multiplication encodeToCommandBuffer:objc_command_buffer
                                    leftMatrix:impl_->left
                                   rightMatrix:impl_->right
                                  resultMatrix:impl_->result];
}
