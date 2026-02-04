#pragma once
#include "framework.cuh"

namespace TEST {

    ///  WRITE -> No ERROR response
    ///
    /// __global__ void DummyFaultyKernel (s32 *data) {
    ///     s32 idx = threadIdx.x + blockIdx.x * blockDim.x;
    ///     data[idx + 1000] = 42;  // Intentionally out-of-bounds write.
    /// }


    ///  READ -> No ERROR response
    /// 
    /// __global__ void DummyFaultyKernel (s32 *data, s32 *sink) {
    ///     s32 idx = threadIdx.x + blockIdx.x * blockDim.x;
    ///     // Read beyond bounds and write into sink to force an error
    ///     s32 val = data[idx + 512];
    ///     sink[idx] = val;
    /// }


    /// Hardware - An ERROR response
    __global__ void DummyFaultyKernel(s32 *data) {
        s32 idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx == 0) {
            s32* invalid_ptr = nullptr;
            invalid_ptr[0] = 123;  // Will cause a clear invalid memory write
        }
    }

}


namespace TEST {

    void FaultyKernel () {
        s32* data1;

        cudaMalloc (&data1, 256 * sizeof (s32));

        DummyFaultyKernel <<<1, 256>>> (data1);
        KERNEL_GET_ERROR ("DummyFaultyKernel");

        cudaFree (data1);
    }

}
