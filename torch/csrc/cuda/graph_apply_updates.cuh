#pragma once
#include <cuda_runtime.h>

// Note: requires CUDA >= 12.4 for device-side cudaGraphKernelNodeUpdatesApply.
extern "C" __global__
void apply_param_updates_kernel(
    const cudaGraphDeviceNode_t* __restrict__ dev_nodes,  // [num_nodes]
    const int* __restrict__ starts,                       // [num_nodes]
    const int* __restrict__ counts,                       // [num_nodes]
    const size_t* __restrict__ offsets,                   // [total_updates]
    const unsigned long long* __restrict__ values_in,     // [total_updates]
    unsigned long long* __restrict__ values_buf,          // [total_updates]
    cudaGraphKernelNodeUpdate* __restrict__ updates,      // [total_updates]
    int num_nodes,
    int total_updates,
    int* __restrict__ status_out                          // [1]
);

extern "C" void launch_apply_param_updates_kernel(
    cudaStream_t stream,
    const cudaGraphDeviceNode_t* dev_nodes,
    const int* starts,
    const int* counts,
    const size_t* offsets,
    const unsigned long long* values_in,
    unsigned long long* values_buf,
    cudaGraphKernelNodeUpdate* updates,
    int num_nodes,
    int total_updates,
    int* status_out);