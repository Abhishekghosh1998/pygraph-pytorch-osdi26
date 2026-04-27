#include <cuda_runtime.h>
#include <cuda.h>
#include "graph_apply_updates.cuh"

// Build-time guard: device-side graph update APIs require recent CUDA.
#ifndef CUDA_VERSION
#define CUDA_VERSION 0
#endif

extern "C" __global__
void apply_param_updates_kernel(
    const cudaGraphDeviceNode_t* __restrict__ dev_nodes,
    const int* __restrict__ starts,
    const int* __restrict__ counts,
    const size_t* __restrict__ offsets,
    const unsigned long long* __restrict__ values_indices,
    unsigned long long* __restrict__ values_buf,
    cudaGraphKernelNodeUpdate* __restrict__ updates,
    int num_nodes,
    int total_updates,
    int* __restrict__ status_out)
{
#if CUDA_VERSION < 12040
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        if (status_out) *status_out = (int)cudaErrorNotSupported;
    }
    return;
#else
    if (threadIdx.x | blockIdx.x) return;

    // // Copy scalar values into a stable device buffer (so pValue points to device memory)
    // // If you already pass device pointers in values_in, this is just a memcpy.
    // for (int i = 0; i < total_updates; ++i) {
    //     values_buf[i] = values_in[i];
    // }

    // Build updates array (one entry per (node,param) patch)
    int cursor = 0;
    for (int n = 0; n < num_nodes; ++n) {
        const int start  = starts[n];
        const int count  = counts[n];
        const cudaGraphDeviceNode_t dev = dev_nodes[n];
        for (int j = 0; j < count; ++j) {
            const int k = start + j;
            updates[k].node  = dev;
            updates[k].field = cudaGraphKernelNodeFieldParam;
            updates[k].updateData.param.offset = offsets[k];
            updates[k].updateData.param.pValue = (void*)&values_buf[values_indices[k]];
            updates[k].updateData.param.size   = sizeof(void*);
        }
        cursor += count;
    }

    // Apply the batch of param updates
    cudaError_t st = cudaGraphKernelNodeUpdatesApply(updates, total_updates);
    if (status_out) *status_out = (int)st;
#endif
}

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
    int* status_out)
{
    apply_param_updates_kernel<<<1, 1, 0, stream>>>(
        dev_nodes, starts, counts, offsets,
        values_in, values_buf, updates,
        num_nodes, total_updates, status_out
    );
    // no CHECK here; we’ll check in the C++ caller with C10_CUDA_KERNEL_LAUNCH_CHECK
}