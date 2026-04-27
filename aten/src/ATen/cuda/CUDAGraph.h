#pragma once

#include <ATen/Tensor.h>
#include <c10/core/Device.h>
#include <c10/cuda/CUDAGraphsC10Utils.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/flat_hash_map.h>

//////////////////////////////////////////////////////////////////////////////
#include <string>
#include <vector>
//////////////////////////////////////////////////////////////////////////////

namespace at {

struct Generator;
struct CUDAGeneratorImpl;
struct CUDAGeneratorState;

namespace cuda {

// Standalone way to get a unique mempool id usable as a pool=... argument
// to CUDAGraph::capture_begin
TORCH_CUDA_CPP_API MempoolId_t graph_pool_handle();

struct KernelParamSlot {
  // Each parameter slot as returned by cuFuncGetParamInfo
  size_t offset = 0;
  size_t size = 0;
  std::vector<uint8_t> bytes; // raw bytes from kernelParams
};

struct KernelNodeInfo {
  // Basic launch config
  unsigned int gridDimX = 0, gridDimY = 0, gridDimZ = 0;
  unsigned int blockDimX = 0, blockDimY = 0, blockDimZ = 0;
  unsigned int sharedMemBytes = 0;

  // Function identity
  std::string funcName;     // cuFuncGetName result (mangled)
  uint64_t funcPtr = 0;     // reinterpret_cast<uint64_t>(CUfunction)
  uint64_t kernelParamsPtr = 0; // reinterpret_cast<uint64_t>(void**)
  uint64_t extraPtr = 0;        // reinterpret_cast<uint64_t>(void**)

  // Parameters (slots, offsets, sizes, raw bytes)
  std::vector<KernelParamSlot> params;
};

struct TORCH_CUDA_CPP_API CUDAGraph {
  CUDAGraph();
  ~CUDAGraph();

  static void inc_pending_event_queries();
  static void dec_pending_event_queries();
  static int num_pending_event_queries();
  // See Note [Explicit Registration of Generators to the CUDA Graph]
  void register_generator_state(c10::intrusive_ptr<at::CUDAGeneratorState> state);
  void register_generator_state(const at::Generator& generator);
  void capture_begin(
      MempoolId_t pool = {0, 0},
      cudaStreamCaptureMode capture_mode = cudaStreamCaptureModeGlobal);
  void capture_end();
  void replay();
  void reset();
  MempoolId_t pool();
  void enable_debug_mode();
  void debug_dump(const std::string& debug_path);

  // Returns the kernel node info cached at capture_end().
  // Empty if capture failed or if there were no kernel nodes.
  const std::vector<KernelNodeInfo>& kernel_nodes() const { return cached_kernel_nodes_; }

  // Optional: dump to a path (JSON-ish text) if you want a quick file output.
  void dump_kernel_nodes_to_file(const std::string& path) const;

  // Opt-in: if true, capture_end() will NOT instantiate (keeps graph_ alive)
  void set_defer_instantiate(bool enable);

  // Explicit instantiate to graph_exec_ after a deferred capture_end()
  void instantiate();

  // Mark given node indices device-updatable and return their cudaGraphDeviceNode_t handles (as u64)
  std::vector<uint64_t> mark_nodes_and_get_devhandles(const std::vector<int>& kernel_node_indices);

  // Sizes for Python-side buffer sizing
  static size_t sizeof_kernel_node_update();
  static size_t sizeof_device_node_handle();

 protected:
#if !defined(USE_ROCM) || ROCM_VERSION >= 50300
  cudaGraph_t graph_ = NULL;
  cudaGraphExec_t graph_exec_ = NULL;
#endif

  static std::atomic<int> pending_event_queries;

  // internal states so reset() can do its best cleaning up
  // Set to true in capture_end if cudaStreamEndCapture succeeded
  // Set back to false soon after, when graph_ is consumed by cudaGraphInstantiate
  // to create graph_exec_, then graph_ is deleted
  bool has_graph_ = false;
  // Set to true in capture_end if cudaGraphInstantiate succeeded
  bool has_graph_exec_ = false;

  // uuid of this instance's current capture, used to
  // specify the pool.
  CaptureId_t id_;

  // the ID assigned by cuda during graph capture,
  // used to identify when a stream is participating in capture
  CaptureId_t capture_id_ = -1;

  // uuid used to request a particular private mempool from CUDACachingAllocator.
  // By default, this will be set to {id_, 0}.
  //
  // If capture_begin is called with "pool=other_graph.pool()", this graph's mempool_id_
  // will be set to the other graph's mempool_id_, and therefore share a mempool with the
  // other graph.
  //
  // If capture_begin is called with "pool=handle" where "handle" came from graph_pool_handle(),
  // it will share a mempool with any other captures that used "pool=handle".
  //
  // Sharing a mempool across graphs saves memory, and it's safe if you
  // know you'll replay those graphs in the same order you captured them.
  MempoolId_t mempool_id_;

  // Stream on which capture began
  at::cuda::CUDAStream capture_stream_;

  // multiple generator states and their wholegraph_increments in this graph
  // that are managed by the CUDA Graph
  ska::flat_hash_map<c10::intrusive_ptr<at::CUDAGeneratorState>, uint64_t>
      captured_generator_states_;

  // Device where capture occurred. Right now, for simplicity, we require all ops
  // in a capture to run on the same device, but this is a limitation of CUDAGraph,
  // not CUDA itself.  We can straightforwardly modify CUDAGraph to support multi-device
  // captures if needed.
  int capture_dev_;

  // cached snapshot taken during capture_end(), before graph_ is destroyed
  std::vector<KernelNodeInfo> cached_kernel_nodes_;

  bool defer_instantiate_ = false;  // default off
  bool needs_instantiate_ = false;  // set by capture_end() if defer_instantiate_==true
};

} // namespace cuda
} // namespace at
