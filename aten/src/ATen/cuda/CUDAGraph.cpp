#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <ATen/cuda/CUDAGraph.h>
#include <ATen/cuda/Exceptions.h>
#include <ATen/Functions.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAFunctions.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <thread>
#include <vector>

////////////////////////////////////////////////////////////////////////////////
#include <fstream>       // for std::ofstream
#include <cstring>       // for std::memcpy
#include <c10/cuda/driver_api.h> // for C10_CUDA_DRIVER_CHECK
////////////////////////////////////////////////////////////////////////////////

namespace at::cuda {

static bool _cuda_graphs_debug = false;
constexpr int kSynchronizeBusyWaitMillis = 10;

MempoolId_t graph_pool_handle() {
#if !defined(USE_ROCM) || ROCM_VERSION >= 50300
  // uuid count starts at 1. 0 is reserved to mean "wasn't set by graph_pool_handle".
  static std::atomic<CaptureId_t> uid{1};
  // Sets just the second value, to distinguish it from MempoolId_ts created from
  // cudaStreamGetCaptureInfo id_s in capture_begin.
  return {0, uid++};
#else
  TORCH_CHECK(false, "CUDA graphs may only be used in Pytorch built with CUDA >= 11.0 or ROCM >= 5.3")
  return {0, 0};
#endif
}


// Get the expected id of a capture sequence so that we can call beginAllocateStreamToPool
// before starting a graph capture
CaptureId_t capture_sequence_id() {
  // id starts at 1:
  // Ensures uuid count starts at 1. 0 is reserved to mean "not set by cudaStreamGetCaptureInfo".
  // (But how do we know GetCaptureInfo never sets id_ to 0? Because that's the current behavior,
  // and I asked cuda devs to keep it that way, and they agreed.)
  static std::atomic<CaptureId_t> uuid{1};
  return uuid++;
}

/**
 * Note [CUDA Graph Wrapper Class]
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 * Q: Why do we need graph capture and launch bindings in Pytorch?
 *    Why can't they live in a user extension, for example?
 *
 * A1: Convenience.
 * A2: To ensure valid numerics on replay, some native CUDA ops (like RNG ops with
 *     CPU statefulness) need cooperation from the capture and replay bindings
 *     (see Note [CUDA Graph-safe RNG states] in CUDAGeneratorImpl.h).
 *
 *     We can't expect users to know about this cooperation.  If users write capture
 *     bindings naively in an extension, they likely won't interact with the native
 *     ops properly.  Their graphs would yield invalid numerics on replay.
 */

/**
 * Note [Interaction with CUDA graph capture] in CUDACachingAllocator.cpp
 * describes memory management for captures.
 */

std::atomic<int> CUDAGraph::pending_event_queries = 0;

// Track any outstanding event queries that could happen e.g., in a NCCL watchdog so that they
// can be resolved before the capture begins. Note that event queries are not allowed during a
// graph capture in the default capture mode.
void CUDAGraph::inc_pending_event_queries() {
  pending_event_queries++;
}

void CUDAGraph::dec_pending_event_queries() {
  TORCH_INTERNAL_ASSERT(pending_event_queries > 0,
    "Attempted to decrement the number of outstanding events to be queried, but it was <= 0.");
  pending_event_queries--;
}

int CUDAGraph::num_pending_event_queries() {
  return pending_event_queries;
}

CUDAGraph::CUDAGraph()
  // CUDAStreams may not be default-constructed.
  : capture_stream_(at::cuda::getCurrentCUDAStream()) {
#if (defined(USE_ROCM) && ROCM_VERSION < 50300)
  TORCH_CHECK(false, "CUDA graphs may only be used in Pytorch built with CUDA >= 11.0 or ROCM >= 5.3");
#endif
}

void CUDAGraph::register_generator_state(
    c10::intrusive_ptr<at::CUDAGeneratorState> state) {
  captured_generator_states_[std::move(state)] = 0;
}

void CUDAGraph::register_generator_state(const at::Generator& generator) {
  c10::intrusive_ptr<CUDAGeneratorImpl> cuda_gen =
      dynamic_intrusive_pointer_cast<CUDAGeneratorImpl>(
          generator.getIntrusivePtr());
  cuda_gen->register_graph(this);
}

void CUDAGraph::capture_begin(MempoolId_t pool/*=0*/, cudaStreamCaptureMode capture_mode) {
#if !defined(USE_ROCM) || ROCM_VERSION >= 50300
  TORCH_CHECK(!has_graph_exec_,
              "This CUDAGraph instance already owns a captured graph. "
              "To capture a new graph, create a new instance.");

  // default generator is always registered
  auto* gen = get_generator_or_default<CUDAGeneratorImpl>(
      c10::nullopt, cuda::detail::getDefaultCUDAGenerator());
  gen->register_graph(this);

  for (auto& [generator_state, wholegraph_increments] :
       captured_generator_states_) {
    generator_state->capture_prologue();
  }

  auto stream = at::cuda::getCurrentCUDAStream();

  TORCH_CHECK(stream != at::cuda::getDefaultCUDAStream(),
              "CUDA graphs must be captured on a non-default stream. "
              "(However, after capture, it's ok to replay them on the "
              "default stream.)");

  capture_stream_ = stream;
  capture_dev_ = c10::cuda::current_device();

  id_ = capture_sequence_id();

  if (pool.first != 0 || pool.second != 0) {
    // Either value being nonzero means the user supplied a pool to share.
    // But only one should be nonzero.
    // If pool was created by another graph's capture_begin, first should be nonzero.
    // If pool was created by graph_pool_handle, second should be nonzero.
    TORCH_INTERNAL_ASSERT(!(pool.first && pool.second));
    mempool_id_ = pool;
  } else {
    // User did not ask us to share a mempool. Use our own id_ as our mempool_id_.
    // Sets just the first value, to distinguish it from MempoolId_ts created by graph_pool_handle().
    mempool_id_ = {id_, 0};
  }

  // Addendum: beginAllocateStreamToPool is now called before cudaStreamBeginCapture to prevent an
  // autograd thread's free() call triggering an invalid cudaEventRecord in the caching allocator
  // due to the capture status being updated _after_ a capture had already started.
  c10::cuda::CUDACachingAllocator::beginAllocateToPool(capture_dev_, mempool_id_, [this](cudaStream_t stream) {
      cudaStreamCaptureStatus status;
      CaptureId_t stream_capture_id;
      AT_CUDA_CHECK(cudaStreamGetCaptureInfo(stream, &status, &stream_capture_id));
      return status == cudaStreamCaptureStatus::cudaStreamCaptureStatusActive && stream_capture_id == capture_id_;
  });

  // At this point, any NCCL watchdogs should be aware that we are in capture mode
  // and therefore should not enqueue any additional work that could be event-queried.
  // We still must wait on any existing work that has not been cleaned up.
  while (num_pending_event_queries()) {
    TORCH_WARN_ONCE("Waiting for pending NCCL work to finish before starting graph capture.");
    std::this_thread::sleep_for(
      std::chrono::milliseconds(kSynchronizeBusyWaitMillis));
  }

  // cudaStreamCaptureModeGlobal is the most conservative option to
  // prevent potentially unsafe CUDA API calls during capture.  See
  // https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1g9d0535d93a214cbf126835257b16ba85
  AT_CUDA_CHECK(cudaStreamBeginCapture(capture_stream_, capture_mode));

  cudaStreamCaptureStatus status;
  AT_CUDA_CHECK(cudaStreamGetCaptureInfo(stream, &status, &capture_id_));
  TORCH_INTERNAL_ASSERT(status == cudaStreamCaptureStatus::cudaStreamCaptureStatusActive);

  TORCH_INTERNAL_ASSERT(id_ > 0);
#else
  TORCH_CHECK(false, "CUDA graphs may only be used in Pytorch built with CUDA >= 11.0 or ROCM >= 5.3")
#endif
}

// helper to read per-parameter bytes given kernelParams (void**)
static inline std::vector<uint8_t> read_param_bytes(void** kernelParams, size_t byteCount) {
  std::vector<uint8_t> out(byteCount);
  // kernelParams[i] points to host memory containing the argument bytes
  // We'll memcpy directly from that pointer into our vector.
  // (Callers pass the right sub-pointer per-arg.)
  std::memcpy(out.data(), *kernelParams, byteCount);
  return out;
}

// Decode CUDA 'extra' array to (buffer_ptr, buffer_size).
// NOTE: For BUFFER_POINTER, 'val' IS the pointer; for BUFFER_SIZE, 'val' points to size_t.
static bool decode_extra_buffer(void** extra, const void** out_buf, size_t* out_size) {
  if (!extra) return false;
  void** p = extra;
  const void* buf_ptr = nullptr;
  size_t buf_size = 0;

  while (*p && *p != CU_LAUNCH_PARAM_END) {
    void* key = *p++;
    if (!*p) break; // malformed
    void* val = *p++;

    if (key == CU_LAUNCH_PARAM_BUFFER_POINTER) {
      buf_ptr = val; // value IS the pointer
    } else if (key == CU_LAUNCH_PARAM_BUFFER_SIZE) {
      buf_size = *reinterpret_cast<size_t*>(val); // value points to size_t
    }
  }

  if (buf_ptr && buf_size > 0) {
    *out_buf = buf_ptr;
    *out_size = buf_size;
    return true;
  }
  return false;
}

// Fill KernelNodeInfo.params from a packed parameter buffer using cuFuncGetParamInfo.
static void fill_params_from_packed_buffer(
    const CUfunction& func,
    const unsigned char* buf,
    size_t buf_size,
    std::vector<KernelParamSlot>& out_params) {

  size_t idx = 0;
  for (;; ++idx) {
    size_t off = 0, sz = 0;
    CUresult r = CUDA_ERROR_NOT_SUPPORTED;

    // Use the driver shim if available (works with PyTorch's dlopen scheme).
    if (auto fn = c10::cuda::DriverAPI::get()->cuFuncGetParamInfo_) {
      r = fn(func, idx, &off, &sz);
    }
    if (r != CUDA_SUCCESS) break;

    KernelParamSlot slot;
    slot.offset = off;
    slot.size = sz;

    // guard against bad sizes
    if (off + sz <= buf_size) {
      slot.bytes.resize(sz);
      std::memcpy(slot.bytes.data(), buf + off, sz);
    } else {
      // Truncate gracefully if reported size exceeds buffer (defensive)
      size_t avail = (off < buf_size) ? (buf_size - off) : 0;
      slot.bytes.resize(avail);
      if (avail) std::memcpy(slot.bytes.data(), buf + off, avail);
    }
    out_params.emplace_back(std::move(slot));
  }

  // If the driver cannot enumerate params (older driver/toolkit), still expose the whole blob.
  if (idx == 0) {
    KernelParamSlot slot;
    slot.offset = 0;
    slot.size = buf_size;
    slot.bytes.assign(buf, buf + buf_size);
    out_params.emplace_back(std::move(slot));
  }
}

void CUDAGraph::capture_end() {
#if !defined(USE_ROCM) || ROCM_VERSION >= 50300
  auto stream = at::cuda::getCurrentCUDAStream();

  TORCH_CHECK(stream == capture_stream_,
              "Capture must end on the same stream it began on.");

  AT_CUDA_CHECK(cudaStreamEndCapture(capture_stream_, &graph_));

  c10::cuda::CUDACachingAllocator::endAllocateToPool(capture_dev_, mempool_id_);

  TORCH_CHECK(graph_ != NULL, "Invalid capture.");
  has_graph_ = true;


  size_t numCUDAGraphNodes = 0;
  AT_CUDA_CHECK(cudaGraphGetNodes(graph_, NULL, &numCUDAGraphNodes));
  if (numCUDAGraphNodes == 0) {
      TORCH_WARN("The CUDA Graph is empty. This usually means that the graph was ",
                 "attempted to be captured on wrong device or stream.");
  }

  // --------- NEW: snapshot kernel nodes before possibly destroying graph_ ----------
  cached_kernel_nodes_.clear();
  if (numCUDAGraphNodes > 0) {
    std::vector<cudaGraphNode_t> nodes(numCUDAGraphNodes);
    AT_CUDA_CHECK(cudaGraphGetNodes(graph_, nodes.data(), &numCUDAGraphNodes));
    for (auto n : nodes) {
      cudaGraphNodeType t;
      AT_CUDA_CHECK(cudaGraphNodeGetType(n, &t));
      if (t == cudaGraphNodeTypeKernel) {
        CUDA_KERNEL_NODE_PARAMS p{};
        C10_CUDA_DRIVER_CHECK(c10::cuda::DriverAPI::get()->cuGraphKernelNodeGetParams_(n, &p));

        KernelNodeInfo info;
        info.gridDimX = p.gridDimX; info.gridDimY = p.gridDimY; info.gridDimZ = p.gridDimZ;
        info.blockDimX = p.blockDimX; info.blockDimY = p.blockDimY; info.blockDimZ = p.blockDimZ;
        info.sharedMemBytes = p.sharedMemBytes;
        info.funcPtr = reinterpret_cast<uint64_t>(p.func);
        info.kernelParamsPtr = reinterpret_cast<uint64_t>(p.kernelParams);
        info.extraPtr = reinterpret_cast<uint64_t>(p.extra);

        // func name
        const char* fname = nullptr;
        C10_CUDA_DRIVER_CHECK(c10::cuda::DriverAPI::get()->cuFuncGetName_(&fname, p.func));
        if (fname) info.funcName = fname;

        // parameter slots via cuFuncGetParamInfo (CUDA 12.2+), guarded by availability
#if defined(CUDA_VERSION) && CUDA_VERSION >= 12020
        if (p.kernelParams) {
          for (size_t idx = 0;; ++idx) {
            size_t off = 0, sz = 0;
            CUresult r = CUDA_ERROR_NOT_SUPPORTED;
            if (auto fn = c10::cuda::DriverAPI::get()->cuFuncGetParamInfo_) {
              r = fn(p.func, idx, &off, &sz);
            }
            if (r != CUDA_SUCCESS) break;

            KernelParamSlot slot;
            slot.offset = off;
            slot.size = sz;

            void** base = reinterpret_cast<void**>(p.kernelParams);
            void* arg_ptr = base[idx];
            slot.bytes.resize(sz);
            std::memcpy(slot.bytes.data(), arg_ptr, sz);

            info.params.emplace_back(std::move(slot));
          }
        }
        if (!p.kernelParams && p.extra) {
          const void* packed_ptr = nullptr;
          size_t packed_size = 0;
          if (decode_extra_buffer(p.extra, &packed_ptr, &packed_size)) {
            const auto* u8 = static_cast<const unsigned char*>(packed_ptr);
            fill_params_from_packed_buffer(p.func, u8, packed_size, info.params);
          }
        }
#else
        // Older CUDA toolkits don't have cuFuncGetParamInfo. We still return launch config & func identity.
#endif

        cached_kernel_nodes_.emplace_back(std::move(info));
      }
    }
  }
  // --------- end NEW ---------

  // --------- NEW: defer branch (opt-in), KEEP graph_, SKIP instantiate/epilogue/destroy ----------
  if (defer_instantiate_) {
    needs_instantiate_ = true;
    return;  // graph_ remains alive; user will call instantiate() later
  }
  // --------- end NEW ----------

  // In typical graph usage some tensors (e.g. the tensors used for graph IO) are not freed
  // between replays.
  // If Pytorch compiles and runs with a CUDA 11.4+ toolkit, there's a chance the allocator backend
  // is cudaMallocAsync.
  // cudaMallocAsync is generally graph-safe, but if some tensors are not freed between replays,
  // the graph's internal bookkeeping requires that we instantiate with
  // cudaGraphInstantiateFlagAutoFreeOnLaunch. See
  // cudaGraphLaunch
  // https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__GRAPH.html#group__CUDART__GRAPH_1g1accfe1da0c605a577c22d9751a09597
  // cudaGraphInstantiateWithFlags
  // https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__GRAPH.html#group__CUDART__GRAPH_1ga2c652a24ba93e52b99a47bec0888233
#if (defined(CUDA_VERSION) && CUDA_VERSION >= 11040)
  int version;
  AT_CUDA_CHECK(cudaDriverGetVersion(&version));
  if (version < 11040) {
#endif
    // Trailing NULL, NULL, 0 arguments were recommended by Cuda driver people,
    // who prefer not to report error message through these arguments moving forward
    // (they prefer return value, or errors on api calls internal to the capture)
#if (defined(CUDA_VERSION) && CUDA_VERSION >= 12000)
    AT_CUDA_CHECK(cudaGraphInstantiate(&graph_exec_, graph_, 0));
#else
    AT_CUDA_CHECK(cudaGraphInstantiate(&graph_exec_, graph_, NULL, NULL, 0));
#endif
#if (defined(CUDA_VERSION) && CUDA_VERSION >= 11040)
  } else {
    AT_CUDA_CHECK(cudaGraphInstantiateWithFlags(&graph_exec_,
                                                graph_,
                                                cudaGraphInstantiateFlagAutoFreeOnLaunch));
  }
#endif

  has_graph_exec_ = true;

  for (auto& [generator_state, wholegraph_increments] :
       captured_generator_states_) {
    wholegraph_increments = generator_state->capture_epilogue();
  }

  // check if debug path is set
  if (!_cuda_graphs_debug) {
    // Now that we've instantiated graph_ into graph_exec_,
    // we don't need graph_ anymore.
    AT_CUDA_CHECK(cudaGraphDestroy(graph_));
    has_graph_ = false;
  } else {
    TORCH_WARN("DEBUG: TORCH_CUDAGRAPHS_DEBUG_PATH detected. graph_ will not be freed until debug_dump is called.");
  }
#else
  TORCH_CHECK(false, "CUDA graphs may only be used in Pytorch built with CUDA >= 11.0 or ROCM >= 5.3")
#endif
}

void CUDAGraph::instantiate() {
#if !defined(USE_ROCM) || ROCM_VERSION >= 50300
  TORCH_CHECK(needs_instantiate_, "instantiate() called without a deferred capture_end().");
  TORCH_CHECK(has_graph_ && !has_graph_exec_, "Bad state for instantiate().");

#if (defined(CUDA_VERSION) && CUDA_VERSION >= 11040)
  int version;
  AT_CUDA_CHECK(cudaDriverGetVersion(&version));
  if (version < 11040) {
#endif
#if (defined(CUDA_VERSION) && CUDA_VERSION >= 12000)
    AT_CUDA_CHECK(cudaGraphInstantiate(&graph_exec_, graph_, 0));
#else
    AT_CUDA_CHECK(cudaGraphInstantiate(&graph_exec_, graph_, NULL, NULL, 0));
#endif
#if (defined(CUDA_VERSION) && CUDA_VERSION >= 11040)
  } else {
    AT_CUDA_CHECK(cudaGraphInstantiateWithFlags(&graph_exec_,
                                                graph_,
                                                cudaGraphInstantiateFlagAutoFreeOnLaunch));
  }
#endif

  has_graph_exec_ = true;
  needs_instantiate_ = false;

  // Run RNG epilogue here (was in capture_end() in default path)
  for (auto& [generator_state, wholegraph_increments] : captured_generator_states_) {
    wholegraph_increments = generator_state->capture_epilogue();
  }

  if (!_cuda_graphs_debug) {
    AT_CUDA_CHECK(cudaGraphDestroy(graph_));
    has_graph_ = false;
  } else {
    TORCH_WARN("DEBUG: TORCH_CUDAGRAPHS_DEBUG_PATH detected. graph_ will not be freed until debug_dump is called.");
  }
#else
  TORCH_CHECK(false, "CUDA graphs may only be used in Pytorch built with CUDA >= 11.0 or ROCM >= 5.3")
#endif
}

void CUDAGraph::dump_kernel_nodes_to_file(const std::string& path) const {
#if !defined(USE_ROCM) || ROCM_VERSION >= 50300
  std::ofstream os(path);
  if (!os) {
    TORCH_WARN("Could not open ", path, " for writing.");
    return;
  }
  os << "{ \"kernel_nodes\": [\n";
  for (size_t i = 0; i < cached_kernel_nodes_.size(); ++i) {
    const auto& k = cached_kernel_nodes_[i];
    os << "  {\"funcName\":\"" << k.funcName << "\","
       << "\"grid\":[" << k.gridDimX << "," << k.gridDimY << "," << k.gridDimZ << "],"
       << "\"block\":[" << k.blockDimX << "," << k.blockDimY << "," << k.blockDimZ << "],"
       << "\"shared\":" << k.sharedMemBytes << ","
       << "\"funcPtr\":" << k.funcPtr << ","
       << "\"kernelParamsPtr\":" << k.kernelParamsPtr << ","
       << "\"extraPtr\":" << k.extraPtr << ","
       << "\"params\":[";
    for (size_t j = 0; j < k.params.size(); ++j) {
      const auto& s = k.params[j];
      os << "{\"offset\":" << s.offset << ",\"size\":" << s.size << ",\"bytes\":\"";
      static const char* hex = "0123456789abcdef";
      for (uint8_t b : s.bytes) { os << hex[b >> 4] << hex[b & 0xF]; }
      os << "\"}";
      if (j + 1 < k.params.size()) os << ",";
    }
    os << "]}";
    if (i + 1 < cached_kernel_nodes_.size()) os << ",";
    os << "\n";
  }
  os << "]}\n";
#else
  TORCH_WARN("dump_kernel_nodes_to_file not supported on ROCm < 5.3");
#endif
}


void CUDAGraph::replay() {
#if !defined(USE_ROCM) || ROCM_VERSION >= 50300
  TORCH_CHECK(has_graph_exec_,
              "Called CUDAGraph::replay without a preceding successful capture.");

  c10::OptionalDeviceGuard device_guard{capture_stream_.device()};

  for (auto& [generator_state, wholegraph_increments] :
       captured_generator_states_) {
    generator_state->replay_prologue(wholegraph_increments);
  }
  // graph_exec_ may be replayed in any stream.
  AT_CUDA_CHECK(cudaGraphLaunch(graph_exec_, at::cuda::getCurrentCUDAStream()));

  int version;
  AT_CUDA_CHECK(cudaDriverGetVersion(&version));
  if (version < 11040) {
    // Workaround for bug in libcuda.so that causes replayed graphs with
    // certain topologies to be corrupted (kernels elided, internal syncs
    // ignored) when replayed back to back without a sync in between.
    // The bug is fixed in CUDA 11.4+.
    AT_CUDA_CHECK(cudaDeviceSynchronize());
  }
#else
  TORCH_CHECK(false, "CUDA graphs is not yet supported on ROCM");
#endif
}

void CUDAGraph::set_defer_instantiate(bool enable) {
  TORCH_CHECK(!has_graph_ && !has_graph_exec_,
              "set_defer_instantiate() must be called before capture begins.");
  defer_instantiate_ = enable;
}

void CUDAGraph::enable_debug_mode() {
#if !defined(USE_ROCM) || ROCM_VERSION >= 50300
  _cuda_graphs_debug = true;
#else
  TORCH_CHECK(false, "CUDA graphs is not yet supported on ROCM");
#endif

}

void CUDAGraph::debug_dump(const std::string& debug_path) {
#if (defined(CUDA_VERSION) && CUDA_VERSION >= 11030)|| (defined(USE_ROCM) && ROCM_VERSION >= 50600)
  if (_cuda_graphs_debug) {
    TORCH_WARN("DEBUG: calling debug_dump()");
    if (has_graph_) {
      TORCH_WARN("DEBUG: calling cudaGraphDebugDotPrint() with ", debug_path);
      C10_CUDA_CHECK_WARN(cudaGraphDebugDotPrint(graph_, debug_path.c_str(), 1<<10)); // most verbose output
      AT_CUDA_CHECK(cudaGraphDestroy(graph_));
    }
  } else {
    TORCH_WARN("CUDA Graphs debug not enabled, set with torch._C._cuda_enable_graphs_debug_mode");
  }
#else
  TORCH_CHECK(false, "CUDA graphs may only be used in Pytorch built with CUDA >= 11.3 or ROCM >= 5.6");
#endif
}

void CUDAGraph::reset() {
#if !defined(USE_ROCM) || ROCM_VERSION >= 50300
  // I'd prefer these checks throw exceptions, not print warnings,
  // but the destructor calls reset(), and at least one CI build
  // refuses to compile with a throwing destructor.
  //
  // Instead of calling reset() in the destructor to clean up, I could
  // call reset() in the __del__ method of a thin Python wrapper,
  // in which case reset would be allowed to throw exceptions.
  // But Stackoverflow does not like user-defined __del__.
  // __del__ prevents Graph instances from EVER being garbage collected
  // if they participate in a reference cycle.
  // And exceptions thrown in __del__ only print a warning anyway.
  //
  // Calling reset() in the C++ destructor, with warnings instead of exceptions
  // if calls fail, is the compromise we chose.
  //
  // If capture_begin, the capture, or capture_end failed at some point, this CUDAGraph, the generator,
  // and the allocator could end up in all kinds of weird states depending where failure occurred.
  // If the user catches the failure exception in a script, or is running in REPL or (god forbid)
  // a Jupyter notebook, I don't see an easy way for reset() to gracefully fix all such possible error states.
  if (has_graph_ || has_graph_exec_) {
    // notifyCaptureDestroy may throw. How should we handle this?
    c10::cuda::CUDACachingAllocator::releasePool(capture_dev_, mempool_id_);
  }
  if (has_graph_) {
    C10_CUDA_CHECK_WARN(cudaGraphDestroy(graph_));
    has_graph_ = false;
  }
  if (has_graph_exec_) {
    C10_CUDA_CHECK_WARN(cudaGraphExecDestroy(graph_exec_));
    has_graph_exec_ = false;
  }
#else
  TORCH_CHECK(false, "CUDA graphs may only be used in Pytorch built with CUDA >= 11.0 or ROCM >= 5.3")
#endif
}

// Returns an id another graph's capture_begin can use to share the same memory pool as this graph.
MempoolId_t CUDAGraph::pool() {
#if !defined(USE_ROCM) || ROCM_VERSION >= 50300
TORCH_CHECK(has_graph_exec_,
              "Called CUDAGraph::pool() without a preceding successful capture.");
#else
  TORCH_CHECK(false, "CUDA graphs may only be used in Pytorch built with CUDA >= 11.0 or ROCM >= 5.3")
#endif
  return mempool_id_;
}

std::vector<uint64_t>
CUDAGraph::mark_nodes_and_get_devhandles(const std::vector<int>& kernel_node_indices) {
#if !(defined(CUDA_VERSION) && CUDA_VERSION >= 12040)
  TORCH_CHECK(false,
      "Device-updatable kernel nodes require CUDA 12.4+ (CUDA_VERSION >= 12040).");
#else
  TORCH_CHECK(has_graph_ && !has_graph_exec_,
              "mark_nodes_and_get_devhandles() requires a captured (uninstantiated) graph.");

  // Enumerate all nodes in the captured graph
  size_t n = 0;
  AT_CUDA_CHECK(cudaGraphGetNodes(graph_, /*nodes=*/nullptr, &n));
  std::vector<cudaGraphNode_t> all_nodes(n);
  if (n) {
    AT_CUDA_CHECK(cudaGraphGetNodes(graph_, all_nodes.data(), &n));
  }

  // Build a compact vector of ONLY kernel nodes, in the same order we expose to Python
  std::vector<cudaGraphNode_t> kernel_nodes;
  kernel_nodes.reserve(all_nodes.size());
  for (auto node : all_nodes) {
    cudaGraphNodeType t;
    AT_CUDA_CHECK(cudaGraphNodeGetType(node, &t));
    if (t == cudaGraphNodeTypeKernel) {
      kernel_nodes.push_back(node);
    }
  }

  // For each requested kernel-node index, mark as device-updatable and fetch dev handle
  std::vector<uint64_t> out;
  out.reserve(kernel_node_indices.size());

  for (int kidx : kernel_node_indices) {
    TORCH_CHECK(0 <= kidx && kidx < static_cast<int>(kernel_nodes.size()),
                "kernel node index ", kidx, " out of bounds (0..",
                static_cast<int>(kernel_nodes.size()) - 1, ")");

    auto node = kernel_nodes[kidx];

    // Set attribute: mark as device-updatable.
    cudaKernelNodeAttrValue attr{};
    attr.deviceUpdatableKernelNode.deviceUpdatable = 1;
    attr.deviceUpdatableKernelNode.devNode = 0;
    AT_CUDA_CHECK(cudaGraphKernelNodeSetAttribute(
        node, cudaKernelNodeAttributeDeviceUpdatableKernelNode, &attr));

    
    uint64_t h = 0;
    static_assert(sizeof(cudaGraphDeviceNode_t) <= sizeof(uint64_t),
                  "cudaGraphDeviceNode_t larger than uint64_t storage.");
    std::memcpy(&h, &attr.deviceUpdatableKernelNode.devNode,
                sizeof(attr.deviceUpdatableKernelNode.devNode));
    out.push_back(h);
  }

  return out;
#endif
}

size_t CUDAGraph::sizeof_kernel_node_update() { return sizeof(cudaGraphKernelNodeUpdate); }
size_t CUDAGraph::sizeof_device_node_handle() { return sizeof(cudaGraphDeviceNode_t); }

CUDAGraph::~CUDAGraph() {
  for (auto& [generator_state, wholegraph_increments] :
       captured_generator_states_) {
    generator_state->unregister_graph(this);
  }
  reset();
}

} // namespace at::cuda
