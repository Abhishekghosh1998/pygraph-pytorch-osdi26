#include <torch/csrc/python_headers.h>

#include <pybind11/chrono.h>

#include <torch/csrc/jit/python/pybind_utils.h>
#include <torch/csrc/utils/pybind.h>

#include <ATen/cuda/CUDAGraph.h>
#include <c10/cuda/CUDAGraphsC10Utils.h>

// Cargo culted partially from csrc/distributed/c10d/init.cpp
// and partially from csrc/cuda/Stream.cpp.
// THCPStream_init is also declared at global scope.

// Because THCPGraph_init is forward declared in the only consumer
// (csrc/Module.cpp) I don't think we need a Graph.h.

template <typename T>
using shared_ptr_class_ = py::class_<T, std::shared_ptr<T>>;

void THCPGraph_init(PyObject* module) {
  // Pybind11 patch notes say "py::module_" is more up-to-date syntax,
  // but CI linter and some builds prefer "module".
  auto torch_C_m = py::handle(module).cast<py::module>();

  torch_C_m.def("_graph_pool_handle", &::at::cuda::graph_pool_handle);

  shared_ptr_class_<::at::cuda::CUDAGraph>(torch_C_m, "_CUDAGraph")
      .def(py::init<>())
      .def(
          "capture_begin",
          [](::at::cuda::CUDAGraph& self,
             c10::optional<c10::cuda::MempoolId_t> pool_opt,
             std::string capture_error_mode) {
            cudaStreamCaptureMode capture_mode;
            c10::cuda::MempoolId_t pool = pool_opt.has_value()
                ? pool_opt.value()
                : c10::cuda::MempoolId_t{0, 0};
            if (capture_error_mode == "global") {
              capture_mode = cudaStreamCaptureModeGlobal;
            } else if (capture_error_mode == "thread_local") {
              capture_mode = cudaStreamCaptureModeThreadLocal;
            } else if (capture_error_mode == "relaxed") {
              capture_mode = cudaStreamCaptureModeRelaxed;
            } else {
              TORCH_CHECK(
                  false,
                  "Unknown capture error mode. Expected `global`, `thread_local`, or `relaxed`, got ",
                  capture_error_mode);
            }
            return self.capture_begin(pool, capture_mode);
          },
          py::arg("pool"),
          py::arg("capture_error_mode"),
          py::call_guard<py::gil_scoped_release>())
      .def(
          "capture_end",
          torch::wrap_pybind_function_no_gil(&at::cuda::CUDAGraph::capture_end))
      .def(
          "register_generator_state",
          [](::at::cuda::CUDAGraph& self, py::handle raw_generator) {
            auto generator = THPGenerator_Unwrap(raw_generator.ptr());
            // We've unwrapped Python object to C++ object,
            // so we could release GIL before calling into C++
            py::gil_scoped_release release;
            return self.register_generator_state(generator);
          },
          py::arg("generator"))
      .def(
          "replay",
          torch::wrap_pybind_function_no_gil(&at::cuda::CUDAGraph::replay))
      .def(
          "reset",
          torch::wrap_pybind_function_no_gil(&at::cuda::CUDAGraph::reset))
      .def(
          "pool",
          torch::wrap_pybind_function_no_gil(&at::cuda::CUDAGraph::pool))
      .def(
          "debug_dump",
          torch::wrap_pybind_function_no_gil(
              &::at::cuda::CUDAGraph::debug_dump))
      .def(
          "enable_debug_mode",
          torch::wrap_pybind_function_no_gil(
              &::at::cuda::CUDAGraph::enable_debug_mode))
      .def(
          "debug_dump",
          torch::wrap_pybind_function_no_gil(
              &::at::cuda::CUDAGraph::debug_dump),
          py::arg("debug_path"))
      .def(
          "kernel_nodes",
          [](::at::cuda::CUDAGraph& self) {
            const auto& nodes = self.kernel_nodes();
            py::list out;
            for (const auto& k : nodes) {
              py::dict d;
              d["func_name"] = k.funcName;
              d["grid"] = py::make_tuple(k.gridDimX, k.gridDimY, k.gridDimZ);
              d["block"] = py::make_tuple(k.blockDimX, k.blockDimY, k.blockDimZ);
              d["shared_mem_bytes"] = k.sharedMemBytes;
              d["func_ptr"] = py::int_(k.funcPtr);
              d["kernel_params_ptr"] = py::int_(k.kernelParamsPtr);
              d["extra_ptr"] = py::int_(k.extraPtr);

              py::list params;
              for (const auto& s : k.params) {
                py::dict ps;
                ps["offset"] = s.offset;
                ps["size"] = s.size;
                // Use Python bytes for raw argument content
                ps["bytes"] = py::bytes(reinterpret_cast<const char*>(s.bytes.data()),
                                        s.bytes.size());
                params.append(std::move(ps));
              }
              d["params"] = std::move(params);
              out.append(std::move(d));
            }
            return out;
          })
      .def(
          "dump_kernel_nodes",
          [](::at::cuda::CUDAGraph& self, const std::string& path) {
            self.dump_kernel_nodes_to_file(path);
          },
          py::arg("path"),
          py::call_guard<py::gil_scoped_release>())
    .def("set_defer_instantiate",
       torch::wrap_pybind_function_no_gil(&at::cuda::CUDAGraph::set_defer_instantiate),
       py::arg("enable"))
    .def("mark_nodes_and_get_devhandles",
        [](::at::cuda::CUDAGraph& self, const std::vector<int>& idxs) {
            return self.mark_nodes_and_get_devhandles(idxs);
        }, py::arg("kernel_node_indices"))
    .def("instantiate",
        torch::wrap_pybind_function_no_gil(&at::cuda::CUDAGraph::instantiate))
    .def_static("sizeof_kernel_node_update",
        &at::cuda::CUDAGraph::sizeof_kernel_node_update)
    .def_static("sizeof_device_node_handle",
        &at::cuda::CUDAGraph::sizeof_device_node_handle);
}
