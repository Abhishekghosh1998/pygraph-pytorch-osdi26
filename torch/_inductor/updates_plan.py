# updates_plan.py
import binascii
from typing import Dict, List, Tuple, Any

def _little_endian_needle(ptr: int) -> bytes:
    return ptr.to_bytes(8, byteorder="little", signed=False)

def _find_all_offsets(hay: bytes, needle: bytes) -> List[int]:
    # linear scan; tolerate overlapping (not needed for 8B, but fine)
    out, i, n = [], 0, len(needle)
    while True:
        j = hay.find(needle, i)
        if j < 0:
            return out
        out.append(j)
        i = j + 1

def analyze_updates(
    kernel_nodes: List[Dict[str, Any]],
    idx_to_dataptr: List[tuple[int, int]],
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Returns (update_specs, warnings)
    update_specs: [{ 'node_idx': int,
                     'is_triton': bool,
                     'func_name': str,
                     'matches': [ { 'arg_blob_index': int, 'offset': int, 'value_ptr': int, 'src_idx': int } , ... ] }]
    """
    warnings: List[str] = []
    specs_by_node: Dict[int, Dict[str, Any]] = {}

    # First pass: find matches per pointer per node
    for src_idx, ptr in idx_to_dataptr:
        needle = _little_endian_needle(ptr)
        for node_idx, kn in enumerate(kernel_nodes):
            name = kn.get("func_name", "")
            is_triton = "triton" in name
            matches_here = []
            for bi, p in enumerate(kn.get("params", [])):
                blob: bytes = p["bytes"]
                base_offset: int = p["offset"]
                offs = _find_all_offsets(blob, needle)
                for off in offs:
                    matches_here.append({"arg_blob_index": bi, "offset": base_offset + off, "value_ptr": ptr, "src_idx": src_idx})
            if matches_here:
                spec = specs_by_node.setdefault(node_idx, {
                    "node_idx": node_idx,
                    "is_triton": is_triton,
                    "func_name": name,
                    "matches": [],
                })
                spec["matches"].extend(matches_here)
                # If any match shows both triton and non-triton for this ptr across nodes, warn+drop globally later
                spec.setdefault("_ptr_classes", {}).setdefault(ptr, set()).add("triton" if is_triton else "nontriton")

    # Second pass: detect pointers that appear in BOTH triton and non-triton kernels; drop them
    bad_ptrs = set()
    for spec in specs_by_node.values():
        for ptr, cls in spec.get("_ptr_classes", {}).items():
            if {"triton", "nontriton"}.issubset(cls):
                bad_ptrs.add(ptr)

    if bad_ptrs:
        for ptr in bad_ptrs:
            warnings.append(f"[skip] dataptr 0x{ptr:x} appears in both triton and non-triton kernels; skipping.")

    # Build final specs, stripping matches that reference bad_ptrs
    final_specs: List[Dict[str, Any]] = []
    for node_idx, spec in sorted(specs_by_node.items()):
        matches = [m for m in spec["matches"] if m["value_ptr"] not in bad_ptrs]
        if not matches:
            continue
        final_specs.append({
            "node_idx": node_idx,
            "is_triton": spec["is_triton"],
            "func_name": spec["func_name"],
            "matches": matches
        })

    return final_specs, warnings


def flatten_updates_for_kernel(update_specs: List[Dict[str, Any]]):
    """
    Flattens update_specs for the device kernel, using a unique 'values' buffer
    keyed by src_idx and sorted by src_idx.

    Returns dict with:
      - uniq_nodes:   [node_idx0, node_idx1, ...]          # node order as in update_specs
      - starts:       [start_k0, start_k1, ...]             # per-node start into flat arrays
      - counts:       [cnt_k0, cnt_k1, ...]                 # per-node count
      - offsets:      [byte offsets per match],             # same length = total_updates
      - value_indices:[index into unique_values per match], # same length = total_updates
      - unique_values:[values buffer ordered by sorted src_idx]  # list of uint64 values
      - src_order:    [sorted src_idx values matching unique_values order]
      - total_updates:int
      - num_nodes:    int

    Notes:
      * We keep each node's block contiguous (required by starts/counts).
      * Within each node, matches are sorted by src_idx so that value_indices are
        monotonic within-node.
      * unique_values are sorted by src_idx globally as requested.
    """

    # ---- 1) Build global unique (src_idx -> value_ptr) and sort by src_idx
    src_to_value: Dict[int, int] = {}
    for spec in update_specs:
        for m in spec.get("matches", []):
            sidx = int(m["src_idx"])
            vptr = int(m["value_ptr"])
            # If the same src_idx appears with different vptrs, last one wins.
            # (Can assert equality if you want stricter behavior.)
            src_to_value[sidx] = vptr

    src_order = sorted(src_to_value.keys())
    unique_values = [src_to_value[s] for s in src_order]
    # Map src_idx -> compact index in unique_values
    srcidx_to_compact = {s: i for i, s in enumerate(src_order)}

    # ---- 2) Flatten per node (contiguous), sorting matches within each node by src_idx
    uniq_nodes: List[int] = []
    starts: List[int] = []
    counts: List[int] = []
    offsets: List[int] = []
    value_indices: List[int] = []

    cursor = 0
    for spec in update_specs:
        node_idx = int(spec["node_idx"])
        matches = list(spec.get("matches", []))
        if not matches:
            continue

        # # Sort this node's matches by src_idx so that value_indices are in ascending order
        # matches.sort(key=lambda m: int(m["src_idx"]))

        uniq_nodes.append(node_idx)
        starts.append(cursor)
        counts.append(len(matches))

        for m in matches:
            offsets.append(int(m["offset"]))
            sidx = int(m["src_idx"])
            value_indices.append(srcidx_to_compact[sidx])

        cursor += len(matches)

    return {
        "uniq_nodes": uniq_nodes,
        "starts": starts,
        "counts": counts,
        "offsets": offsets,
        "value_indices": value_indices,   # use this to index into unique_values on device
        "unique_values": unique_values,   # sorted by src_idx
        "src_order": src_order,           # the sorted src_idx list
        "total_updates": len(offsets),
        "num_nodes": len(uniq_nodes),
    }

def _next_pow2_at_most(x, limit=1024):
    p = 1
    while p < x and p < limit:
        p <<= 1
    return p

def _build_per_update_tables(plan):
    num_nodes = int(plan["num_nodes"])
    total     = int(plan["total_updates"])
    starts    = list(map(int, plan["starts"]))
    counts    = list(map(int, plan["counts"]))
    offsets   = list(map(int, plan["offsets"]))
    validx    = list(map(int, plan["value_indices"]))

    node_of_k = [0] * total
    for n in range(num_nodes):
        s, c = starts[n], counts[n]
        for j in range(c):
            node_of_k[s + j] = n
    return node_of_k, offsets, validx

def _emit_switch_get_kconst(node_of_k, offs_of_k, validx_of_k):
    lines = []
    lines.append("struct KConst { int nidx; size_t offs; int vidx; };")
    lines.append("__device__ __forceinline__ KConst get_kconst(int idx) {")
    lines.append("    switch (idx) {")
    for i, (nidx, off, vidx) in enumerate(zip(node_of_k, offs_of_k, validx_of_k)):
        # offs as true 64-bit immediates; vidx/nidx are ints
        lines.append(f"        case {i}: return {{ {int(nidx)}, (size_t){int(off)}ULL, {int(vidx)} }};")
    lines.append("        default: return {0, (size_t)0ULL, 0}; // unreachable")
    lines.append("    }")
    lines.append("}")
    return "\n".join(lines)

def generate_apply_kernel_rtc(plan):
    """
    NVRTC-ready kernel with hard-coded per-update constants (no __constant__ arrays).
    Returns:
      {
        'name': str,
        'src':  str,
        'blocks': int,
        'threads': int,
        'shared_bytes': int,
        'shared_bytes_expr': str,
        'requires_cuda': '>= 12.4',
      }
    """
    node_of_k, offs_of_k, validx_of_k = _build_per_update_tables(plan)

    total_updates = int(plan["total_updates"])
    num_nodes     = int(plan["num_nodes"])

    # Choose kernel flavor
    warp_variant = (total_updates <= 32)
    if warp_variant:
        kernel_base = "tweak_node_batched_warp_"
        threads = 32
    else:
        kernel_base = "tweak_node_batched_"
        # one block; enough threads to cover updates with a reasonable pow2, cap at 1024
        threads = max(32, _next_pow2_at_most(total_updates, 1024))

    kernel_name = f"{kernel_base}{total_updates}"

    shared_bytes_expr = f"(size_t){total_updates} * sizeof(cudaGraphKernelNodeUpdate)"
    
    get_kconst_src = _emit_switch_get_kconst(node_of_k, offs_of_k, validx_of_k)

    header = f"""// NVRTC codegen for device-updatable param patches
// Generated for total_updates = {total_updates}, num_nodes = {num_nodes}
// Requires CUDA 12.4+ (cudaGraphKernelNodeUpdatesApply)

# include <cuda_runtime.h>

{get_kconst_src}

extern "C" __global__
void {kernel_name}(
    const cudaGraphDeviceNode_t* __restrict__ dev_nodes,
    unsigned long long* __restrict__ values_buf,
    int* __restrict__ status_out)
{{
    // Stage updates in shared memory
    extern __shared__ unsigned char s_mem[];
    auto* updates = reinterpret_cast<cudaGraphKernelNodeUpdate*>(s_mem);
"""

    if warp_variant:
        body = f"""
    const int lane = threadIdx.x & 31;
    if (lane < {total_updates}) {{
        cudaGraphKernelNodeUpdate u{{}};
        const KConst kc = get_kconst(lane);

        const auto node = dev_nodes[kc.nidx];
        u.node                   = node;
        u.field                  = cudaGraphKernelNodeFieldParam;
        u.updateData.param.offset= kc.offs;
        u.updateData.param.pValue= (void*)(&values_buf[kc.vidx]);
        u.updateData.param.size  = sizeof(void*);

        updates[lane] = u;
    }}
    __syncwarp();

    if ((threadIdx.x & 31) == 0) {{
        cudaError_t st = cudaGraphKernelNodeUpdatesApply(updates, {total_updates});
        if (status_out) *status_out = (int)st;
    }}
}}
"""
    else:
        body = f"""
    const int tid = threadIdx.x;
    const int stride = blockDim.x;

    for (int i = tid; i < {total_updates}; i += stride) {{
        cudaGraphKernelNodeUpdate u{{}};
        const KConst kc = get_kconst(i);

        const auto node = dev_nodes[kc.nidx];
        u.node                   = node;
        u.field                  = cudaGraphKernelNodeFieldParam;
        u.updateData.param.offset= kc.offs;
        u.updateData.param.pValue= (void*)(&values_buf[kc.vidx]);
        u.updateData.param.size  = sizeof(void*);

        updates[i] = u;
    }}
    __syncthreads();

    if (threadIdx.x == 0) {{
        cudaError_t st = cudaGraphKernelNodeUpdatesApply(updates, {total_updates});
        if (status_out) *status_out = (int)st;
    }}
}}
"""

    src = header + body

    return {
        "name": kernel_name,
        "src": src,
        "blocks": 1,
        "threads": threads,
        "shared_bytes_expr": shared_bytes_expr,
        "requires_cuda": ">= 12.4",
    }