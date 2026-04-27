import dataclasses
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import torch


@dataclasses.dataclass(frozen=True)
class FunctionID:
    "Unique counter of a function wrapped in cudagraphify_impl"
    id: int


@dataclasses.dataclass(frozen=False) # just for hacky quick implementation
class WrappedFunction:
    """
    Represents a function that you want to record for CUDA graph replay,
    with a little more metadata so we can identify if we have an applicable
    CUDA graph in our CUDA graph tree for it.
    """

    model: Callable[..., Any]
    static_input_idxs: List[int]
    id: FunctionID
    constants: Tuple[torch.Tensor, ...]
    placeholders: List[torch.fx.Node]
    mutated_input_idxs: List[int]
    
    indirect_codegen_handle: Optional[Any]
    indirect_model: Optional[Callable[..., Any]]
    is_backward: bool
    

def get_placeholders(graph: torch.fx.Graph) -> List[torch.fx.Node]:
    return [node for node in graph.nodes if node.op == "placeholder"]


def get_mutating_use_stack_trace(placeholder_node: torch.fx.Node) -> Optional[str]:
    # reinplaced uses might have a single, non-copy_ use
    if len(placeholder_node.users) == 1:
        return next(iter(placeholder_node.users)).meta.get("stack_trace", None)

    for use in placeholder_node.users:
        if use.target == torch.ops.aten.copy_.default:
            if stack_trace := use.meta.get("stack_trace", None):
                return stack_trace

    return None


def format_default_skip_message(reason: str) -> str:
    return f"skipping cudagraphs due to {reason}"


def get_mutation_stack_trace(
    placeholders: List[torch.fx.Node], mutation_indices: Iterable[int]
) -> str:
    stack_trace: Optional[str] = ""

    for idx in mutation_indices:
        placeholder = placeholders[idx]
        if stack_trace := get_mutating_use_stack_trace(placeholder):
            break

    if stack_trace:
        msg = f"skipping cudagraphs due to mutation on input. Found from : \n {stack_trace}"
        return msg

    return format_default_skip_message("mutated inputs")


def check_for_mutation(
    func: WrappedFunction,
    inputs: List[torch.Tensor],
    is_cuda_graph_recorded_tensor: Callable[[torch.Tensor], bool],
) -> Optional[str]:
    default_msg = format_default_skip_message("mutated inputs")

    # doesnt work for non-trees because the warmup run would apply mutation twice
    if torch._inductor.config.triton.cudagraph_trees:
        # checking if mutation is only on parameters/static inputs
        mutation_indices = [
            idx
            for idx in func.mutated_input_idxs
            if not (
                idx in func.static_input_idxs
                or is_cuda_graph_recorded_tensor(inputs[idx])
            )
        ]
        has_mutation = len(mutation_indices) != 0
        if not has_mutation:
            return None

        return get_mutation_stack_trace(func.placeholders, mutation_indices)

    else:
        has_mutation = len(func.mutated_input_idxs) != 0
        return None if not has_mutation else default_msg


def get_use_stack_trace(node) -> Optional[str]:
    for use in node.users:
        if stack_trace := use.meta.get("stack_trace", None):
            return stack_trace
    return None


#########################################################################Added by Me #################################################################
root_sources = []
visited_nodes = set()
disable_automatic_skipping_cudagraph_fix = False
def get_source_nodes(node):
    global root_sources
    # if the node is already visited, then return, 
    # because the reaching root nodes from this node has already been found
    # and added to root_sources
    if node in visited_nodes:
        return
    visited_nodes.add(node)
    # check if root root node or not
    # if node.args is an empty tuple, or if the first element of node.args is not a node or a "list of nodes"
    
    if node.args == ():
        if node not in root_sources:
            root_sources.append(node)
        return
    
    source_nodes = []
    # node.args is a non-empty tuple, traverse through each element of node.args
    # if the element is a node, then append it to source nodes, else if the element is a list of nodes, or tuple of nodes then append each node to source_nodes
    for arg in node.args:
        if isinstance(arg, torch.fx.Node):
            source_nodes.append(arg)
        elif isinstance(arg, (list, tuple)): # added the tuple option as well, because in some cases the nodes are not just enclosed in a list, but also in a tuple
            for ele in arg:
                if isinstance(ele, torch.fx.Node):
                    source_nodes.append(ele)

    # first_ele_of_arg = node.args[0] if isinstance(node.args[0], list) else [node.args[0]]
    # print(f"{first_ele_of_arg=}")

    if source_nodes == []:
        if node not in root_sources:
            root_sources.append(node)
        return
    
    # if not isinstance(first_ele_of_arg[0], torch.fx.Node):
    #     print(f"3.{root_sources=}")
    #     root_sources.append(node)
    #     return

    # source_nodes = first_ele_of_arg
    for node in source_nodes:
        # print(f"{type(node)=}, {node=}")
        get_source_nodes(node)

def get_corresponding_torch_ir_graph_node(node, torch_ir_gm):
    if 'source_fn_stack' in node.meta:
        inductor_ir_node_source_fn_stack = node.meta['source_fn_stack']
    else:
        print("The node does not have the source_fn_stack attribute")
        return
    
    for node in torch_ir_gm.graph.nodes:
        if 'source_fn_stack' in node.meta and node.meta['source_fn_stack'] == inductor_ir_node_source_fn_stack:
            return node
#########################################################################################################################################################

def check_multiple_devices_or_any_cpu_nodes(
    device_node_mapping: Dict[torch.device, torch.fx.Node], 
    ###################################added by me############################################
    torch_ir_gm = None,
    ##########################################################################################
) -> Optional[str]:
    # print(f"{device_node_mapping=}")
    # print("The torch_ir_gm")
    # torch_ir_gm.graph.print_tabular()

    if cpu_node := device_node_mapping.get(torch.device("cpu")):

        ###########################################################################################
        ################################# printing the CPU node####################################
        # print("#############################PRINTING THE CPU NODE#################################")
        # from prettytable import PrettyTable

        # table = PrettyTable()
        # table.field_names = ["Op", "Name", "Target", "Args", "Kwargs"]

        # # Add rows to the table
        # table.add_row([cpu_node.op, cpu_node.name, cpu_node.target, cpu_node.args, cpu_node.kwargs])
        
        # # Print the table
        # print(table)
        # ##########################################################################################
        # print(f"{cpu_node.op=}, {cpu_node.name=}, {cpu_node.target=}, {cpu_node.args=}, {cpu_node.kwargs=}")
        ############################################################################################
        msg = f"cpu device ({cpu_node.name})"
        # print each attribute of the cpu_node
        attr = dir(cpu_node)

        from .virtualized import V 
        
        global root_sources
        global visited_nodes
        global disable_automatic_skipping_cudagraph_fix
        root_sources = []
        visited_nodes = set()

        ############# dealing with the case where the user is trying to put a GPU tensor on the CPU ############
        if "device_put" in cpu_node.name:
            # check if the user is trying put a GPU tensor on the CPU
            # Then we cannot do anything about it, because the user is trying to put a GPU tensor on the CPU
            # In such case the cpu_node is of the form:
            # --> cpu_node.op='call_function', cpu_node.name='device_put', cpu_node.target=<OpOverload(op='prims.device_put', overload='default')>, cpu_node.args=(sigmoid, device(type='cpu')), cpu_node.kwargs={}
            device = cpu_node.args[1].type
            if device == "cpu":
                print("The user is trying to put a GPU tensor on the CPU. This cannot be avoided."
                      "\nHence cannot deal with skipping the cudagraphs"
                      "\nDisabling our optimization for the entire program, as this CPU device node"
                      "\nshall be propaged to other FX Graphs as well.")
                disable_automatic_skipping_cudagraph_fix = True
        
        if disable_automatic_skipping_cudagraph_fix:
            if stack_trace := get_use_stack_trace(cpu_node):
                return format_default_skip_message(f"{msg}. Found from : \n {stack_trace}")

            return format_default_skip_message(msg)
        #########################################################################################################
        get_source_nodes(cpu_node)
        # print(f"{root_sources=}")

        # We are storing the root_sources, because for placeholder nodes, we shall not have the source_fn_stack attribute
        # as a result, we shall not be able to get the corresponding node in the torch_ir_gm
        # Hence, we are storing the root_sources, so that we can work with the root_sources directly.
        V.root_sources = root_sources

        # get the corresponding node in the torch_ir_gm
        corresponding_torch_ir_gm_root_sources = []
        for node in root_sources:
            corresponding_torch_ir_gm_node = get_corresponding_torch_ir_graph_node(node, torch_ir_gm)
            if corresponding_torch_ir_gm_node is not None:
                corresponding_torch_ir_gm_root_sources.append(corresponding_torch_ir_gm_node)

        # print(f"{corresponding_torch_ir_gm_root_sources=}")

        # Doing a backtracing of the `corresponding_torch_ir_gm_root_sources` to get the get the source nodes in the torch_ir_gm
        # which does not have any dependencies. It it important for graphs of the form:
        # opcode         name    target                                                     args        kwargs                    meta
        # -------------  ------  ---------------------------------------------------------  ----------  ------------------------  ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # call_function  tensor  <built-in method tensor of type object at 0x76ec30207f20>  (0.25,)     {'dtype': torch.float32}  {'source_fn_stack': [('tensor', <built-in method tensor of type object at 0x76ec30207f20>)], 'stack_trace': '  File "/home/abhishek/pytorch-benchmarks/lib/python3.10/site-packages/torchvision/ops/poolers.py", line 125, in torch_dynamo_resume_in__setup_scales_at_122\n    lvl_min = -torch.log2(torch.tensor(scales[0], dtype=torch.float32)).item()\n', 'example_value': FakeTensor(..., size=()), 'mutation_region_id': 0}
        # call_function  log2    <built-in method log2 of type object at 0x76ec30207f20>    (tensor,)   {}        {'source_fn_stack': [('log2', <built-in method log2 of type object at 0x76ec30207f20>)], 'stack_trace': '  File "/home/abhishek/pytorch-benchmarks/lib/python3.10/site-packages/torchvision/ops/poolers.py", line 125, in torch_dynamo_resume_in__setup_scales_at_122\n    lvl_min = -torch.log2(torch.tensor(scales[0], dtype=torch.float32)).item()\n', 'example_value': FakeTensor(..., size=()), 'mutation_region_id': 0}
        # output         output  output                                                     ((log2,),)  {}                        {'creation_timestamp': 0, 'mutation_region_id': 0}
        # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # In the above example, corresponding_torch_ir_gm_root_sources will have the node log2, but based on that logic in compile_fx, we cannot add "device: cuda".
        # `log2` is dependent on the tensor `node`. That is where we should add the device as cuda. Hence, we need to backtrace to the source node, which does not have any dependencies.
        # 
        # However while applying the backtracing logic, we need to be careful: For example:
        # opcode         name                  target                                                        args                       kwargs            meta
        # -------------  --------------------  ------------------------------------------------------------  -------------------------  ----------------  -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # placeholder    l_box_lists_0_tensor  L_box_lists_0_tensor                                          ()                         {}                {'stack_trace': '  File "/home/abhishek/pytorch-benchmarks/lib/python3.10/site-packages/detectron2/modeling/poolers.py", line 95, in convert_boxes_to_pooler_format\n    boxes = torch.cat([x.tensor for x in box_lists], dim=0)\n  File "/home/abhishek/pytorch-benchmarks/lib/python3.10/site-packages/detectron2/modeling/poolers.py", line 95, in <listcomp>\n    boxes = torch.cat([x.tensor for x in box_lists], dim=0)\n', 'example_value': FakeTensor(..., device='cuda:0', size=(63, 4)), 'tensor_dict': {}, 'grapharg': GraphArg(source=AttrSource(base=GetItemSource(base=LocalSource(local_name='box_lists', cell_or_freevar=False), index=0, index_is_slice=False), member='tensor', get_static=False), _example=<torch.utils.weak.TensorWeakRef object at 0x768421cdba60>, is_unspecialized=False, fake_tensor=FakeTensor(..., device='cuda:0', size=(63, 4)), is_tensor=True, example_strong_ref=None), 'mutation_region_id': 0}
        # call_function  boxes                 <built-in method cat of type object at 0x768654da7a00>        ([l_box_lists_0_tensor],)  {'dim': 0}        {'source_fn_stack': [('cat', <built-in method cat of type object at 0x768654da7a00>)], 'stack_trace': '  File "/home/abhishek/pytorch-benchmarks/lib/python3.10/site-packages/detectron2/modeling/poolers.py", line 95, in convert_boxes_to_pooler_format\n    boxes = torch.cat([x.tensor for x in box_lists], dim=0)\n', 'example_value': FakeTensor(..., device='cuda:0', size=(63, 4)), 'mutation_region_id': 0}
        # call_method    size                  size                                                          (l_box_lists_0_tensor,)    {}                {'source_fn_stack': [('size', 'size')], 'stack_trace': '  File "/home/abhishek/pytorch-benchmarks/lib/python3.10/site-packages/detectron2/modeling/poolers.py", line 97, in convert_boxes_to_pooler_format\n    sizes = shapes_to_tensor([x.__len__() for x in box_lists])\n  File "/home/abhishek/pytorch-benchmarks/lib/python3.10/site-packages/detectron2/modeling/poolers.py", line 97, in <listcomp>\n    sizes = shapes_to_tensor([x.__len__() for x in box_lists])\n  File "/home/abhishek/pytorch-benchmarks/lib/python3.10/site-packages/detectron2/structures/boxes.py", line 240, in __len__\n    return self.tensor.shape[0]\n', 'example_value': torch.Size([63, 4]), 'mutation_region_id': 0}
        # call_function  getitem               <built-in function getitem>                                   (size, 0)                  {}                {'source_fn_stack': [('getitem', <built-in function getitem>)], 'stack_trace': '  File "/home/abhishek/pytorch-benchmarks/lib/python3.10/site-packages/detectron2/modeling/poolers.py", line 97, in convert_boxes_to_pooler_format\n    sizes = shapes_to_tensor([x.__len__() for x in box_lists])\n  File "/home/abhishek/pytorch-benchmarks/lib/python3.10/site-packages/detectron2/modeling/poolers.py", line 97, in <listcomp>\n    sizes = shapes_to_tensor([x.__len__() for x in box_lists])\n  File "/home/abhishek/pytorch-benchmarks/lib/python3.10/site-packages/detectron2/structures/boxes.py", line 240, in __len__\n    return self.tensor.shape[0]\n', 'example_value': 63, 'mutation_region_id': 0}
        # call_function  sizes                 <built-in method as_tensor of type object at 0x768654da7a00>  ([getitem],)               {'device': None}  {'source_fn_stack': [('as_tensor', <built-in method as_tensor of type object at 0x768654da7a00>)], 'stack_trace': '  File "/home/abhishek/pytorch-benchmarks/lib/python3.10/site-packages/detectron2/modeling/poolers.py", line 97, in convert_boxes_to_pooler_format\n    sizes = shapes_to_tensor([x.__len__() for x in box_lists])\n  File "/home/abhishek/pytorch-benchmarks/lib/python3.10/site-packages/detectron2/layers/wrappers.py", line 38, in shapes_to_tensor\n    return torch.as_tensor(x, device=device)\n', 'example_value': FakeTensor(..., size=(1,), dtype=torch.int64), 'mutation_region_id': 0}
        # output         output                output                                                        ((boxes, sizes),)          {}                {'creation_timestamp': 0, 'mutation_region_id': 0}
        # --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # `sizes` turn out to be the offending node, but if we do backtrace, it leads us to `l_box_lists_0_tensor` a placeholder. Where 
        # adding a device='cuda' option is invalid. So, in such cases we do not back trace.
    
        root_sources = []
        visited_nodes = set()
        # traverse through each node in the corresponding_torch_ir_gm_root_sources and get the source nodes, the result shall be automatically populated in the root_sources
        for node in corresponding_torch_ir_gm_root_sources:
            get_source_nodes(node)
        
        # check if any of the nodes in the root_sources is a placeholder node, if yes, then we shall not backtrace
        check_for_placeholder = any([node.op == "placeholder" for node in root_sources])

        if not check_for_placeholder:
            corresponding_torch_ir_gm_root_sources = root_sources
            # print("After backtracing in torch_ir_gm:")
            # print(f"{corresponding_torch_ir_gm_root_sources=}")          
        
        V.corresponding_torch_ir_gm_root_sources = corresponding_torch_ir_gm_root_sources

        # for a in attr:
        #     print(f"{a}={getattr(cpu_node, a)}")
        ################ dealing with the case where all nodes are CPU nodes ####################################
        if all(node.type == "cpu" for node in device_node_mapping):
            print("For the current FX Graph, all the nodes are CPU nodes."
                  "\n Hence, disabling our optimization for this FX Graph.")
            V.corresponding_torch_ir_gm_root_sources = None
            V.root_sources = None
        ############################################################################################
        if stack_trace := get_use_stack_trace(cpu_node):
            return format_default_skip_message(f"{msg}. Found from : \n {stack_trace}")

        return format_default_skip_message(msg)

    if (
        len(device_node_mapping) == 1
        and next(iter(device_node_mapping.keys())).type == "cuda"
    ):
        return None

    keys_repr = (repr(key) for key in device_node_mapping.keys())
    return format_default_skip_message(f"multiple devices: {', '.join(keys_repr)}")


def check_lowering_disable_cudagraph(
    device_node_mapping: Dict[torch.device, torch.fx.Node],
    ###################################added by me############################################
    torch_ir_gm = None,
    ##########################################################################################
):
    return check_multiple_devices_or_any_cpu_nodes(device_node_mapping ,
                                                   ###################################added by me############################################
                                                   torch_ir_gm = torch_ir_gm,
                                                   ##########################################################################################
                                                   )


@dataclasses.dataclass
class BoxedDeviceIndex:
    value: Optional[int]

    def set(self, device_idx: Optional[int]):
        assert device_idx is None or isinstance(device_idx, int)
        self.value = device_idx
