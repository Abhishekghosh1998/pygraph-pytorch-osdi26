from contextlib import ExitStack
from dataclasses import dataclass
from typing import Any, Optional
from torch._inductor.virtualized import V, NullHandler
from torch._inductor.debug import DebugContext

def _is_set(x: Any) -> bool:
    return not isinstance(x, NullHandler)

@dataclass
class VState:
    debug: Optional[Any] = None
    ops: Optional[Any] = None
    real_inputs: Optional[Any] = None
    fake_mode: Optional[Any] = None
    aot_compilation: Optional[bool] = None

def capture_V_state() -> VState:
    st = VState()
    try:
        st.ops = V.get_ops_handler()
    except Exception:
        pass
    try:
        st.real_inputs = V.get_real_inputs()
        if not _is_set(st.real_inputs): st.real_inputs = None
    except Exception:
        pass
    try:
        st.fake_mode = V.get_fake_mode()
        if not _is_set(st.fake_mode): st.fake_mode = None
    except Exception:
        pass
    try:
        st.aot_compilation = V.get_aot_compilation()
    except Exception:
        pass
    try:
        dbg = V.get_debug_handler()
        st.debug = None if not _is_set(dbg) else dbg
    except Exception:
        st.debug = None
    return st

class apply_V_state:
    def __init__(self, st: VState, ensure_debug: bool = True):
        self.st = st
        self.ensure_debug = ensure_debug
        self.stack = ExitStack()
    def __enter__(self):
        if self.st.ops is not None:
            self.stack.enter_context(V.set_ops_handler(self.st.ops))
        if self.st.real_inputs is not None:
            self.stack.enter_context(V.set_real_inputs(self.st.real_inputs))
        if self.st.fake_mode is not None:
            self.stack.enter_context(V.set_fake_mode(self.st.fake_mode))
        if self.st.aot_compilation is not None:
            self.stack.enter_context(V.set_aot_compilation(self.st.aot_compilation))
        # Install a debug handler: snapshot if present, else create a fresh one
        if self.st.debug is not None:
            self.stack.enter_context(V.set_debug_handler(self.st.debug))
        elif self.ensure_debug:
            self.stack.enter_context(V.set_debug_handler(DebugContext()))
        return self
    def __exit__(self, *exc):
        return self.stack.__exit__(*exc)
