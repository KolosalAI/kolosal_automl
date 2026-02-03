# ---- utils.py ------------------------------
from functools import singledispatch
from enum import Enum
import numpy as np
import copy
import types
import copyreg, _thread, threading

_LOCK_TYPES = (_thread.LockType, type(threading.Lock()), type(threading.RLock()))

_PRIMITIVES = (str, int, float, bool, bytes, type(None))

_IMMUTABLE_META = (
    type,                       # all classes, built‑ins included
    types.FunctionType,
    types.BuiltinFunctionType,
    types.MethodType,
    types.ModuleType,
    types.CodeType,
)
def _return_none():
    """Return None - used as replacement for unpicklable objects during serialization."""
    return None

def _patch_pickle_for_locks():
    """Register reducers that replace unpicklable objects with None when pickling."""
    def _reduce_unpicklable(_):
        return (_return_none, ())

    # Handle locks
    lock_types = {
        getattr(_thread, "LockType", None),          # CPython alias (may be missing)
        type(threading.Lock()),                      # <class '_thread.lock'>
        type(threading.RLock()),                     # <class '_thread.RLock'>
    }
    for lt in filter(None, lock_types):
        try:
            copyreg.pickle(lt, _reduce_unpicklable)
        except Exception:
            pass      # already registered / not supported
            
    # Handle queue objects
    try:
        import queue
        queue_types = {
            queue.SimpleQueue,
            queue.Queue,
            queue.PriorityQueue,
            queue.LifoQueue
        }
        for qt in queue_types:
            try:
                copyreg.pickle(qt, _reduce_unpicklable)
            except Exception:
                pass  # already registered / not supported
    except (ImportError, AttributeError):
        pass  # Queue module or specific queue type not available
def _scrub(obj, memo=None):
    """
    Return a version of *obj* with every nested Lock/RLock replaced by None,
    leaving primitives, classes, functions, modules, etc. unchanged.
    Uses *memo* to stay cycle‑safe.
    """
    if memo is None:
        memo = {}
    oid = id(obj)
    if oid in memo:
        return memo[oid]

    # ------------------------------------------------------------------ #
    # trivial cases
    if isinstance(obj, _PRIMITIVES) or obj is None:
        memo[oid] = obj
        return obj

    if isinstance(obj, _LOCK_TYPES):
        return None

    if isinstance(obj, _IMMUTABLE_META):
        memo[oid] = obj          # classes, functions, modules -> untouched
        return obj

    # ------------------------------------------------------------------ #
    # containers
    if isinstance(obj, dict):
        cleaned = {k: _scrub(v, memo) for k, v in obj.items()}
        memo[oid] = cleaned
        return cleaned

    if isinstance(obj, (list, tuple, set, frozenset)):
        cls = type(obj)
        cleaned = cls(_scrub(v, memo) for v in obj)
        memo[oid] = cleaned
        return cleaned

    # ------------------------------------------------------------------ #
    # objects with mutable attribute space
    if hasattr(obj, "__dict__") or hasattr(obj, "__slots__"):
        try:
            clone = copy.copy(obj)
        except Exception:
            memo[oid] = obj       # copying failed – leave original
            return obj

        memo[oid] = clone

        # scrub __dict__
        if hasattr(clone, "__dict__"):
            for attr, val in clone.__dict__.items():
                try:
                    setattr(clone, attr, _scrub(val, memo))
                except Exception:
                    pass

        # scrub __slots__
        for slot in getattr(clone, "__slots__", []):
            if hasattr(clone, slot):
                try:
                    setattr(clone, slot, _scrub(getattr(clone, slot), memo))
                except Exception:
                    pass
        return clone

    # fallback: leave object as‑is
    memo[oid] = obj
    return obj

@singledispatch
def _json_safe(obj):
    """Best‑effort conversion to something the std‑lib json encoder accepts."""
    try:
        return str(obj)
    except Exception:
        return f"<non‑serializable:{type(obj).__name__}>"

@_json_safe.register(type(None))
@_json_safe.register(bool)
@_json_safe.register(int)
@_json_safe.register(float)
@_json_safe.register(str)
def _(obj):
    return obj                      # already JSON‑native

@_json_safe.register(np.generic)
def _(obj):
    return obj.item()               # NumPy scalar ➝ Python scalar

@_json_safe.register(np.ndarray)
def _(obj):
    return obj.tolist()             # array ➝ list

@_json_safe.register(Enum)
def _(obj):
    return obj.value                # enum ➝ underlying value

@_json_safe.register(dict)
def _(obj: dict):
    return {k: _json_safe(v) for k, v in obj.items()}

@_json_safe.register(list)
@_json_safe.register(tuple)
@_json_safe.register(set)
def _(obj):
    t = type(obj)
    return t(_json_safe(v) for v in obj)

@_json_safe.register(type)
def _(obj):
    return f"<class '{obj.__module__}.{obj.__name__}'>"

