import logging
import typing
import warnings
from functools import wraps
from pathlib import Path
from typing import Any, Optional

import numpy as np
from numpy import ndarray

from src.lib.loglib import trace_decorator
from src.party.ravelang import pnodes_utils
from src.party.ravelang.PNode import PNode

if typing.TYPE_CHECKING:
    from src.party.models.rembg_ import Rembg

def deprecated(instead: str):
    """Decorator to mark functions as deprecated with a suggested alternative"""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            warnings.warn(
                f"\n{func.__name__} is deprecated and will be removed in a future version. "
                f"Use {instead} instead.\n"
                f"Called from: {func.__name__}(args={args}, kwargs={kwargs})",
                DeprecationWarning,
                stacklevel=2,
            )
            return func(*args, **kwargs)

        return wrapper

    return decorator


log = logging.getLogger("rv")

# These are values that cannot be used as signal names
# Because they are either special or cannot change during rendering
rv_protected = [
    "session",
    "signals",
    "f",
    "f_exists",
    "img",
    "img_f",
    "img_f1",
    "w",
    "h",
    "w2",
    "h2",
    "t",
    "dt",
    "ref",
    "tr",
    "len",
    "n",
    "draft",
    "is_array_mode",
    "is_custom_img",
    "has_init_frame",
    "rembg_model",
]

RV_SIGNAL_INIT_STRETCH = "init_stretch"
RV_SIGNAL_INIT_ANCHOR_Y = "init_anchor_y"
RV_SIGNAL_INIT_ANCHOR_X = "init_anchor_x"
RV_SIGNAL_INIT_PIVOT_X = "init_pivot_x"
RV_SIGNAL_INIT_PIVOT_Y = "init_pivot_y"
RV_SIGNAL_INIT_ZOOM = "init_zoom"

UNDEFINED = object()


class Signal:
    def __init__(self, array: np.ndarray, name: str):
        if isinstance(array, memoryview):
            log.error("Signal: array is a memoryview, converting to ndarray")
            array = np.array(array)

        self.array = array
        self.name = name

    def __getitem__(self, index):
        return self.array[index]

    def __setitem__(self, index, value):
        self.array[index] = value

    def __len__(self):
        return len(self.array)


class SignalBlock:  # type: ignore
    """A dictionary of signals that can be manipulated as a unit, with efficient slicing operations.
    Maintains full dict interface while adding signal-specific operations.
    """

    def __init__(self, name: str, n: int = 0):
        super().__init__()
        self.__protected__ = {"n", "signals", "is_array_mode", "name"}
        self.signals: dict[str, Signal] = {}
        self.is_array_mode = True
        self.name = name
        self.n = n
        self.path = None

    def save_cache(self, path):
        """
        Save the signal block to disk as a numpy .npz file.
        The cache includes all signals and their arrays.
        """
        if not path:
            return

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Create a dictionary of arrays to save
        arrays = {}
        for name, signal in self.signals.items():
            arrays[name] = signal.array

        # Save metadata as a structured numpy array
        metadata_dtype = np.dtype([
            ('name', 'U256'),  # Unicode string up to 256 chars
            ('n', np.int64),
            ('is_array_mode', np.bool_)
        ])
        metadata = np.array(
            [(self.name, self.n, self.is_array_mode)],
            dtype=metadata_dtype
        )
        arrays['_metadata'] = metadata

        # Save to .npz file
        np.savez_compressed(path, **arrays)
        log.info(f"Saved signals to cache ({path})")

    def load_cache(self, path):
        """
        Load a signal block from disk.
        Returns True if successful, False if cache doesn't exist or is invalid.
        """
        if not path:
            return False

        path = Path(path)
        if not path.exists():
            return False

        try:
            # Load arrays from .npz file
            data = np.load(path, allow_pickle=False)

            # Load metadata from structured array
            metadata = data['_metadata']
            self.name = str(metadata['name'][0])  # Convert from numpy string to Python string
            self.n = int(metadata['n'][0])
            self.is_array_mode = bool(metadata['is_array_mode'][0])

            # Clear existing signals
            self.signals.clear()

            # Load signals
            for name in data.files:
                if name != '_metadata':
                    self.signals[name] = Signal(data[name], name)

            log.info(f"Loaded signals from cache ({path})")
            return True
        except Exception as e:
            log.error(f"Failed to load cache from {path}: {e}")
            return False

    def protect(self):
        self.__protected__ = set(self.__dict__.keys())

    def clone(self, name=None):
        return self.Clone(self, name)

    @classmethod
    def Clone(cls, other, name=None):
        """Create a new SignalBlock with the same signals"""
        if name is None:
            name = other.name
        ret = SignalBlock(name, other.n)
        ret.load(other)
        return ret

    def slice(self, src: int, dst: int) -> "SignalBlock":
        """Creates a new SignalGroup containing sliced signals from src (inclusive) to dst (exclusive)"""
        sliced = SignalBlock(f"{self.name}_slice_{src}_{dst}", n=dst - src)
        for name, signal in self.signals.items():  # Using dict.items() directly
            sliced.signals[name] = Signal(signal.array[src:dst], name)
        return sliced

    def blit(
        self,
        bdata: "SignalBlock",
        src: Optional[int] = None,
        dst: Optional[int] = None,
    ):
        """
        Copies signals from bdata into this group at specified range from
        src (inclusive) to dst (exclusive). If no range specified, copies full length.
        """
        src = src or 0
        dst = dst or src + len(bdata)

        # Create missing signals
        for name, _ in bdata.signals.items():
            if name not in self:
                self.signals[name] = Signal(np.zeros(self.n), name)

        self.reshape(max(self.n, dst))

        # Copy data
        for name, signal in bdata.signals.items():
            self.signals[name].array[src:dst] = signal.array

    @deprecated("simply create a new SignalBlock")
    def has_signal(self, name):
        return name in self.signals

    def __dir__(self):
        """
        Return a list of attributes for the RenderVars instance.
        This includes the standard attributes, methods, and all signal names.
        """
        standard_attrs = set(super().__dir__())  # Get standard attributes and methods
        signal_names = set(self.signals.keys())  # Add all signal names
        protected_attrs = (
            self.__protected__ - standard_attrs
        )  # Add protected names that are not already in standard_attrs
        all_attrs = standard_attrs.union(
            signal_names, protected_attrs
        )  # Combine all attributes

        return sorted(all_attrs)

    def __getitem__(self, key):
        """
        Allows block[key] dict-style access and block[min:max] slicing operations.
        Also provides access to signals stored in __dict__.
        """
        if isinstance(key, slice):
            return self.slice(key.start or 0, key.stop or self.n)
        if isinstance(key, int):
            return self.slice(key, key + 1)
        if key in self.__dict__:
            return self.__dict__[key]
        return self.get_signal_value(key, True)  # Fall back to signal value

    def __setitem__(self, key, value):
        """
        Allows block[k]=v dict-style assignment and block[min:max]=block slices
        Supports both direct Signal assignment and array/value assignment.
        """
        if isinstance(value, SignalBlock):
            if isinstance(key, slice):
                start = key.start or 0
                stop = key.stop or self.n
            elif isinstance(key, int):
                start = key
                stop = key + 1
            self.blit(value, start)
        else:
            # Default to setting as attribute
            setattr(self, key, value)

    def __len__(self):
        return self.n if self.n is not None else self.signals.__len__()

    def __getattr__(self, name: str) -> Any:  # type: ignore[attr-defined, call-arg, arg-type, misc, union-attr]
        """Get a signal by name, creating it if it doesn't exist."""
        if name in self.__protected__:
            return super().__getattr__(name)
        if name not in self.signals:
            self.signals[name] = Signal(name, self.n)
        return self.signals[name]

    def __setattr__(self, key, value):  # type: ignore[attr-defined, misc, union-attr]
        self.set(key, value)

    def __contains__(self, name):
        return self.has_signal(name) or name in self.__dict__

    def load(self, source: "SignalBlock"):
        """
        Loads data from another SignalBlock, replacing all existing signals and reshaping.
        Similar to blit() but clears existing signals first.
        """
        # Clear existing signals
        self.signals.clear()
        self.reshape(source.n)
        self.blit(source)

    @property
    def array_view(self) -> dict:
        """Get a view of just the numpy arrays for all signals"""
        return {name: signal.array for name, signal in self.signals.items()}

    def get_signal_value(self, key, return_zero_on_missing_float_value=False):
        if key in self.__dict__:
            return self.__dict__[key]

        if key in self.signals:
            signal = self.signals[key]

            # I have no idea why this happens randomly
            if isinstance(signal, np.ndarray):
                signal = Signal(signal, key)
                log.error(f"What da fuck, we had a ndarray in the signal!!")

            ret = signal.array if self.is_array_mode else signal[self.f]
            # log.info(f'ret: {ret}')
            return ret
        else:
            # Return zero and create a new signal if it doesn't exist (in array mode)
            if self.is_array_mode:
                self.signals[key] = Signal(np.zeros(self.n), key)
                dat = self.signals[key].array

                return dat
            else:
                if return_zero_on_missing_float_value:
                    return 0.0

                raise AttributeError(
                    f"RenderVars has no attribute '{key}'"
                )  # This has caused too many issues

    def get_array(self, key):
        """Get a signal array specifically, even in frame mode."""
        return self.signals[key].array

    def set(self, key: str, value: np.ndarray | Signal | float | int):
        is_array_assignment = isinstance(value, ndarray) and len(value.shape) == 1
        is_signal_assignment = isinstance(value, Signal)
        is_number_assignment = isinstance(value, (float, int)) and not isinstance(
            value, bool
        )
        is_array_mode = self.__dict__.get("is_array_mode", False)

        if is_array_assignment:
            assert isinstance(value, np.ndarray)
            signal = Signal(value, key)
        elif is_signal_assignment:
            signal = value
        elif is_number_assignment and is_array_mode:
            if key in self.__protected__:
                self.__dict__[key] = value
                # log.info(f"set: {key} is a protected field, setting on __dict__...")
                return

            signal = Signal(np.ones(self.n) * value, key)
        else:
            self.__dict__[key] = value
            return

        if key in self.__protected__:
            log.info(
                f"set_signal: {key} is protected and cannot be set as a signal. Skipping..."
            )  # TODO make a debug message
            return

        array = signal.array
        if self.n > array.shape[0]:
            log.info(f"set: {key} signal is too short. Padding with last value...")
            array = np.pad(array, (0, self.n - array.shape[0]), "edge")
        elif self.n < array.shape[0]:
            log.info(
                f"set: {key} signal is longer than n, extending RenderVars.n to {array.shape[0]}..."
            )
            self.reshape(array.shape[0])
        signal.array = array

        self.signals[key] = signal

    def get_first(self, *keys):
        for key in keys:
            if key in self.signals:
                return self.signals[key].array
            if key in self.__dict__:
                ret = self.__dict__[key]
                if ret is not None:
                    return ret
        return None

    def reshape(self, n):
        if self.n == n:
            return
        self.n = n
        self.resize_signals_to_n()

    def resize_signals_to_n(self):
        # Resize all signals (paddding with zeros)
        for signal in self.signals.values():
            signal.array = np.pad(signal.array, (0, self.n - len(signal.array)))
