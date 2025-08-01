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


# class RenderVars(SignalBlock):  # type: ignore
#     """
#     Manipulate render array & state with large ndarrays called signals.
#     Various
#     """
#
#     null: "RenderVars"
#
#     def __init__(self, session: Optional[Session] = None):
#         super().__init__("rv")
#         self.__protected__ = set(rv_protected)
#         self.session: Optional[Session] = session
#         self.prompt: str = ""
#         self.promptneg: str = ""
#         self.prompt_base: str = ""
#         self.prompt_node: Optional[PNode] = None
#         self.sampler: str = "dpmpp"
#         self.nextseed: int = 0
#         self.w: int = userconf.default_width
#         self.h: int = userconf.default_height
#         self.f: int = 0
#         self.draft: int = 1
#         self.dry: bool = False
#         self.img: Optional[np.ndarray] = None
#         self.prev_img: Optional[np.ndarray] = None
#         self.init_img: Optional[np.ndarray] = None
#         self.is_custom_img: bool = False
#         self.rembg_model: str = "u2net"
#         self._last_cache_f: Optional[int] = None  # Track last cached frame
#
#         # Legacy support
#         self._blocks = {}  # Old group storage
#         self._selected_block = "base"
#
#         self.protect()
#
#     @property
#     def storage(self):
#         if self.session is None:
#             return None
#         return self.session.storage
#
#     @property
#     def f_last(self):
#         if self.session is None:
#             return 0
#         return self.session.f_last
#
#     @property
#     def f_first(self):
#         if self.session is None:
#             return 0
#         return self.session.f_first
#
#     @property
#     def fps(self):
#         if self.session is None:
#             return 24
#         return self.session.fps
#
#     @fps.setter
#     def fps(self, value):
#         if self.session is None:
#             return
#         self.session.fps = value
#
#     @property
#     def w2(self):
#         return self.w // 2
#
#     @property
#     def h2(self):
#         return self.h // 2
#
#     @property
#     def speed(self):
#         return sqrt(self.x ** 2 + self.y ** 2)  # magnitude of x and y
#
#     # noinspection PyAttributeOutsideInit
#     @speed.setter
#     def speed(self, value):
#         # magnitude of x and y
#         speed = self.speed
#         if speed > 0:
#             self.x = abs(self.x) / speed * value
#             self.y = abs(self.y) / speed * value
#
#     @property
#     def duration(self):
#         return self.n / self.fps
#
#     @property
#     def size(self):
#         return self.w, self.h
#
#     @property
#     def t(self):
#         return self.f / self.fps
#
#     @t.setter
#     def t(self, value):
#         self._t = value
#
#     @property
#     def dt(self):
#         return 1 / self.fps
#
#     def __getattr__(self, key) -> int | str | float | np.ndarray:  # type: ignore[attr-defined, call-arg, arg-type, misc, union-attr]
#         """
#         Access signals as object attributes. (x.)
#         """
#         if key in self.__dict__:
#             return self.__dict__[key]
#         return self.get_signal_value(key)
#
#     @dev_function()
#     def set_prompt(self, prompt: str, **node_pool: dict[str, Any]):
#         assert self.session is not None
#
#         from src.party.ravelang import pnodes
#         from src.party import pnodes_std
#         from src.party import maths
#
#         log.info(f"RenderVars.set_prompt({prompt})")
#         cachedir = self.session.dirpath / ".prompts"
#         seed = maths.str_to_seed(self.session.name)
#
#         if not prompt:
#             self.prompt_node = None
#             return
#
#         changed = prompt != self.prompt_base
#         if changed:
#             log.info(
#                 "RenderVars.set_prompt -> change detected, creating root node and baking"
#             )
#
#             # Bake the prompt
#             # ----------------------------------------
#
#             resources = {**pnodes_std.__dict__, **node_pool}
#             t_end = self.n / self.fps
#
#             # Prompt changed mid-run
#             if pnodes.has_cached(cachedir, prompt, seed):
#                 root = pnodes.load_cache(cachedir, prompt, seed)
#                 if root is None:
#                     raise RuntimeError(f"Failed to load cached prompt: {prompt}")
#             else:
#                 root = PNode(prompt, join_num=99999999)
#                 root.unfold(pnodes_std.default_templates, resources)
#                 root.bake(t_end, self.fps, seed=seed)
#                 pnodes.save_cache(cachedir, prompt, root)
#
#             # Log what we did
#             # ----------------------------------------
#             num_nodes = pnodes_utils.count_node_tree(root)
#             log.info(f"pnodes.bake_prompt: Baked {num_nodes} nodes")
#             # if num_nodes < 50000:
#             #     root.print_recursive(t_end)
#             # else:
#             #     log.info(f"pnodes.bake_prompt: Too many nodes to print tree ({num_nodes})")
#
#             root.print_prompt_timeline(0, 10, 2.583)
#
#             self.prompt = root.get_combined_children_text(self.t)
#             self.prompt_base = prompt
#             self.prompt_node = root
#
#     def to_seconds(self, f):
#         return np.clip(f, 0, self.n - 1) / self.fps
#
#     def to_frame(self, t):
#         return int(np.clip(t * self.fps, 0, self.n - 1))
#
#     def zeros(self) -> ndarray:
#         return np.zeros(self.n)
#
#     def ones(self, v=1.0) -> ndarray:
#         return np.ones(self.n) * v
#
#     def set_frames(self, n_frames):
#         if isinstance(n_frames, int):
#             self.n = n_frames
#         if isinstance(n_frames, np.ndarray):
#             self.n = len(n_frames)
#
#     def set_duration(self, duration):
#         self.reshape(int(duration * self.fps))
#         # self.n = int(duration * self.fps)
#         # self.resize_signals_to_n()
#
#     # region Image State
#
#     def set_size(
#         self,
#         w: Optional[int] = None,
#         h: Optional[int] = None,
#         *,
#         frac: int = 64,
#         remote: Optional[tuple[int, int]] = None,
#         resize: bool = True,
#         crop: bool = False,
#     ):
#         """
#         Set the configured target render size,
#         and resize the current frame.
#         """
#         # 1920x1080
#         # 1280x720
#         # 1024x576
#         # 768x512
#         # 640x360
#
#         log.info(f"set_size({w}, {h})")
#
#         w = w or self.w
#         h = h or self.h
#
#         w = int(w)
#         h = int(h)
#
#         draft = 1
#         draft += jargs.args.draft
#
#         if jargs.args.remote and remote:
#             w, h = remote
#
#         self.w = w // self.draft
#         self.h = h // self.draft
#         self.w = self.w // frac * frac
#         self.h = self.h // frac * frac
#
#         if resize and self.img is not None:
#             self.img = self.resize(self.img, crop)
#
#     def resize(self, img: ImageUnion, crop=False) -> ImageUnion:
#         """
#         Resize an image to the dimensions of this RenderVars.
#         """
#         return convert.resize(img, self.w, self.h, crop)
#
#     def seek(self, i=None, *, printing=False):
#         if i is None:
#             # Seek to next
#             # self.f = get_next_leadnum(self.dirpath)
#             self.f = self.f_last + 1
#         elif isinstance(i, int):
#             # Seek to i
#             i = max(i, 1)  # Frames start a 1
#             self.f = i
#         else:
#             log.error(f"Invalid seek argument: {i}")
#             return
#
#         # self._image = None
#         self.load_f()
#         self.load_f_img()
#
#         if printing:
#             log.info(f"({self.name}) seek({self.f})")
#
#     def seek_min(self, *, printing=False):
#         if self.session.dirpath.exists() and any(self.session.dirpath.iterdir()):
#             # Seek to next
#             minlead = get_min_leadnum(self.session.dirpath)
#             self.seek(minlead, printing=printing)
#
#     def seek_max(self, *, printing=False):
#         if self.session.dirpath.exists() and any(self.session.dirpath.iterdir()):
#             self.f = get_max_leadnum(self.session.dirpath)
#             self.seek(self.f, printing=printing)
#
#     def seek_next(self, i=1, printing=False):
#         self.f += i
#         self.seek(self.f, printing=printing)
#
#     def seek_new(self, printing=False):
#         self.seek(None, printing=printing)
#
#     def delete_f(self, f=None):
#         # Delete the current frame
#         f = f or self.f
#         if f is None:
#             return
#
#         path = None
#         exists = None
#         if f == self.f:
#             path = self.session.dirpath.det_frame_path(f)
#             exists = path.exists()
#
#         if exists:
#             if f == self.f:
#                 path.unlink()
#                 self.f = np.clip(self.f - 1, 0, self.f_last)
#                 self.session.f_last = self.f or 0
#                 self.f_last_path = self.session.det_frame_path(self.f_last) or 0
#                 self.load_f()
#             else:
#                 # Offset all frames after to make sequential
#                 path.unlink()
#                 # TODO this shouldn't be needed
#                 self.session.make_sequential()
#                 self.session.load()
#
#                 log.info(f"Deleted {path}")
#                 return True
#
#         return False
#
#     @trace_decorator
#     def load_f(self, f=None, *, clamp_to_last=False, img=True):
#         """
#         Load the statistics for the current frame or any specified frame,
#         and load the frame image.
#         """
#         f = f or self.f
#
#         self.f_exists = False
#         if img:
#             if self.f <= self.f_last:
#                 self.f_exists = self.load_f_img(default="rv")
#             elif clamp_to_last:
#                 self.f_exists = self.load_f_img(default="rv")
#
#     @trace_decorator
#     def load_f_img(self, path: str | int | None = None, default=UNDEFINED):
#         """
#         Load a file or a frame by number.
#         Args:
#             path: A path to a file
#             @param path:
#             @param default: The default image to load if the file does not exist.
#         """
#         if path is None:
#             path = self.f
#
#         self.is_custom_img = False
#
#         match path:
#             case int() | np.int32() | np.int64() | float():
#                 path = self.session.det_frame_path(int(path))
#             case str():
#                 path = Path(path)
#             # case Path() as p if p.is_absolute(): path = p
#             # case Path(): path = self.session.dirpath / self.file
#             case _:
#                 raise ValueError(f"Unsupported path type: {type(path)}")
#
#         if path.suffix in paths.image_exts:
#             if path.exists():
#                 self.img = convert.load_cv2(path)
#                 self.f_exists = True
#                 return True
#
#         if default is not UNDEFINED:
#             self.img = self.make_filler(default)
#             return False
#
#         raise FileNotFoundError(
#             f"Could not find frame {self.f} in session {self.session.name}"
#         )
#
#     def set_img_custom(self, img):
#         self.img = img
#         self.is_custom_img = True
#
#     @property
#     def rembg(self) -> 'Rembg':
#         """Get the singleton instance of rembg, using the model specified in rembg_model signal."""
#         from src.party.models.rembg_ import Rembg
#         model_type = "u2net"  # Default model
#         if self.has_signal("rembg_model"):
#             model_type = self["rembg_model"]
#         return Rembg.get_instance(model_type)
#
#     def compute_cache_hash(self, extra_paths: list[str]) -> str:
#         """Compute a hash of the session script and relevant source files"""
#         if not self.session:
#             return None
#
#         script_path = self.session.res_script()
#         if not script_path or not os.path.exists(script_path):
#             return None
#
#         hash_obj = hashlib.sha256()
#         with open(script_path, 'rb') as f:
#             hash_obj.update(f.read())
#
#         for pattern in extra_paths:
#             for source_path in glob.glob(str(paths.root / pattern), recursive=True):
#                 if os.path.exists(source_path) and os.path.isfile(source_path):
#                     with open(source_path, 'rb') as f:
#                         hash_obj.update(f.read())
#
#         return hash_obj.hexdigest()
#
#     def get_cache_path(self, hash:str) -> Path:
#         """Get the cache path based on script hash"""
#         if not self.session:
#             return None
#
#         cache_dir = self.session.dirpath / "signals"
#         return cache_dir / f"script_{hash}.npz"
#
#     def init_frame(self, f, cleanse_null_images=True, init_res='init'):
#         """
#         Prime the render state for rendering a frame.
#         Signals will be loaded and the frame will be resized.
#         Null frame images will be replaced with black images.
#         """
#         self.is_array_mode = False
#         self.dry = False
#         self.nextseed = random.randint(0, 2 ** 32 - 1)
#         self.seed = self.nextseed
#         self.f = int(f)
#         self.ref = 1 / 12 * self.fps
#         self.tr = self.t * self.ref
#
#         self.resize_signals_to_n()
#         self.load_signal_floats()
#
#         if self.f > 1:
#             self.img_f = self.res_frame_cv2(self.f, default="black")
#             self.img_f1 = self.res_frame_cv2(self.f - 1, default="black")
#             if not self.is_custom_img:
#                 self.img = self.res_frame_cv2(self.f - 1, default="black")
#         else:
#             self.img_f = np.zeros((self.h, self.w, 3), dtype=np.uint8)
#             self.img_f1 = np.zeros((self.h, self.w, 3), dtype=np.uint8)
#             self.img = np.zeros((self.h, self.w, 3), dtype=np.uint8)
#
#         # Automatically match size
#         if self.w is None:
#             self.w = self.img.shape[1]
#         if self.h is None:
#             self.h = self.img.shape[0]
#
#         self.prev_img = self.img_f1
#
#         # Load this frame's init image
#         # ----------------------------------------
#
#         if init_res:
#             self.init_img = self.get_init_img(init_res, self.f)
#             self.has_init_frame = self.init_img is not None
#             if not self.has_init_frame:
#                 self.init_img = self.make_filler('black')
#
#     def get_init_img(self, init_res, f):
#         frame_path = self.session.res_frame(init_res, f)
#         if not frame_path.exists():
#             return None
#
#         # Get transforms
#         pivot = (0.5, 0.5)
#         if self.has_signal(RV_SIGNAL_INIT_ANCHOR_X): pivot = (self[RV_SIGNAL_INIT_ANCHOR_X], pivot[1])
#         if self.has_signal(RV_SIGNAL_INIT_ANCHOR_Y): pivot = (pivot[0], self[RV_SIGNAL_INIT_ANCHOR_Y])
#         if self.has_signal(RV_SIGNAL_INIT_PIVOT_X): pivot = (self[RV_SIGNAL_INIT_PIVOT_X], pivot[1])
#         if self.has_signal(RV_SIGNAL_INIT_PIVOT_Y): pivot = (pivot[0], self[RV_SIGNAL_INIT_PIVOT_Y])
#
#         sizing = "crop"
#         if self.has_signal(RV_SIGNAL_INIT_STRETCH) and self[RV_SIGNAL_INIT_STRETCH] > 0.5:
#             sizing = "resize"
#
#         iimg = self.res_frame_cv2("init", f, origin=pivot, size_mode=sizing, default="black")
#         if iimg is not None:
#             iimg = cv2.resize(iimg, (self.w, self.h), interpolation=cv2.INTER_AREA)
#
#             # Apply rembg if use_rembg is set
#             if self.has_signal("use_rembg") and self["use_rembg"] > 0.5:
#                 log.info(f"Applying rembg to iimg ...")
#
#                 # Get rembg parameters if available
#                 bg_color = (0, 0, 0)  # Default black background
#                 if self.has_signal("rembg_bg_color"):
#                     bg_color = tuple(int(x * 255) for x in self["rembg_bg_color"])
#
#                 blur_radius = 0
#                 if self.has_signal("rembg_blur"):
#                     blur_radius = int(self["rembg_blur"] * 10)  # Scale 0-1 to 0-10
#
#                 threshold = 128
#                 if self.has_signal("rembg_threshold"):
#                     threshold = int(
#                         self["rembg_threshold"] * 255
#                     )  # Scale 0-1 to 0-255
#
#                 # Process with rembg using the configured model
#                 iimg = self.rembg.replace_bg(
#                     iimg,
#                     bg_color=bg_color,
#                     threshold=threshold,
#                     blur_radius=blur_radius,
#                 )
#
#         if self.has_signal(RV_SIGNAL_INIT_ZOOM):
#             from src.party import maths, tricks
#             iimg = tricks.mat2d(iimg, z=self.init_zoom)
#
#         return iimg
#
#     def res_frame_cv2(
#         self,
#         resname,
#         f=None,
#         *,
#         loop=False,
#         size_mode="auto",
#         origin=(0.5, 0.5),
#         default=UNDEFINED,
#     ):
#         ret = self.session.res_frame_cv2(
#             resname, f, loop=loop, fps=self.fps, default=default
#         )
#         if isinstance(ret, np.ndarray):
#             return self.match_size(origin, ret, size_mode)
#
#         return self.make_filler(default)
#
#     def match_size(self, origin, input_img, size_mode: str):
#         if not size_mode or input_img is None or self.img is None:
#             return input_img
#
#         if size_mode == "auto":
#             # switch to crop if the frame is larger than the image, otherwise resize up
#             if (
#                 input_img.shape[0] > self.img.shape[0]
#                 or input_img.shape[1] > self.img.shape[1]
#             ):
#                 size_mode = "crop"
#             else:
#                 size_mode = "resize"
#
#         if size_mode == "crop":
#             input_img = convert.crop_or_pad(
#                 input_img, self.img.shape[1], self.img.shape[0], "black", anchor=origin
#             )
#         elif size_mode == "resize" or size_mode == "rescale" or size_mode == "stretch":
#             input_img = cv2.resize(input_img, (self.img.shape[1], self.img.shape[0]))
#
#         return input_img
#
#     def make_filler(self, filler_type):
#         match filler_type:
#             case "none":
#                 return None
#             case "rv":
#                 return self.img
#             case "img":
#                 return self.img
#             case "black":
#                 return np.zeros((self.h, self.w, 3), dtype=np.uint8)
#             case "white":
#                 return np.ones((self.h, self.w, 3), dtype=np.uint8) * 255
#             case "pink":
#                 return np.stack(
#                     (
#                         np.ones((self.h, self.w, 3), dtype=np.uint8) * 255,
#                         np.zeros((self.h, self.w, 3), dtype=np.uint8),
#                         np.zeros((self.h, self.w, 3), dtype=np.uint8),
#                     ),
#                     axis=2,
#                 )
#
#     # endregion
#
#     def get_constants(self):
#         """
#         Get the constants for the current render configuration.
#         """
#         from src.party.maths import np0, np01
#
#         n = self.n
#         t = np01(n)
#         indices = np0(self.n - 1, self.n)
#
#         return n, t, indices
#
#     @deprecated("simply handle SignalBlocks externally")
#     def clear_signals(self):
#         # remove from __dict__
#         for k, v in self.signals.items():
#             if k in self.__dict__:
#                 del self.__dict__[k]
#             if f"{k}s" in self.__dict__:
#                 del self.__dict__[f"{k}s"]
#
#         self.signals.clear()
#         self._blocks.clear()
#         self._selected_block = None
#
#     def load_signal_floats(self):
#         """
#         Load the current frame values for each signal into __dict__.
#         """
#         for name, value in self.signals.items():
#             signal = self.signals[name]
#             try:
#                 # log.info("fetch", name, self.f, len(signal))
#
#                 f = self.f
#                 self.__dict__[f"{name}s"] = signal
#                 if self.f > len(signal) - 1:
#                     self.__dict__[name] = 0
#                 else:
#                     self.__dict__[name] = signal[f]
#
#             except IndexError:
#                 log.info(
#                     f"rv.set_frame_signals(IndexError): {name} {self.f} {len(signal)}"
#                 )
#
#     def load_signal_arrays(self):
#         """
#         Load the current frame values for each signal into __dict__.
#         """
#         for name, value in self.signals.items():
#             self.__dict__[name] = value.array
#             self.__dict__[f"{name}s"] = value
#
#     # region Signal Groups
#     @deprecated("rv.select_group(name)")
#     def select_gsignal(self, name: str):
#         """Legacy method to select a signal group"""
#         if name not in self._blocks:
#             self._block.signalss[name] = SignalBlock(name, self.n)
#
#         # Keep old API references in sync
#         self._selected_block = name
#         self.signals = self._block.signalss[name].signals
#         self._block.signalss[name] = self.signals
#
#         return self._block.signalss[name]
#
#     @deprecated("source_group.slice(start, end)")
#     def copy_gframe(self, src_name: str, dst_name: str, i: int):
#         """Legacy method to copy a single frame between groups"""
#         source = self._blocks.get(src_name) or self.create_group(src_name)
#         dest = self._blocks.get(dst_name) or self.create_group(dst_name)
#
#         # Copy just the single frame
#         dest.blit(source.slice(i, i + 1), i)
#
#     @deprecated("dest_group.blit(source_group[start:end], start)")
#     def copy_gframes(self, src_name: str, dst_name: str, i_start: int, i_end: int):
#         """Legacy method to copy frames between groups"""
#         source = self._blocks.get(src_name) or self.create_group(src_name)
#         dest = self._blocks.get(dst_name) or self.create_group(dst_name)
#
#         dest.blit(source.slice(i_start, i_end), i_start)
#
#     @deprecated("dest_group.blit(source_group)")
#     def copy_gsignal(self, src_name: str, dst_name: str = None):
#         """Legacy method to copy entire signal groups"""
#         if dst_name is None:
#             dst_name = self._selected_block
#
#         source = self._blocks.get(src_name) or self.create_group(src_name)
#         dest = self._blocks.get(dst_name) or self.create_group(dst_name)
#
#         dest.blit(source)
#
#     def create_group(self, name: str):
#         """Create a new SignalBlock with the specified name"""
#         block = SignalBlock(name, self.n)
#         self._block.signalss[name] = block
#         return block
#
#     # Legacy aliases with deprecation warnings
#     @deprecated("rv.select_group(name)")
#     def set_signal_set(self, name: str):
#         return self.select_gsignal(name)
#
#     @deprecated("source_group.slice(start, end)")
#     def copy_set_frame(self, src_name: str, dst_name: str, i: int):
#         return self.copy_gframe(src_name, dst_name, i)
#
#     @deprecated("dest_group.blit(source_group[start:end], start)")
#     def copy_set_frames(self, src_name: str, dst_name: str, i_start: int, i_end: int):
#         return self.copy_gframes(src_name, dst_name, i_start, i_end)
#
#     @deprecated("dest_group.blit(source_group)")
#     def copy_set(self, src_name: str, dst_name: str = None):
#         return self.copy_gsignal(src_name, dst_name)
#
#     def get_targets(self, dst_name: str, src_name: str):
#         """Legacy helper that maps old style group access to new API"""
#         warnings.warn(
#             "get_targets is deprecated. Use SignalGroup objects directly instead.",
#             DeprecationWarning,
#             stacklevel=2,
#         )
#
#         # Get or create the source group
#         src = self._blocks.get(src_name)
#         if not src:
#             src = self.create_group(src_name)
#
#         # Handle destination group
#         is_current = dst_name is None or dst_name == self._selected_block
#         if is_current:
#             dst = self._selected_block
#         else:
#             dst = self._blocks.get(dst_name) or self.create_group(dst_name)
#
#         return (
#             dst.signals,
#             is_current,
#             src.signals,
#         )  # Return in old format for compatibility
#
#     @deprecated("simply handle SignalBlocks externally")
#     def set_signal_set(self, name):
#         self.select_gsignal(name)
#
#     @deprecated("simply handle SignalBlocks externally")
#     def copy_set_frame(self, src_name, dst_name, i):
#         self.copy_gframe(self, src_name, dst_name, i)
#
#     @deprecated("simply handle SignalBlocks externally")
#     def copy_set_frames(self, src_name, dst_name, i_start, i_end):
#         self.copy_gframes(src_name, dst_name, i_start, i_end)
#
#     @deprecated("simply handle SignalBlocks externally")
#     def copy_set(self, src_name, dst_name=None):
#         self.copy_gsignal(src_name, dst_name)
#
#     # endregion


RenderVars.null = RenderVars()


# 	def get_timestamp_string(self, signal_name):
# 		import datetime
# 		ret = '''
# chapters = PTiming("""
# '''
# 		indices = np.where(self._signals[signal_name] > 0.5)[0]
# 		for v in indices:
# 			from src.gui.old.RyucalcPlot import to_seconds
# 			time_s = to_seconds(v)
# 			delta = datetime.timedelta(seconds=time_s)
# 			s = str(delta)
# 			if '.' not in s:
# 				s += '.000000'
#
# 			ret += s + "\n"
#
# 		ret += '"""'
# 		return ret
