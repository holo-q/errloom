"""
A renderer with a common interface to communicate with.
The renderer has all the logic you would find in a simple game engine,
so it keeps track of time and targets for a specific FPS.

A 'render script' must be loaded which is a python file which implements a render logic.
The file is watched for changes and automatically reloaded when it is modified.
A valid render script has the following functions:

- def on_callback(rv, name)  (required)

Various renderer events can be hooked with this on_callback by checking the name,
and will always come in this order:

- 'load' is called when the script is loaded for the very first time.
- 'setup' is called whenever the script is loaded or reloaded.

The render script is universal and can be used for other purpose outside of the renderer
by simply calling on_callback with a different name.

Devmode:
    - We will not check for script changes every frame.

CLI Mode:
    - Always rendering on
    - Starts from last frame

GUI Mode:
    - Initialized in STOP mode
    - Starts from first frame

"""
import datetime
from enum import Enum
import importlib
import logging
from pathlib import Path
import sys
import time
import traceback
import types
from typing import Optional, Union

import numpy as np
from PIL import Image
from yachalk import chalk

from jargs import args
from src import paths
from src.convert import TImage, get_cv2
from src.hud import HUD
from src.lib import loglib
from src.lib.corelib import invoke_safe
from src.lib.loglib import trace_decorator, tracer
from src.rendervars import RenderVars
from errloom.session import Session


SCRIPT_FUNC_INIT = 'init'
SCRIPT_STR_INIT_RES = 'media'

SCRIPT_BOOL_DEMUCS = 'demucs'
SCRIPT_BOOL_PROMPT_REALITME = 'prompt_realtime'
UNDEFINED = object()


allowed_module_reload_names = [
    'scripts.',
    'src.party.',
    # 'src.classes.',
]

script_change_detection_paths = [
    paths.root_scripts,
    paths.root_src,
    # -> Also checks the session path automatically as part of the logic
]

# These globals will NOT be persisted when reloading the script
module_reloader_blacklisted_global_persistence = [
    'prompt'
]

log = logging.getLogger('renderer')

current = None

class RenderMode(Enum):
    PAUSE = 0
    PLAY = 1
    RENDER = 2

class RenderRepetitionMode(Enum):
    ONCE = 0
    FOREVER = 1

class InitLevel(Enum):
    UNINITIALIZED = 0
    LIGHT = 1
    HEAVY = 2


class RendererStep(Enum):
    INITIALIZATION = 0
    READY = 1
    RENDERING = 2
    STOPPED = 3

class SeekImageMode(Enum):
    NORMAL = 0
    NO_IMAGE = 1
    IMAGE_ONLY = 2

class SeekRequest:
    def __init__(self, f_target, img_mode=SeekImageMode.NORMAL):
        self.f_target = f_target
        self.img_mode = img_mode

class PauseRequest(Enum):
    NONE = 0
    PAUSE = 1
    PLAY = 2
    TOGGLE = 3

class TickState:
    """
    Information, state and signals pertaining to the last tick() call
    Reset on every tick() call, do not depend on persistence
    """

    def __init__(self):
        self.was_paused = False
        self.dt = 0
        self.f_changed = False
        self.just_seeked = False
        self.just_paused = False
        self.just_unpaused = False
        self.fast_mode = False
        self.catchup_end = False
        self.catchedup = False
        self.finished_render = False
        self.dry_draw = False
        self.reached_f_last = False

class DrawState:
    def __init__(self):
        self.last_prompt = ''

class LoopStats:
    def __init__(self):
        self.n_rendered = 0  # Number of frames rendered
        self.draw_start_time = 0  # Time when we switched to rendering state, resets on pause/stop/playback/etc.
        self.last_rendered_f = None
        self.last_script_check = 0

    @property
    def draw_duration_seconds(self):
        return (time.time() - self.draw_start_time) * 1000

    @property
    def draw_duration_real(self):
        return datetime.timedelta(seconds=self.draw_duration_seconds)


class RendererRequests:
    """
    The renderer runs in a loop and we wish to do everything within that loop
    so we can handle any edge cases. The session must not change while rendering,
    for example.
    """

    def __init__(self):
        self.seek: SeekRequest | None = None  # Frames to seek to
        self.script_check = False  # Check if the script has been modified
        self.render = None
        self.pause = PauseRequest.NONE  # Pause the playback/renderer
        self.stop = False  # Stop the whole renderer


class PlaybackState:
    def __init__(self):
        self.from_f = 1
        self.until_f = None  # The frame to stop at. None means never end
        self.looping = False  # Enable looping
        self.loop_frame = 0  # Frame to loop back to
        self.dt_accumulated = 0
        self.allow_past_end = False
        self.audio = None

class Renderer:
    null:'Renderer'

    def __init__(self,
                 session:Optional[Session]=None,
                 *,
                 gui=False,
                 is_null=False,
                 readonly=args.readonly,
                 devmode=args.dev):
        # Options
        self.enable_readonly = readonly  # Cannot render, used as an image viewer
        self.enable_save = True  # Enable saving the frames
        self.enable_save_hud = True  # Enable saving the HUD frames
        self.enable_unsafe = args.unsafe  # Should we let calls to the script explode so we can easily debug?
        self.enable_bake_on_script_reload = True  # Should we bake the script on reload? TODO move this to the UI
        self.enable_dynamic_hud_populate = False  # Dry runs to get the HUD data when it is missing, can be laggy if the script is not optimized
        self.detect_script_every = -1 if gui else 1  # in gui mode we watch for focus events instead which is less laggy

        # State (internal)
        self._is_dev = devmode or readonly  # Are we in dev mode?
        self._is_gui = gui  # Are we in GUI mode? (hobo, ryusig, audio)
        self._is_gui_dispatch = True  # Whether or not the GUI handles the dispatching of iter_loop
        self._script = None  # The script module
        self._script_level = InitLevel.UNINITIALIZED
        self._script_baked = False
        self._script_name = ''  # Name of the script file
        self._script_modpath = None
        self._script_time_cache = {}
        self._has_script_error = False  # Whether a hotreload
        self._has_frame_error = False  # Whether the last frame errored out
        self.draw_progress = 0

        # Public
        self.state = RendererStep.READY
        self.mode = RenderMode.PAUSE
        self.render_repetition = RenderRepetitionMode.FOREVER
        self.rv = RenderVars()
        self.requests = RendererRequests()
        self.playback = PlaybackState()
        self.stats = LoopStats()
        self.hud = HUD(self.is_cli)
        self.tickstate = TickState()

        # Signals
        self.draw_callback = None

        if is_null:
            return

        self.set_session(session)

    # self.set_globals(on_script=True, on_libs=True)

    def _invoke_safe(self, *args, **kwargs):
        if 'unsafe' not in kwargs:
            kwargs['unsafe'] = self.enable_unsafe
        return invoke_safe(*args, **kwargs)

    @property
    def script(self):
        return self._script

    @property
    def session(self):
        return self.rv.session

    @session.setter
    def session(self, value):
        self.rv.session = value

    @property
    def is_cli(self):
        return not self._is_gui


    @property
    def is_gui(self):
        return self._is_gui and not args.remote


    @property
    def is_gui_dispatch(self):
        return self._is_gui and self._is_gui_dispatch  # and self.mode.value < RenderMode.RENDER.value

    @property
    def stop(self):
        if not self.session: return
        self.requests.stop = True

    @property
    def is_paused(self):
        return self.mode == RenderMode.PAUSE

    @property
    def is_playing(self):
        return self.mode == RenderMode.PLAY

    @property
    def is_running(self):
        return self.state == RendererStep.READY or self.state == RendererStep.RENDERING

    @property
    def is_dev(self):
        return self._is_dev

    @is_dev.setter
    def is_dev(self, value):
        self._is_dev = value

    @property
    def has_frame_error(self):
        return self._has_frame_error

    @property
    def has_script_error(self):
        return self._has_script_error

    @property
    def reached_n_max(self):
        self.rv.f
        return self.rv.f >= self.rv.n

    @property
    def init_level(self):
        return self._script_level

    def is_run_complete(self):
        return self.mode == RenderMode.RENDER and self.rv.f >= self.rv.n - 1

    def should_draw(self):
        ret = self.mode == RenderMode.RENDER
        ret = ret and (self.rv.f != self.stats.last_rendered_f or self.mode == RenderMode.PAUSE)  # Don't draw the same frame twice (if we draw faster than playback)
        ret = ret and not self._has_script_error and not self._has_frame_error  # User must fix the script before we can draw
        return ret

    def should_save(self):
        return self.enable_save and not self._is_dev

    def set_image(self, img: TImage):
        img = get_cv2(img, target_size=(self.rv.w, self.rv.h))
        self.rv.set_img(img)

    def set_session(self, session: Optional[Session]):
        self.rv.__init__(session)
        self.requests.__init__()

        if session is None:
            return

        self.rv.fps = session.fps
        self.rv.n = int(session.f_last)
        self.rv.load_f_img(default='black')

        if self._is_gui:
            self.seek(session.f_last)
            self.requests.pause = True  # In GUI mode we let the user decide when to start the render
        else:
            self.seek(session.f_last + 1)

        with tracer('Script loading'):
            self._script = None
            # self._invoke_safe(self.load_script)
            pass

        self.rv.init_frame(self.rv.f)
        # self.set_globals(on_script=False, on_libs=True)

        if self.is_cli:
            self.ensure_bake()

    # TODO we may not need this anymore
    # if self._is_gui:
    # 	from src.gui import hobo
    # 	hobo.on_session_changed(self.rv.session)

    def toggle_play(self):
        if not self.session: return
        if self.mode == RenderMode.RENDER: return
        self.playback.looping = False
        self.playback.until_f = 0
        self.requests.pause = PauseRequest.TOGGLE
        if self.mode == RenderMode.RENDER:
            self.mode = RenderMode.PLAY


    def set_pause(self):
        """
        Set the pause state.
        """
        if not self.session: return
        if self.mode == RenderMode.RENDER: return
        self.playback.looping = False
        self.playback.until_f = 0
        self.requests.pause = PauseRequest.PAUSE


    def set_play(self):
        """
        Set the play state.
        """
        if not self.session: return
        if self.mode == RenderMode.RENDER: return
        self.playback.looping = False
        self.playback.until_f = 0
        self.requests.pause = PauseRequest.PLAY


    def seek(self, f_target, *, with_pause=None, clamp=False, img_mode=SeekImageMode.NORMAL):  # TODO we need to check usage of this to update imgmode
        """
        Seek to a frame.
        Note this is not immediate, it is a request that will be handled as part of the render loop.
        """
        if not self.session:
            return False
        if self.mode == RenderMode.RENDER:
            return False

        if isinstance(f_target, str):
            f_seconds = paths.parse_time_to_seconds(f_target)
            f_target = self.rv.to_frame(f_seconds)

        if clamp:
            if self.session.f_first is not None and f_target < self.session.f_first:
                f_target = self.session.f_first
            if self.session.f_last is not None and f_target >= self.session.f_last + 1:
                f_target = self.session.f_last + 1

        f_target = max(f_target, 0)

        self.requests.seek = SeekRequest(f_target, img_mode)
        self.playback.looping = False

        if with_pause is not None:
            self.requests.pause = with_pause

        # if not pause:
        #     print(f'NON-PAUSING SEEK MAY BE BUGGY')
        return True


    def seek_t(self, t_target):
        """
        Seek to a time in seconds.
        Note this is not immediate, it is a request that will be handled as part of the render loop.
        Args:
            t_target: The time in seconds.
        """
        f_target = int(self.rv.fps * t_target)
        self.seek(f_target)

    def toggle_render_repetition(self):
        if not self.session: return
        if self.render_repetition == RenderRepetitionMode.ONCE:
            self.render_repetition = RenderRepetitionMode.FOREVER
        else:
            self.render_repetition = RenderRepetitionMode.ONCE


    def toggle_render(self, repetition=RenderRepetitionMode.FOREVER):
        if not self.session: return
        self.render_repetition = repetition
        if self.mode == RenderMode.RENDER:
            self.set_render_mode(RenderMode.RENDER)
        else:
            self.set_render_mode(RenderMode.RENDER)


    def render_once(self):
        """
        Render a frame.
        Note this is not immediate, it is a request that will be handled as part of the render loop.
        """
        if not self.session: return
        self.set_render_mode(RenderMode.RENDER)
        self.render_repetition = RenderRepetitionMode.ONCE


    def render_forever(self):
        if not self.session: return
        self.set_render_mode(RenderMode.RENDER)
        self.render_repetition = RenderRepetitionMode.FOREVER


    def set_render_mode(self, newmode):
        if not self.session: return
        self.mode = newmode
        self._has_frame_error = False

    # region Script
    def print_script_fn(self, funcname):
        log.info(f'--> {self.session.name}.{self._script_name}.{funcname}')


    @trace_decorator
    def script_emit(self, name, **kwargs):
        self.set_globals(on_script=True, on_libs=True)
        # log.info("")
        print("")
        self.print_script_fn(name)

        if hasattr(self._script, name):
            func = getattr(self._script, name)
            return self._invoke_safe(func, **kwargs)
        if hasattr(self._script, 'callback'):
            return self._invoke_safe(self._script.callback, name, **kwargs)


    def script_get(self, name, default=None):
        if self._script is not None:
            return self._script.__dict__.get(name, default)

    def script_bool(self, name, default_value=False) -> bool:
        if self._script is not None:
            return self._script.__dict__.get(name, default_value) is True

        return default_value

    def script_has(self, name):
        if self._script is not None:
            return self._script.__dict__.get(name, None) is not None


    def collect_hotreload_changes(self, dry=False):
        """
        Detect changes in the scripts folder and reload all modules.
        Returns:
            Changed files (list of Path),
            Added files (list of Path)
        """
        if self.session is None:
            return [], []

        def detect_changes(path):
            l_changed = []
            l_added = []

            # Check all .py recursively
            for file in Path(path).rglob('*.py'):
                key = file.relative_to(paths.root).as_posix()
                is_new = key not in self._script_time_cache
                if is_new:
                    if not dry: self._script_time_cache[key] = file.stat().st_mtime
                    l_added.append(file)
                    continue

                # Compare last modified time of the file with the cached time
                # mprint("Check file:", file.relative_to(path).as_posix())
                if self._script_time_cache[key] < file.stat().st_mtime:
                    if not dry: self._script_time_cache[key] = file.stat().st_mtime
                    l_changed.append(file)
                    self.modified = True

            # mprint(key, file.stat().st_mtime)
            return l_changed, l_added

        all_changed = []
        all_added = []
        detection_paths = [*script_change_detection_paths]
        if paths.root in self.session.dirpath.parents or self.session.dirpath == paths.root:
            detection_paths.append(self.session.dirpath)

        for p in detection_paths:
            changed, added = detect_changes(p)
            all_changed.extend(changed)
            all_added.extend(added)

        return all_changed, all_added


    def apply_hotreload_if_changed(self):
        if self.script is None: return

        self.had_errored = self._has_script_error
        self._has_script_error = False

        changed, added = self.collect_hotreload_changes()
        if not changed and not added:
            return

        log.info('')
        log.info('==============================================')
        for filepath in changed:
            modpath = paths.filepath_to_modpath(filepath)
            if modpath is None:
                continue

            log.info(chalk.dim(chalk.blue(f"Change detected in {modpath} ({filepath})")))

            self.load_module(modpath)
            if modpath == self._script_modpath:
                log.info(f"Saving script backup for frame {self.rv.f}")
                self.session.save_script_backup(self.session.f_last)

                if self._script_baked:
                    log.info("Re-baking script to ensure signals are up to date ...")
                    self.invoke_bake()

        is_rendering = self.mode == RenderMode.RENDER
        if is_rendering and self._has_script_error:
            # Stop rendering if there was an error
            self._has_frame_error = True
            self.set_pause()
        elif not is_rendering and not self._has_script_error:
            self._has_frame_error = False
            if not self.is_gui:  # In CLI mode, script change is a resume signal
                self.render_forever()

    def reload_script(self, hard=False):
        if self.session is None:
            log.info("No session, cannot reload script")
            return

        if hard:
            self._script_level = InitLevel.UNINITIALIZED
            log.info("Reloading script (hard) ...")
        else:
            log.info("Reloading script...")

        self.load_script()
        if self.enable_bake_on_script_reload:
            self.ensure_bake()

        # Switch back to rendering automatically if it was stopped (error)
        if not self.mode == RenderMode.RENDER and self.is_cli:
            self.mode = RenderMode.RENDER

    def load_script(self):
        """
        Load or reload the script.
        """
        if self.session is None: return
        if self.script is not None:
            log.error("Script is already loaded!")
            return

        self._script = None
        self._script_baked = False
        self._has_frame_error = False
        self._has_script_error = False

        with tracer('renderer.load_script'):
            filepath = self.session.res_script(touch=True)
            modpath = paths.filepath_to_modpath(filepath, default=None)
            if modpath is None:
                log.warn("Could not resolve module path for script, session will be readonly.")
                return

            self._script_name = 'script'
            self._script_modpath = modpath
            if not modpath:
                self._has_script_error = True
                return

            if self.load_module(module_path=modpath):
                # any script loaded callback would be placed here
                self.session.save_script_backup()
            else:
                self._has_script_error = True


    def load_module(self, module_path, keep_globals=True) -> bool:
        """
        Load or reload a module, returns True if
        successful and no errors were encountered.
        """
        is_error = False
        module = None

        def error():
            nonlocal is_error
            chalk.red("<!> SCRIPT ERROR <!>")
            traceback.print_exc()
            is_error = True

        def reload_script():
            if self._script is None:
                self.rv.__init__(self.rv.session)
                self.print_script_fn('import')
                self._script = importlib.import_module(module_path, package='imported_renderer_script')
            else:
                self.print_script_fn('reload')
                importlib.reload(self._script)

        def reload_module():
            nonlocal module
            module = sys.modules.get(module_path, None)
            if module is None:
                return

            importlib.reload(module)

        oldglobals = None
        if module_path in sys.modules:
            try:
                module = importlib.import_module(module_path)
                oldglobals = module.__dict__.copy()
                oldglobals = {k: v for k, v in oldglobals.items() if not isinstance(v, types.FunctionType) and k not in module_reloader_blacklisted_global_persistence}  # Don't keep functions
            except Exception:
                if self.enable_unsafe: raise
                error()

        is_script = module_path == self._script_modpath
        try:
            if is_script:
                reload_script()
            else:
                reload_module()
        except Exception:
            if self.enable_unsafe: raise
            error()
            if is_script:
                self._has_script_error = True

        # Restore the globals
        if keep_globals and module is not None and oldglobals is not None and module_path in sys.modules:
            # # Remove globals that were initialized by the module (newer)
            # for k in list(module.__dict__.keys()):
            # 	if k not in oldglobals:
            # 		del module.__dict__[k]
            module.__dict__.update(oldglobals)

        # update functions for every instance of each class defined in the module using gc
        # import inspect
        # classdefs = dict()
        # classtypes = list()
        # for name, classobj in module.__dict__.items():
        # 	if inspect.isclass(classobj) and classobj.__module__ == module.__name__:
        # 		classdefs[name] = classobj
        # 		classtypes.append(classobj)
        # # convert classtypes to a tuple
        # classtypes = tuple(classtypes)
        # import gc
        # for obj in gc.get_objects():
        # 	if isinstance(obj, classtypes):
        # 		log.info(f"Updating '{obj.__class__.__name__}' class type")
        # 		newclass = classdefs[obj.__class__.__name__]
        # 		# Is one of the updated classes --> update the functions
        # 		for name, obj in obj.__dict__.items():
        # 			# Get the new one and set
        # 			if isinstance(obj, types.FunctionType):
        # 				log.info(f'Updating {obj.__name__}')
        # 				newobj = getattr(newclass, name)
        # 				setattr(obj, name, newobj)

        return not is_error


    def ensure_init(self, init_level: InitLevel = InitLevel.LIGHT):
        """
        Invoke script initialization functions if the script is not
        yet in an initialization state.
        """
        if self._script is None:
            self.load_script()
            if self._script is None:
                return False

        # Things that are not strictly necessary, but are useful for most scripts
        if self._script_level.value < init_level.value:
            self.script_emit(SCRIPT_FUNC_INIT)
            self._script_level = InitLevel.LIGHT

        if self._script_level.value < init_level.value:
            try:
                self.script_emit('init_heavy', unsafe=True)
                self._script_level = InitLevel.HEAVY
            except Exception:
                if self.enable_unsafe:
                    raise
                else:
                    chalk.red("<!> SCRIPT ERROR <!>")
                    traceback.print_exc()
                    self._has_script_error = True

                return False

        return True

    def ensure_bake(self):
        if self._script_baked:
            return

        self.ensure_init(InitLevel.LIGHT)
        self.invoke_bake()
        self._script_baked = True

    @trace_decorator
    def invoke_bake(self):
        from src.party import maths, std
        # log.info("")
        print("")
        log.info("--> pre-bake")

        self.set_globals(on_libs=True, on_script=True)
        maths.reset()
        maths.set_seed(self.session.name)
        self.rv.clear_signals()
        self.rv.resize_signals_to_n()
        self.rv.is_array_mode = True

        hashstr = self.rv.compute_cache_hash([
            '**/rendervars.py',
            '**/renderer.py',
            '**/std.py',
        ])

        cache_path = self.rv.get_cache_path(hashstr)
        log.info(f"cache path: {cache_path}")
        if not self.rv.load_cache(cache_path):
            std.bake_media(demucs=self.script_bool(SCRIPT_BOOL_DEMUCS, True))
            if self.script_has('bake'):
                self.script_emit('bake')
            elif self.script_has('start'):
                log.info("script.start is a deprecated name, please rename to bake (same functionality)")
                self.script_emit('start')
            self.rv.is_array_mode = False

            if cache_path:
                self.rv.save_cache(cache_path)

        self.check_render_complete()

    @trace_decorator
    def _invoke_frame(self):
        rv = self.rv

        self.script_emit('frame_before')
        self.setup_prompt()
        # std.bake_media(demucs=self.script_bool(SCRIPT_BOOL_DEMUCS, True))

        # TODO this seems to be handled in rv.init_frame already.
        if self.is_run_complete():
            self.mode = RenderMode.PAUSE
            log.info(f"Render complete, no data left. (f={rv.f} n={rv.n})")
            return

        with tracer('invoke_frame.invocation'):
            self._has_frame_error = not self.script_emit('frame', failsleep=0.25)

    def setup_prompt(self, tick=True):
        rv = self.rv

        if not self._is_dev and rv.n > 0:
            if 'promptneg' in self._script.__dict__:
                rv.promptneg = self._script.__dict__['promptneg']
            if 'prompt' in self._script.__dict__:
                self._invoke_safe(rv.set_prompt, **self._script.__dict__)

        if rv.prompt_node is not None and not self._is_dev:
            from src.party.ravelang import pnodes
            log.info(f'evaluating rv.prompt through pnode (get_text_at_time(t={rv.t}))')
            rv.prompt = pnodes.eval_text(rv.prompt_node, rv.t)

            if tick and self.script_bool(SCRIPT_BOOL_PROMPT_REALITME):
                log.info('ticking prompt root.')
                rv.prompt_node.tick(rv.t, rv.dt)


    # endregion

    # region Core functionality


    def set_globals(self, *, on_script, on_libs):
        """
        Set the globals on the script module and various library modules.
        TODO these modules should be definable somewhere and easily extendable
        @param on_script: Update the script module?
        @param on_libs:  Update the library modules?
        """
        # from scripts.interfaces import comfy_lib
        from src import RenderGlobals
        from src.party import maths
        from src.party.ravelang import pnodes

        if on_script and self._script:
            self._script.rv = self.rv
            self._script.s = self.session
            self._script.ses = self.session
            self._script.session = self.session
            self._script.f = self.rv.f
            self._script.dt = self.rv.dt
            self._script.fps = self.rv.fps
            self._script.hud = self.hud

        if on_libs:
            def set(mod: types.ModuleType | object):
                mod.renderer = self
                mod.rv = self.rv
                mod.session = self.session
                mod.f = self.rv.f
                mod.dt = self.rv.dt
                mod.fps = self.rv.fps
                mod.hud = self.hud

            from scripts.interfaces import diffusers_turbo
            from src.adapters import discomfy_client
            from src.party import audio, std, tricks, tricks_flow, tricks_util

            set(tricks)
            set(tricks_util)
            set(tricks_flow)
            set(maths)
            set(pnodes)
            set(std)
            set(audio)
            set(self.hud)
            set(diffusers_turbo)
            set(RenderGlobals)
            set(discomfy_client)
            pass

    def check_playback_complete(self):
        tickstate = self.tickstate

        if not tickstate.reached_f_last and not tickstate.catchedup:
            return

        if self.playback.looping:  # -> LOOPING
            self.seek(self.playback.loop_frame)
            self.mode = RenderMode.PLAY
        else:  # -> AUTOSTOP
            self.playback.until_f = None
            self.mode = RenderMode.PAUSE

    def check_render_complete(self):
        if not self.reached_n_max:
            return False

        if self._is_dev:
            self.seek(self.session.f_first)
            log.info("Catched up to max frame, looping.")
        else:
            self.mode = RenderMode.PAUSE
            self.seek(self.session.f_last)
            log.info(f"Finished rendering. (n={self.rv.n})")  # TODO OS notification
            self.tickstate.finished_render = True
        return True

    # @trace_decorator
    def tick(self):
        initial_mode = self.mode
        rv = self.rv
        stats = self.stats
        playback = self.playback
        tickstate = self.tickstate
        tickstate.__init__()

        # Script live reload detection
        # ----------------------------------------
        with tracer("renderiter.reload_script_check"):
            script_elapsed = time.time() - stats.last_script_check
            if self.requests.script_check \
                or -1 < self.detect_script_every < script_elapsed \
                or self.is_cli:
                self.requests.script_check = False
                self.apply_hotreload_if_changed()
                stats.last_script_check = time.time()

        # Flush render request
        # ----------------------------------------
        just_started_render = False
        if self.requests.render:
            self.mode = RenderMode.RENDER
            just_started_render = True
            self.requests.render = False
            stats.draw_start_time = time.time()
            stats.n_frames = 0

        # Flush pause request
        # ----------------------------------------
        if self.requests.pause != PauseRequest.NONE:
            if self.requests.pause == PauseRequest.TOGGLE:
                if self.mode == RenderMode.PLAY:
                    self.mode = RenderMode.PAUSE
                elif self.mode == RenderMode.PAUSE:
                    self.mode = RenderMode.PLAY
                    playback.from_f = self.rv.f
            elif self.requests.pause == PauseRequest.PAUSE:
                self.mode = RenderMode.PAUSE
            elif self.requests.pause == PauseRequest.PLAY:
                self.mode = RenderMode.PLAY

            self.requests.pause = PauseRequest.NONE

        # Flush seek requests
        # ----------------------------------------
        # TODO I don't remember why we do this, we should try direct seek again without requests
        sreq = self.requests.seek
        self.requests.seek = None
        if sreq:
            f_prev = self.rv.f
            img_prev = self.rv.img

            # TODO this used to be handled in the session class, watch out for potential bugs.
            self.rv.f = sreq.f_target
            if sreq.img_mode == SeekImageMode.NORMAL:
                self.rv.load_f(clamp_to_last=True)
                self.rv.load_f_img(self.rv.f, default='black')
            elif sreq.img_mode == SeekImageMode.IMAGE_ONLY:
                self.rv.load_f_img(f_prev, default='pink')
            elif sreq.img_mode == SeekImageMode.NO_IMAGE:
                self.rv.load_f(clamp_to_last=True)
                self.rv.img = img_prev

            # TODO we may want this, but it should require manual activation
            # if self.rv.f > self.session.f_last:
            # 	self.session.f_last = self.rv.f
            # 	self.session.f_last_path = self.session.f_path
            # elif self.rv.f < self.session.f_first:
            # 	self.session.f_first = self.rv.f
            # 	self.session.f_first_path = self.session.f_path

            if self.playback.audio is not None:
                self.playback.audio.seek(rv.t)

            tickstate.f_changed = True
            tickstate.just_seeked = True

        self.just_seeked = sreq

        # Playback accumulator
        # ----------------------------------------
        if not self.mode == RenderMode.PAUSE and not just_started_render:
            tickstate.dt = time.time() - (stats.draw_start_time or time.time())

            stats.draw_start_time = time.time()
            playback.dt_accumulated += tickstate.dt
            print(f"{playback.dt_accumulated} += {tickstate.dt}")
        else:
            stats.draw_start_time = None
            playback.dt_accumulated = 0
            tickstate.dt = 9999999

        # Fast-mode:
        # less than a second (>1 FPS) to render
        # ----------------------------------------
        tickstate.fast_mode = self.mode == RenderMode.PLAY and tickstate.dt < 1
        if tickstate.fast_mode and self.mode == RenderMode.RENDER:
            tickstate.fast_mode = self.render_repetition == RenderRepetitionMode.FOREVER

        # Tick playback
        # ----------------------------------------
        f_start = self.rv.f if self.session else 0

        if playback.audio is not None and playback.audio.playing:
            # Synchronize rv.f with the audio player's position
            from just_playback import Playback
            assert isinstance(playback.audio, Playback)
            time_s = playback.audio.curr_pos
            f = int(self.rv.fps * time_s)
            if f != self.rv.f:
                tickstate.f_changed = True
                self.rv.f = f
        else:
            while playback.dt_accumulated >= 1 / self.rv.fps:
                # print("FRAME += 1", self.rv.f, "->", self.rv.f + 1)
                print(f"{rv.f} += 1")
                self.rv.f += 1
                playback.dt_accumulated -= 1 / rv.fps
                tickstate.f_changed = True

                # Stop after one iteration since we are saving, we want
                # to render every frame without fail
                if self.should_save() or self.is_cli:
                    playback.dt_accumulated = 0
                    break

        # playback.dt_accumulated = 0

        # When there are no frames, snap to frame 1
        if self.rv.f_first == 0 and self.rv.f_last == 0:
            self.rv.f = 1

        # Apply new frame
        # ----------------------------------------
        if tickstate.f_changed:
            self.rv.load_f(img=not self.mode == RenderMode.RENDER)
            self.hud.update_draw_signals()

        tickstate.reached_f_last = not rv.f_exists and not tickstate.was_paused and self.rv.f >= self.session.f_last
        tickstate.catchedup = playback.until_f and self.rv.f >= playback.until_f > f_start
        if tickstate.f_changed and not tickstate.just_seeked or self.mode == RenderMode.RENDER:
            if (self.mode == RenderMode.PLAY and self.mode != RenderMode.RENDER):
                if not playback.allow_past_end:
                    self.check_playback_complete()

        tickstate.just_paused = initial_mode == RenderMode.PLAY and self.mode != RenderMode.PLAY
        tickstate.just_unpaused = initial_mode != RenderMode.PLAY and self.mode == RenderMode.PLAY

        if tickstate.just_paused:
            playback.allow_past_end = False

        # Draw the frame
        # ----------------------------------------
        if self.should_draw():
            if self.draw_callback:
                self.draw_callback(self.rv.f)
            else:
                self.draw()

            if self._has_frame_error or self.render_repetition == RenderRepetitionMode.ONCE:
                self.mode = RenderMode.PAUSE
        elif tickstate.f_changed:
            tickstate.dry_draw = (not self.session.storage.has_frame_data(rv.f, 'hud') and
                                  rv.f_exists and
                                  self.enable_dynamic_hud_populate)
            if tickstate.dry_draw:
                self.draw(dry=True)
        else:
            pass

        tickstate.was_paused = self.mode == RenderMode.PAUSE


    @trace_decorator
    def draw(self, f=None, dry=False):
        """
        Render a frame.
        """
        rv = self.rv

        self.ensure_init(InitLevel.LIGHT if self.is_dev else InitLevel.HEAVY)
        self.ensure_bake()
        # self.setup_init_media()
        if self.check_render_complete():
            log.error("Cannot draw, problem with script initialization!")
            return

        # Remove corrupted frames and walk back
        # with trace('corruption_check'):
        #     rv.start_frame(f, scalar, cleanse_nulls=False)
        #     while rv.img_f1 is None:
        #         print("Deleting corrupted frame", f)
        #         session.delete_f(f)
        #         f -= 1
        #         rv.start_frame(f, scalar, cleanse_nulls=False)

        self.state = RendererStep.RENDERING

        self.hud.clear()

        rv.init_frame(
            f=f or rv.f,
            init_res=self.script_get(SCRIPT_STR_INIT_RES, default="init"))
        rv.dry = dry
        rv.trace = 'frame'

        with tracer('printing'):
            # Print the header
            if rv.n > 0:
                ss = f'frame {rv.f} / {rv.n}'
            else:
                ss = f'frame {rv.f}'

            elapsed_str = str(self.stats.draw_duration_real)
            ss += f' :: {self.stats.n_rendered / rv.fps:.2f}s rendered in {elapsed_str} ----------------------'
            # log.info("")
            print("")
            log.info(ss)

        start_f = rv.f
        start_img = rv.img

        self.hud.clear()
        self.hud.snap('init', self.rv.init_img)
        self.hud.snap('renderer.draw: rv_img (start)', rv.img)
        self._invoke_frame()
        self.hud.snap('renderer.draw: rv_img (final)', rv.img)

        with tracer('post_render'):
            if self._has_frame_error or dry:
                # Restore the frame number
                self.seek(start_f)
                rv.img = start_img
            else:
                self.session.storage.set_frame_data(rv.f, 'hud', list(self.hud.rows))

                if self.should_save():
                    frame_path = self.session.write_frame(rv.f, rv.img)
                    self.session.storage.write()
                    if rv.f > self.session.f_last:
                        self.session.f_last = rv.f
                        self.session.f_last_path = frame_path
                # session.save_data()

                self.stats.last_rendered_f = rv.f

                if (self.enable_save_hud or self.is_cli) and not self._is_dev:
                    # self.hud.save(self.session, self.hud.to_pil(self.session))
                    if self.hud.has_snaps > 0:
                        path = self.session.res_path('.snapshot.png').as_posix()
                        # self.hud.save_tiled_snapshots(path)

                self.stats.n_rendered += 1

        # rv.unset_signals()
        rv.is_array_mode = False
        rv.load_signal_arrays = True

        self.state = RendererStep.READY
        if self._has_frame_error:
            self.requests.pause = True

        rv.is_custom_img = False


Renderer.null = Renderer(is_null=True)

# endregion


# region Event Hooks
# def on_audio_playback_start(t):
#     global request_pause
#
#     seek_t(t)
#     requests.request_pause = False
#
#
# def on_audio_playback_stop(t_start, t_end):
#     global request_pause
#
#     seek_t(t_start)
#     requests.request_pause = True


# audio.on_playback_start.append(on_audio_playback_start)
# audio.on_playback_stop.append(on_audio_playback_stop)
# endregion


# def reload_modules(self, exclude):
# 	modules_to_reload = []
# 	for x in sys.modules:
# 		for modpath in allowed_module_reload_names:
# 			if x.startswith(modpath) and modpath != exclude:
# 				modules_to_reload.append(x)
#
# 	for modpath in modules_to_reload:
# 		self.reload_module(modpath)
