import importlib
import logging
import math
import os
import re
import shutil
import typing
from collections.abc import Iterator
from datetime import timedelta
from pathlib import Path
from typing import Optional, Tuple
from urllib.parse import urlparse

from errloom.session_entry import SessionEntry
from errloom.lib.untested_error import UntestedError

# from src.classes import folder_paths
# from src.gui.session_entry import SessionEntry
# from src.utils.untested_error import UntestedError
# import userconf

if typing.TYPE_CHECKING:
    from errloom.session import Session

log = logging.getLogger("paths")

UNDEFINED = object()
THE_PATH = object()
Pathlike = typing.Union[str, Path]

root = Path(__file__).resolve().parent.parent  # TODO this isn't very robust
root_src = root / "errloom"
root_scripts = root / "scripts"  # Add scripts directory
session_timestamp_format = "%Y-%m-%d_%Hh%M"

userdir = Path("~/.errloom").expanduser().resolve()  # TODO maybe a better directory, or we make it configurable
usertmp = userdir / "tmp"
userconf = userdir / "userconf.py"
user_holowares = userdir / "hol"
user_sessions = userdir / "sessions"
user_inits = [user_sessions, user_sessions / "inits"]
user_tmp = userdir / usertmp  # Temporary files
user_logs = userdir / "logs"

# Root and subdirs
# ----------------------------------------

userdir.mkdir(parents=True, exist_ok=True)
usertmp.mkdir(parents=True, exist_ok=True)
userconf.mkdir(parents=True, exist_ok=True)
user_holowares.mkdir(parents=True, exist_ok=True)
user_sessions.mkdir(parents=True, exist_ok=True)
user_tmp.mkdir(parents=True, exist_ok=True)
user_logs.mkdir(parents=True, exist_ok=True)

# These suffixes will be stripped from the plugin IDs for simplicity
plugin_suffixes = ["_plugin"]

video_exts = [".mp4", ".mov", ".avi", ".mkv", ".webm"]
image_exts = [".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"]
audio_exts = [".wav", ".mp3", ".ogg", ".flac", ".aac", ".opus"]
text_exts = [".py", ".txt", ".md", ".json", ".xml", ".html", ".css", ".js", ".csv", ".tsv", ".yml", ".yaml", ".ini"]
audio_ext = ".flac"
audio_codec = "flac"
leadnum_zpad = 8

extra_session_paths = []
all_session_paths = [user_sessions, *extra_session_paths]

# Add comfyui to the path
# ----------------------------------------
# root_comfy = Path(userconf.comfy_root)
# root_comfy_nodes = root_comfy / "custom_nodes"
# if root_comfy.exists():
#     sys.path.append(root_comfy.as_posix())
# plugin_path = (root_comfy / src_plugins_name).as_posix()
# sys.path.append(plugin_path)


# Import ComfyUI model paths
# ----------------------------------------


# def get_model_path(type, ckpt_name, required=False):
#     """
#     Get a SD model full path from its name, searching all the defined model locations
#     """
#     ret = folder_paths.get_full_path(type, ckpt_name)
#     if ret is not None:
#         return ret
#
#     if required:
#         raise ValueError(f"Model {ckpt_name} not found")
#     return None
#
#
# def determine_model_path(type, ckpt_name):
#     root = folder_paths.folder_names_and_paths[type][0][0]
#     return Path(root) / ckpt_name
#
#
# folder_paths.load_base_model_paths(userconf.comfy_model_path_root, None)
# for yaml_path in userconf.comfy_model_path_configs:
#     folder_paths.load_extra_model_paths_config(yaml_path, print_path=False)

# Main API
# ----------------------------------------


def is_image(file):
    file = Path(file)
    return file.suffix in image_exts


def is_video(file):
    file = Path(file)
    return file.suffix in video_exts


def is_audio(file):
    file = Path(file)
    return file.suffix in audio_exts


def is_youtube_link(url):
    """
    This function tells you whether a link fits like a glove into youtube-dl or yt-dlp.
    What a shame it would be for such a link to not go -straight- into yt-dlp.
    """
    url = str(url)
    return "youtube.com" in url or "youtu.be" in url


def is_timestamp_range(string):
    # Match a timestamp range such as 00:33-00:45
    return re.match(r"^\d{1,2}:\d{2}-\d{1,2}:\d{2}$", string) is not None


def is_temp_extraction_path(path):
    return path.name.startswith(".")


def get_cache_file_path(filepath, cachename, ext=".npy"):
    filepath = Path(filepath)

    # If the file is in a hidden folder, we save the cache in the parent folder
    if filepath.parent.stem.startswith("."):
        filepath = filepath.parent.parent / filepath.name

    # If the data would be hidden, we unhide. Cached audio data should never be hidden!
    if filepath.name.startswith("."):
        filepath = filepath.with_name(filepath.name[1:])

    cachepath = filepath.with_stem(f"{Path(filepath).stem}_{cachename}").with_suffix(
        ext
    )
    return cachepath


def as_string(path: Pathlike):
    if isinstance(path, str):
        return path
    elif isinstance(path, Path):
        return path.as_posix()
    else:
        raise "Unknown path type!"


def get_timestamp_range(string):
    """
    Match a timestamp range such as 00:33-00:45 and return min/max seconds (two values)
    """
    if is_timestamp_range(string):
        lo, hi = string.split("-")
        lo = parse_time_to_seconds(lo)
        hi = parse_time_to_seconds(hi)
        return lo, hi
    else:
        raise ValueError(f"Invalid timestamp range: {string}")


def res_path(resname: Path | str) -> Path | None:
    return res(resname, extensions=image_exts, default=THE_PATH, root=user_inits)


def res(
    resname: Path | str,
    extensions: str | list | tuple | None = None,
    *,
    default=None,
    hide=False,
    root=None,
) -> Path | None:
    """
    Get a session resource, e.g. init video
    - If subpath is absolute, returns it
    - If subpath is relative, tries to find it in the session dir
    - If not found, tries to find it in the parent dir
    - If not found, check userconf.init_paths and copy it to the session dir if it exists
    - If not found, returns None
    - Tries all extensions passed

    args:
        resname: The name of the resource
        extensions: A single extension or a list of extensions to try, e.g. ['mp4', 'avi']
        return_missing: If True, returns the path to the missing file instead of None
    """
    resname = Path(resname)
    if hide:
        resname = resname.with_name(f".{resname.name}")

    def try_path(path):
        if path.exists():
            return path
        elif path.with_name(f".{path.name}").exists():
            return path.with_name(f".{path.name}")
        return None

    def try_ext(path, ext):
        # Remove dot from extension
        if ext:
            if ext[0] == ".":
                ext = ext[1:]
            path = path.with_suffix("." + ext)

        # Test
        userconf_mod = importlib.import_module("userconf")
        checkpaths = [root / path, user_sessions / path]
        checkpaths.extend([p / path for p in userconf_mod.init_paths])
        for i, p in enumerate(checkpaths):
            p = try_path(p)
            if p:
                # Copy to session dir
                src = p
                dst = root / path
                cp(src, dst, overwrite=True)
                return dst

        return checkpaths[0] if default == THE_PATH else default

    if resname.is_absolute() and resname.exists():
        return resname
    elif isinstance(extensions, (list, tuple)):
        # Try each extension until we find one that exists
        for e in extensions:
            p = try_ext(resname, e)
            if p and p.exists():
                return p
        return p if default == THE_PATH else default
    else:
        return try_ext(resname, extensions)


def parse_time_to_seconds(time_str):
    """Convert time string to total seconds without using exceptions.
    Supports formats: HH:MM:SS.fff, HH:MM:SS, MM:SS.fff, MM:SS, S.fff, S"""

    # Define format patterns and extract functions
    formats = [
        (
            "%H:%M:%S.%f",
            lambda m: (int(m[0]), int(m[1]), int(m[2]), float(f"0.{m[3]}")),
        ),
        ("%H:%M:%S", lambda m: (int(m[0]), int(m[1]), int(m[2]), 0)),
        ("%M:%S.%f", lambda m: (0, int(m[0]), int(m[1]), float(f"0.{m[2]}"))),
        ("%M:%S", lambda m: (0, int(m[0]), int(m[1]), 0)),
        ("%S.%f", lambda m: (0, 0, int(m[0]), float(f"0.{m[1]}"))),
        ("%S", lambda m: (0, 0, int(m[0]), 0)),
    ]

    # Try each format pattern
    for fmt, extract in formats:
        # Convert format to regex pattern
        pattern = (
            fmt.replace("%H", r"(\d{1,2})")
            .replace("%M", r"(\d{1,2})")
            .replace("%S", r"(\d{1,2})")
            .replace("%f", r"(\d{1,6})")
        )

        match = re.match(f"^{pattern}$", time_str)
        if match:
            hours, mins, secs, msecs = extract(match.groups())
            return timedelta(
                hours=hours, minutes=mins, seconds=secs, microseconds=int(msecs * 1e6)
            ).total_seconds()

    # return 0.0  # Return 0 if no format matches
    raise UntestedError()

def increment_path(path):
    match = re.search(r'(\d+)$', path.name)
    if match is not None:
        # name_1 -> name_2 -> name_3 etc until we find first available
        base_name = path.name[: match.start()] + path.name[match.end():]
        next_num = int(match.group(1)) + 1
        while True:
            name = base_name + str(next_num)
            path = user_sessions / name
            if not path.exists():
                break
            next_num += 1
    else:
        # name -> name_1
        name = path.name + '_1'
        path = user_sessions / name

    return path


def short_pid(pid):
    """
    Convert 'user/my-repository' to 'my_repository'
    """
    if isinstance(pid, Path):
        pid = pid.as_posix()

    if "/" in pid:
        pid = pid.split("/")[-1]

    # Strip suffixes
    for suffix in plugin_suffixes:
        if pid.endswith(suffix):
            pid = pid[: -len(suffix)]

    # Replace all dashes with underscores
    pid = pid.replace("-", "_")

    return pid


def split_jid(jid, allow_jobonly=False):
    """
    Split a plugin jid into a tuple of (plug, job)
    """
    if "." in jid:
        s = jid.split(".")
        return s[0], s[1]

    if allow_jobonly:
        return None, jid

    raise ValueError(f"Invalid plugin jid: {jid}")


def get_frameranges(frames: str | None) -> Iterator[str]:
    if frames is None:
        return

    ranges = frames.split(":")
    yield from ranges


def parse_frame_string(frames, name="none", min=None, max=None):
    """
    parse_frames('1:5', 'example') -> ('example_1_5', 1, 5)
    parse_frames(':5', 'example') -> ('example_5', None, 5)
    parse_frames('3:', 'example') -> ('example_3', 3, None)
    parse_frames(None, 'banner') -> ("banner", None, None)
    """
    if isinstance(frames, tuple):
        return *frames[0:2], name
    if isinstance(frames, int):
        return frames, frames, f'{name}_f{frames}'

    if frames is not None:
        sides = frames.split(":")
        lo = sides[0]
        hi = sides[1]
        if frames.endswith(":"):
            lo = int(lo)
            return lo, max, f'{name}_{lo}_{max or "max"}'
        elif frames.startswith(":"):
            hi = int(hi)
            return min, hi, f"{name}_0_{hi}"
        else:
            lo = int(lo)
            hi = int(hi)
            return lo, hi, f"{name}_{lo}_{hi}"
    else:
        return None, None, name


# region Leadnums
def get_leadnum_zpad(path=None):
    """
    Find the amount of leading zeroes for the 'leading numbers' in the directory names and return it
    e.g.:
    0003123 -> 7
    00023 -> 5
    023 -> 3
    23 -> 2
    23_session -> 2
    48123_session -> 5
    """
    biggest = 0
    smallest = math.inf

    for file in Path(path).iterdir():
        if file.suffix in image_exts:
            match = re.match(r"^(\d+)\.", file.name)
            if match is not None:
                num = match.group(1)
                size = len(num)
                # print(size, smallest, biggest, file)
                biggest = max(biggest, size)
                smallest = min(smallest, size)

    if smallest != biggest:
        return smallest
    return biggest


def is_leadnum_zpadded(path=None):
    return get_leadnum_zpad(path) >= 2


def get_next_leadnum(path=None):
    return (get_max_leadnum(path) or 0) + 1


def get_max_leadnum(path=None):
    lo, hi = get_leadnum(path)
    return hi


def get_min_leadnum(path=None):
    lo, hi = get_leadnum(path)
    return lo


def get_leadnum(path=None):
    """
    Find the largest 'leading number' in the directory names and return it
    e.g.:
    23_session
    24_session
    28_session
    23_session

    return value is 28
    """
    if isinstance(path, str):
        return find_leadnum(path)

    smallest = math.inf
    biggest = 0
    for parent, dirs, files in os.walk(path):
        for file in files:
            stem, suffix = os.path.splitext(file)
            if suffix in image_exts:
                # Only process stems that are purely numeric
                if stem.isdigit():
                    num = int(stem)
                    smallest = min(smallest, num)
                    biggest = max(biggest, num)
        break

    if biggest == 0 and smallest == math.inf:
        return None, None

    return smallest, biggest


def find_leadnum(path=None, name=None):
    if name is None:
        name = Path(path).name

    # Find the first number
    match = re.match(r"^(\d+)", name)
    if match is not None:
        return int(match.group(1))

    return None


# endregion


# region Utilities

def get_image_glob(path):
    path = Path(path)
    l = []
    # Add files using image_exts and os.walk
    for parent, dirs, files in os.walk(path):
        for file in files:
            if os.path.splitext(file)[1] in image_exts:
                l.append(os.path.join(parent, file))
    return l

# endregion


def get_first_match(path, suffix, name=None):
    path = Path(path)
    if not path.exists():
        return None

    for p in path.iterdir():
        if suffix is not None and p.suffix == suffix:
            return p
        if name is not None and p.name == name:
            return p
        pass

    return None


# region Scripts

def get_latest_session() -> Tuple[Optional["Session"], float]:
    """
    Return the most recently modified session directory (on the session.json file inside)
    If no session is found, return None
    """
    latest = None
    latest_mtime = 0

    baselevel = len(str(user_sessions).split(os.path.sep))
    # if curlevel <= baselevel + 1:
    #     [do stuff]

    for parent, dirs, files in os.walk(user_sessions):
        curlevel = len(str(parent).split(os.path.sep))
        if "session.json" in files and curlevel == baselevel + 1:
            mtime = os.path.getmtime(os.path.join(parent, "session.json"))
            if mtime > latest_mtime:
                latest_mtime = mtime
                latest = parent

    return latest, latest_mtime

def filepath_to_modpath(filepath, default=UNDEFINED) -> str:
    filepath = Path(filepath)
    if filepath.suffix == ".py":
        filepath = filepath.with_suffix("")

    if not filepath.is_relative_to(root):
        if default is not UNDEFINED:
            return default  # type: ignore
        raise ValueError(f"Invalid path: {filepath}")

    filepath = filepath.relative_to(root)
    # TODO check if root is in filepath

    # modpath = get_script_file_path(name)
    #
    # # Decide if the script is in the scripts folder or part of the session
    # if scripts_name in modpath.parts:
    #     return f'{scripts_name}.{modpath.relative_to(scripts).with_suffix("").as_posix().replace("/", ".")}'
    # elif sessions_name in modpath.parts:
    #     return f'{sessions_name}.{modpath.relative_to(sessions).with_suffix("").as_posix().replace("/", ".")}'

    ret = filepath.as_posix().replace("/", ".")
    if ret.endswith("."):
        ret = ret[:-1]
    if ret.startswith("."):
        ret = ret[1:]

    if ".." in ret:  # invalid path, inside a .folder
        raise ValueError(f"Invalid path: {filepath}")

    return ret

# endregion


def guess_suffix(suffixless_path):
    # Convert the input to a Path object
    path = Path(suffixless_path)

    if path.suffix != "":
        return path

    # Check if the suffixless path already exists
    if path.exists():
        return path

    # Get the directory and stem from the suffixless path
    directory = path.parent
    stem = path.stem

    # Iterate over the files in the directory
    for file_path in directory.iterdir():
        file_stem = file_path.stem

        # Check if the file has the same stem as the suffixless path
        if file_stem == stem:
            return file_path

    # No matching file found, return None
    # printerr(f"Couldn't guess extension: {suffixless_path}")
    return path


def rmtree(path):
    """
    Remove a directory and all its contents
    """
    path = Path(path)
    if path.exists():
        import shutil

        shutil.rmtree(path)


def mktree(path):
    path = Path(path)
    if path.suffix:
        path = path.parent
    path.mkdir(parents=True, exist_ok=True)


def rmclean(path):
    """
    Remove a directory and all its contents, and then recreate it clean
    """
    path = Path(path)
    if path.exists():
        import shutil

        shutil.rmtree(path)

    path.mkdir(parents=True, exist_ok=True)
    return path


def file_tqdm(path, start, target, process, desc="Processing"):
    from tqdm import tqdm
    import time

    tq = tqdm(total=target)
    tq.set_description(desc)
    last_num = start
    while process.poll() is None:
        cur_num = len(list(path.iterdir()))
        diff = cur_num - last_num
        if diff > 0:
            tq.update(diff)
            last_num = cur_num

        tq.refresh()
        time.sleep(1)

    tq.update(target - tq.n)  # Finish the bar
    tq.close()


def touch(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.touch(exist_ok=True)


def rm(path):
    path = Path(path)
    if path.exists():
        path.unlink()


def cp(path1, path2, overwrite=False):
    path1 = Path(path1)
    path2 = Path(path2)
    if path1.exists() and path1 != path2:
        if overwrite or not path2.exists():
            shutil.copy(path1, path2)
        else:
            raise FileExistsError(f'Cannot cp, file already exists and overwrite is false: {path2}')


def exists(dat):
    return Path(dat).exists()


def is_valid_url(url):
    raise UntestedError()
    result = urlparse(url)
    return bool(result.scheme and result.netloc)


def first_existing(*paths, default_value=None, default_index=None):
    """
    Get the first existing path from a list of paths.
    """
    for p in paths:
        if p and p.exists():
            return p

    if default_value is not None:
        return default_value
    elif default_index is not None:
        return paths[default_index]
    else:
        raise FileNotFoundError("paths.first_existing: None of the provided path exists and no default is provided!")


def unhide(flowdir):
    flowdir = Path(flowdir)
    if flowdir.name.startswith("."):
        flowdir = flowdir.with_name(flowdir.name[1:])
    return flowdir


def hide(path: Pathlike):
    path = Path(path)
    if not path.name.startswith("."):
        path = path.with_name(f".{path.name}")
    return path


def seconds_to_ffmpeg_timecode(param):
    # Convert absolute seconds (e.g. 342) to ffmpeg timecode (e.g. 00:05:35.00) string
    return str(timedelta(seconds=param))


def is_session(path: Path | str):
    """
    Check if a path is estimated to have been in a discore session, using simple heuristics.
    @param path:
    @return:
    """
    path = Path(path)
    has_file = (
        (path / "storage.json").exists()
        or (path / "script.py").exists()
        or (path / "00000001.jpg").exists()
    )

    return path.exists() and path.is_dir() and has_file


def iter_sessions(update_cache=False):
    from errloom import storage

    if not update_cache:
        return storage.application.cached_session_paths_detected

    files = []

    # Explore root paths recursively and add folders that contain a script.py
    def explore(root):
        if not root.exists():
            return

        for p in root.iterdir():
            if p.is_dir() and not p.name.startswith("."):
                if p in storage.application.cached_session_paths_detected:
                    continue

                if is_session(p):
                    files.append(p)
                    print(f"Found session: {p}")
                explore(p)

    for path in all_session_paths:
        explore(Path(path))

    files.sort(key=lambda x: x.stat().st_mtime, reverse=False)
    files = [f for f in files if f.is_dir()]
    files = list(set(files))

    storage.application.cached_session_paths_detected = files
    return files


def get_session_entries() -> list[SessionEntry]:
    from errloom import storage

    dirs = []
    dirs.extend(
        str(p)
        for p in user_sessions.iterdir()
        if user_sessions.exists()
        and p.is_dir()
        and not p.name.startswith(".")
        and is_session(p)
    )
    dirs.extend(storage.application.recent_sessions)
    dirs = list(set(dirs))

    dirs = [s for s in dirs if Path(s).exists()]  # remove non-existent paths
    dirs = list(set(dirs))  # remove duplicates

    ret = [SessionEntry(s) for s in dirs]
    ret = sorted(ret, key=lambda x: x.last_modified, reverse=True)
    return ret


def is_init(name_or_path):
    path = Path(name_or_path)
    return path.name.startswith(".init") or path.name.startswith("init")


def sanitize_prompt(prompt: str) -> str:
    import re

    # Convert to lowercase
    sanitized = prompt.lower()

    # Replace spaces with underscores
    sanitized = sanitized.replace(" ", "_")

    # Remove any non-alphanumeric characters except underscores and hyphens
    sanitized = re.sub(r"[^a-z0-9_-]", "", sanitized)

    # Truncate to a reasonable length (e.g., 50 characters)
    sanitized = sanitized[:50]

    # Remove leading/trailing underscores or hyphens
    sanitized = sanitized.strip("_-")

    # If the sanitized string is empty, use a default name
    if not sanitized:
        sanitized = "untitled_prompt"

    return sanitized
