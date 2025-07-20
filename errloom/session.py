# CONTEXT
# ----------------------------------------
#
# This file is a WIP module for providing structured "session" directories for a run of any kind.
# It used to be a single Session class, but it is being broken up to facilitate its reuse in other projects.
# Ideally multiple functions should be broken up, but there's old code that we want to keep working... for now.

import logging
import math
import os
import re
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, List, NewType, Optional, Set, Tuple

import cv2
import numpy as np
from tqdm import tqdm

import errloom.argp
from errloom.storage import Storage
from errloom.utils.log import tracer
from src import convert, paths
from src.paths import (get_leadnum, get_leadnum_zpad, get_max_leadnum, get_min_leadnum, is_leadnum_zpadded, leadnum_zpad, parse_frame_string, Pathlike)


logger = logging.getLogger('session')

SessionPath = NewType('SessionPath', str)
ScriptPath = NewType('ScriptPath', Path)
FrameRange = NewType('FrameRange', str)

UNDEFINED = object()
FRAME_EXTRACTION_INFO_FILE_NAME = 'info.json'
DEFAULT_SUFFIX = '.jpg'


class DirMixin:
    """
    A wrapper around a directory which introduces additional standards and rules.
    """

    def __init__(self, dirpath: Pathlike):
        self.dirpath = Path(dirpath) if isinstance(dirpath, str) else dirpath
        self._total_megabytes = None

    def name(self):
        return self.dirpath.name

    def get_total_megabytes(self, update_cache=False):
        if self._total_megabytes is not None and not update_cache:
            return self._total_megabytes

        total_size = 0
        for dirpath, dirnames, filenames in os.walk(self.dirpath):
            for f in filenames:
                if Path(f).exists():
                    fp = os.path.join(dirpath, f)
                    total_size += os.path.getsize(fp)
        self._total_megabytes = total_size / (1024 * 1024)  # Convert to MB
        return self._total_megabytes

    def exists(self):
        return self.dirpath.exists()

    def rmtree(self):
        shutil.rmtree(self.dirpath.as_posix())


class ResdirMixin(DirMixin):
    """
    A directory which is an interface to get resources.
    """

    def __init__(self, dirpath: Pathlike):
        super().__init__(dirpath)

    def res(
        self,
        resname: Path | str,
        extensions: str | list | tuple | None = None,
        *,
        default=None,
        hide=False,
    ) -> Path | None:
        return paths.res(
            resname, extensions, default=default, hide=hide, root=self.dirpath
        )

    def res_hidden(
        self,
        resname: Path | str,
        extensions: str | list | tuple | None = None,
        *,
        default=None,
    ) -> Path | None:
        return paths.res(
            resname, extensions, default=default, hide=True, root=self.dirpath
        )

    def res_path(
        self,
        resname: Path | str,
        extensions: str | list | tuple | None = None,
        *,
        hide=False,
    ) -> Path:
        ret = paths.res(
            resname, extensions, default=paths.THE_PATH, hide=hide, root=self.dirpath
        )
        assert ret is not None
        return ret

    def res_path_hidden(
        self, resname: Path | str, extensions: str | list | tuple
    ) -> Path:
        ret = paths.res(
            resname, extensions, default=paths.THE_PATH, hide=True, root=self.dirpath
        )
        assert ret is not None
        return ret

    # @trace_decorator
    def res_cv2(self, subpath: Path | str, *, ext: str) -> np.ndarray | None:
        """
        Get a session resource, e.g. init video
        """
        if not isinstance(subpath, (Path, str)):
            return convert.load_cv2(subpath)

        path = self.res(subpath, extensions=ext or ('jpg', 'png'))
        if path:
            im = convert.as_cv2(path)
            # TODO an equivalent function should be created in RenderVars which provisdes the resize to match
            # if im is not None and self.img is not None:
            # 	im = convert.crop_or_pad(im, self.img.shape[1], self.img.shape[0], 'black')
            return im
        else:
            return None

    # @trace_decorator
    def res_frame(
        self,
        name: str,
        f: int,
        *,
        subdir: str = '',
        ext=None,
        loop: bool = False,
        fps: Optional[int] = None,
    ) -> Path | None:
        """
        Get a session resource path, and automatically fetch a frame from it.
        Usage:

        resid='video' # Get the current session frame from video.mp4
        resid=3 # Get frame 3 from the current session
        """

        def get_dir_frame(dirpath):
            nonlocal f
            json_path = dirpath / FRAME_EXTRACTION_INFO_FILE_NAME
            if json_path.exists():
                json = convert.load_json(json_path)
                if fps is not None and fps != json['fps']:
                    f = int(f * (json['fps'] / fps))

            if loop:
                l = list(dirpath.iterdir())
                f = f % len(l)

            framestr = str(f)
            suffix = next(
                (
                    iext
                    for iext in paths.image_exts
                    if (
                    dirpath / f'{framestr.zfill(paths.leadnum_zpad)}.{ext}'
                ).exists()
                ),
                'jpg',
            )
            framename = f'{framestr.zfill(paths.leadnum_zpad)}.{suffix}'
            return dirpath / framename

        if isinstance(name, int):
            assert isinstance(self, FrameSeqMixin)
            return self.det_frame_path(name)

        elif isinstance(name, str):
            if subdir:
                name = f'{name}/{subdir}'

            # If the resid is a string, parse it
            file = self.res_path(name, extensions=[*paths.video_exts, *paths.image_exts])

            # File exists and is not a video --> return directly / same behavior as res(...)
            if file.exists() and paths.is_image(file) and f is None:
                return file
            elif file.exists() and paths.is_video(file):
                # Iterate dir and find the matching file, regardless of the extension
                framedir = self.extract_frames(file)
                return get_dir_frame(framedir)
            else:
                respath = self.res_path(name, extensions=paths.image_exts)
                return get_dir_frame(respath)
        elif isinstance(name, Path) and name.is_dir():
            return get_dir_frame(name / subdir)
        else:
            respath = self.res_path(name)
            return get_dir_frame(respath)

    # @trace_decorator
    def res_frame_count(self, resname: str, *, fps=None):
        dirpath = self.extract_frames(resname)
        if dirpath is None:
            return 0
        else:
            json_path = dirpath / FRAME_EXTRACTION_INFO_FILE_NAME
            if not json_path.exists():
                # Determine from directory file count, which is slower
                if fps is not None:
                    raise ValueError(f'fCannot determine frame count from fps without json stats. (searched for {json_path})')

                return len(list(dirpath.iterdir()))
            else:
                # Determine from json data
                json = convert.load_json(json_path)
                if fps is not None and fps != json['fps']:
                    return int(json['frame_count'] * (json['fps'] / fps))
                else:
                    return json['frame_count']

    # @trace_decorator
    def res_frame_cv2(
        self,
        name: str,
        f: int | None = None,
        *,
        loop=False,
        fps=None,
        default=UNDEFINED,
    ):
        if fps is None and hasattr(self, 'fps'):
            fps = self.fps

        ret = None

        frame_path = self.res_frame(name, f, ext=paths.image_exts, loop=loop, fps=fps)
        if frame_path is not None and frame_path.exists():
            with tracer(f'session.kres_frame_cv2: imread'):
                path = frame_path.as_posix()
                exists = os.path.exists(path)
                if exists:
                    ret = cv2.imread(path)
                    if ret is not None:
                        ret = cv2.cvtColor(ret, cv2.COLOR_BGR2RGB)

        if ret is None:
            if default is UNDEFINED:
                raise FileNotFoundError(
                    f'Could not find frame {f} in {self.dirpath.name}'
                )
            elif default is None:
                return None
            else:
                return default

        return ret

    def res_frameiter(self, resname, description='Enumerating...', load=True):
        framedir = self.extract_frames(resname)
        input_files = list(sorted(framedir.glob('*.jpg')))
        n = len(input_files)
        for i in tqdm(range(1, n), desc=description):
            path = input_files[i]
            img = load_cv2(path) if paths.is_image(path) and load else None
            yield i, img, path

    def res_frameiter_pairs(self, resname, description='Enumerating ...', load=True):
        framedir = self.extract_frames(resname)
        input_files = list(sorted(framedir.glob('*.jpg')))
        n = len(input_files)
        for i in tqdm(range(1, n - 1), desc=description):
            path1 = input_files[i]
            path2 = input_files[i + 1]
            img1 = load_cv2(path1) if paths.is_image(path1) and load else None
            img2 = load_cv2(path2) if paths.is_image(path2) and load else None

            yield i, img1, img2, path1, path2

    def get_flow_compute_dirpath(self, resname):
        resdir = self.res_frame_dirpath(resname)
        flowdir = paths.unhide(resdir.parent / f'{resdir.name}.flow/')
        flowdir = paths.hide(flowdir)
        return flowdir

    def init_img_exists(self, name='init'):
        path = self.res_path(name, extensions=paths.image_exts)
        return path.exists()

    def res_frame_dirpath(self, name, frames: tuple | None = None, global_scope=False):
        src = self.res_path(name, extensions=paths.video_exts)
        if not src.exists():
            print(f'Session.extract_frames: Could not find video file {src.name}')
            return None

        assert isinstance(self, FrameSeqMixin)
        lo, hi, name = self.get_frame_range(frames, src.stem)

        if global_scope:
            return paths.res_path(name)
        else:
            dst = src.with_name(f'.{name}')

        return dst

    def res_frame_pattern(self, path):
        pattern = f'{path}/%0{paths.leadnum_zpad}d.jpg'
        return pattern

    # @trace_decorator
    def extract_frames(
        self,
        name,
        *,
        size=None,
        f_nth=1,
        f_range: tuple | None = None,
        overwrite=False,
        warn_existing=False,
        minsize=1024,
        fps=None,
    ) -> Path | str | None:
        # already exists?
        if Path(name).is_dir():
            return name

        logger.info("(session.extract_frames)")

        src = self.res(name, extensions=paths.video_exts)
        dst = self.res_frame_dirpath(name)
        if not dst:
            raise FileNotFoundError(
                f'Could not find video file {name} for frame extraction.'
            )

        if dst.exists():
            if not overwrite:
                if warn_existing:
                    logger.info(
                        f'Frame extraction already exists for {src.name} at {dst} skipping ...'
                    )
                return dst
            else:
                logger.info(
                    f'Frame extraction already exists for {src.name} at {dst} overwriting ...'
                )

        pattern = self.res_frame_pattern(dst)
        lo, hi, name = self.get_frame_range(f_range, src)
        if fps is None and hasattr(self, 'fps'):
            fps = self.fps

        vf = ''
        if f_range is not None:
            vf += f'select=between(t,{lo},{hi})'
        else:
            vf = f'select=not(mod(n\\,{f_nth}))'

        size = (minsize, minsize) if size is None else size
        vf, w, h = vf_rescale(vf, max(minsize, size[0]), max(minsize, size[1]), size[0], size[1])

        paths.rmclean(dst)
        corelib.shlexrun([
            'ffmpeg',
            '-i', f'{src}',
            '-vf', f'{vf}',
            '-q:v', '5',
            '-loglevel', 'error',
            '-stats',
            pattern,
        ])

        # Delete frames to reach the target fps
        orig_fps = get_video_fps(src)
        if fps is not None:
            # Delete frames to reach the target fps
            threshold = orig_fps / fps
            elapsed = 0

            for i, img, file in self.res_frameiter(dst, f'Timescaling the extracted frames to match target fps {fps}'):
                # Check if the frame should be deleted based on the threshold
                if elapsed >= threshold:
                    elapsed -= threshold
                else:
                    paths.rm(file)
                elapsed += 1

            Session(dst).make_sequential()
        else:
            fps = orig_fps

        frame_count = len(list(dst.iterdir()))
        convert.save_json({'fps': fps, 'w': w, 'h': h, 'frame_count': frame_count}, dst / 'info.json')
        return dst


class InitResMixin(ResdirMixin):
    """
    A standard for a resource directory which offers an init media.
    """

    def res_init(self, name=None):
        name = name or 'init'
        img = self.res(name, extensions=paths.image_exts)
        mus = self.res(name, extensions=paths.audio_exts)
        vid = self.res(name, extensions=paths.video_exts)
        return img, mus, vid

    def has_init_res(self, name=None):
        """Return true if we have any init media at all"""
        name = name or 'init'
        img = self.res(name, extensions=paths.image_exts)
        mus = self.res(name, extensions=paths.audio_exts)
        vid = self.res(name, extensions=paths.video_exts)
        return img or mus or vid

    def res_music(self, name=None, *, optional=True):
        """
        Get the music resource for this session, or specify by name and auto-detect extension.
        """

        def _res_music(n):
            n = n or 'music'
            if self.exists:
                return self.res(f'{n}', extensions=paths.audio_exts)

        if name:
            v = _res_music(name)
        else:
            v = _res_music('music') or _res_music('init')
        if not v and not optional:
            raise FileNotFoundError('Could not find music file in session directory')
        return v

    def extract_init(self, name='init', size=None):
        frame_path = self.extract_frames(name, size=size)
        music_path = self.extract_music(name)

        return frame_path, music_path

    def extract_music(self, src='init', overwrite=False, hide=False):
        input = self.res(src, extensions=paths.video_exts)
        if not input.exists():
            print(f'Session.extract_music: Could not find video file {input.name}')
            return

        output = self.res_path(f'{src}.wav', hide=hide)
        cmd = f'ffmpeg -i {input} -acodec pcm_s16le -ac 1 -ar 44100 {output}'
        if output.exists():
            if not overwrite:
                print(f'Music extraction already exists for {input.name}, skipping ...')
                return
            paths.rm(input)

        os.system(cmd)

    def download_init(self, init, width=-1, height=-1):
        # TODO  the bulk of this function should be moved elsewhere, shouldn't be bound to a session
        class InitEntry:
            def __init__(self, path, time_start, time_end):
                self.fullpath = Path(path)
                self.time_start = time_start
                self.time_end = time_end

        # Split on ; and \n and remove empty strings
        init_tokens = re.split(r'[;\n]', init)
        init_tokens = [t for t in init_tokens if t]
        video_inputs = []
        audio_inputs = []
        time_start, time_end = None, None  # in seconds
        for init_token in init_tokens:
            init_token = init_token.strip()
            init_token = init_token.replace('\\', '/')

            # If matches a timestamp, set time_start and time_end
            if paths.is_timestamp_range(init_token):
                time_start, time_end = paths.get_timestamp_range(init_token)
                continue

            if paths.is_youtube_link(init_token):
                dldir = paths.user_tmp
                files_before = list(dldir.glob('*'))

                # Use yt-dlp to download the video
                ytdlp_args = []
                if time_start is not None and time_end is not None:
                    ytdlp_args.extend([
                        '--download-sections',
                        f'*{int(time_start)}-{int(time_end)}',
                    ])

                # Prefer 480p
                ytdlp_args.extend([
                    '-f',
                    'bestvideo[height<=480]+bestaudio/best[height<=480]',
                ])

                # if w is not None and h is not None:
                #     ytdlp_args.extend(['--postprocessor-args', f'-vf scale=-1:512'])

                subprocess.run(
                    [
                        'yt-dlp',
                        '-o',
                        f'{dldir}/%(title)s.%(ext)s',
                        *ytdlp_args,
                        init_token,
                    ],
                    check=True,
                )

                # Find the downloaded file (this is a bit hacky)
                files_after = list(dldir.glob('*'))
                new_files = [f for f in files_after if f not in files_before]

                if len(new_files) == 0:
                    raise Exception(
                        'There was a problem with downloading a YouTube video!'
                    )
                elif len(new_files) == 1:
                    # Add to videos
                    video_inputs.append(InitEntry(new_files[0], time_start, time_end))
                else:
                    raise Exception(
                        'yt-dlp downloaded more than one file... is that really possible?'
                    )

                break

            if not Path(init_token).is_absolute():
                for p in paths.user_inits:
                    fullpath = p / init_token
                    fullpath = paths.guess_suffix(fullpath)
                    if fullpath.exists():
                        init_token = fullpath
                        break

            if Path(init_token).suffix == '':
                init_token = paths.guess_suffix(init_token)
                if not init_token:
                    pass
            # return

            if not Path(init_token).exists():
                raise f"Couldn't find an init media: {init_token}"

            if paths.is_video(init_token):
                video_inputs.append(InitEntry(init_token, time_start, time_end))
            elif paths.is_audio(init_token):
                audio_inputs.append(InitEntry(init_token, time_start, time_end))
            else:
                raise Exception('Invalid init token')

            time_start, time_end = None, None

        # Now we will concatenate the videos and audios, or copy the video if there is only one
        has_custom_audio = len(audio_inputs) > 0

        # Video concatenation
        # ----------------------------------------
        init_video = None
        init_audio = None
        video_concat_path = f'{paths.user_tmp}/init.mkv'
        audio_concat_path = f'{paths.user_tmp}/audio.ogg'
        audio_concat_codec = 'libvorbis'
        if len(video_inputs) == 1:
            init_video = video_inputs[0]
        elif len(video_inputs) > 1:
            # Inputs
            # ----------------------------------------
            input_args = []
            for i, video in enumerate(video_inputs):
                if video.time_start:
                    input_args.extend([
                        '-ss',
                        paths.seconds_to_ffmpeg_timecode(video.time_start or 0),
                    ])
                if video.time_end:
                    input_args.extend([
                        '-to',
                        paths.seconds_to_ffmpeg_timecode(video.time_end or 0),
                    ])
                input_args.extend(['-i', video.fullpath.as_posix()])
            # input_args.extend(['-i', video.path])

            # Mapping
            # ----------------------------------------
            mapping_args = []
            for i, video in enumerate(video_inputs):
                mapping_args.append(f'[{i}:v]')
                if not has_custom_audio:
                    mapping_args.append(f'[{i}:a]')

            concat_opt = f'concat=n={len(video_inputs)}:v=1:a={0 if has_custom_audio == 1 else 1} [v]'
            if not has_custom_audio:
                concat_opt += ' [a]'
            mapping_args.extend([concat_opt])
            mapping_args = ['-filter_complex', ' '.join(mapping_args)]

            mapping_args.extend(['-map', '[v]'])
            if not has_custom_audio:
                mapping_args.extend(['-map', '[a]'])

            ffmpeg_args = [
                'ffmpeg',
                '-y',
                *input_args,
                *mapping_args,
                '-c:v',
                'libx264',
                video_concat_path,
            ]
            ffmpeg_args.extend(['-vf', f'scale={width or -1}:{height or -1}'])
            subprocess.run(ffmpeg_args, check=True)
            init_video = InitEntry(Path(video_concat_path), None, None)

        # Audio concatenation
        # ----------------------------------------
        if len(audio_inputs) > 0:
            # Inputs
            # ----------------------------------------
            input_args = []
            for i, i_audio in enumerate(audio_inputs):
                if i_audio.time_start:
                    input_args.extend([
                        '-ss',
                        paths.seconds_to_ffmpeg_timecode(i_audio.time_start or 0),
                    ])
                if i_audio.time_end:
                    input_args.extend([
                        '-to',
                        paths.seconds_to_ffmpeg_timecode(i_audio.time_end or 0),
                    ])
                input_args.extend(['-i', i_audio.fullpath.as_posix()])
            # input_args.extend(['-i', video.path])

            # Mapping
            # ----------------------------------------
            filter_string = ''
            for i, video in enumerate(audio_inputs):
                filter_string += f'[{i}:a]'
            filter_string += f'concat=n={len(audio_inputs)}:v=0:a=1 [a]'

            ffmpeg_args = [
                'ffmpeg',
                '-y',
                *input_args,
                '-filter_complex',
                filter_string,
                '-map',
                '[a]',
                '-c:a',
                audio_concat_codec,
                '-b:a',
                '320k',
                audio_concat_path,
            ]
            print(' '.join(ffmpeg_args))
            subprocess.run(ffmpeg_args, check=True)
            init_audio = InitEntry(Path(audio_concat_path), None, None)
            ffmpeg_args = []

        # Flatten audio & video into one file
        # ----------------------------------------
        # Now it's time to combine the video and audio
        init_final = (init_video or init_audio).fullpath.as_posix()
        if init_video and init_audio:
            init_path = f'{paths.user_tmp}/init.mp4'
            ffmpeg_args = [
                'ffmpeg',
                '-y',
                '-i',
                init_video,
                '-i',
                init_audio,
                '-c',
                'copy',
                init_path,
            ]
            subprocess.run(ffmpeg_args, check=True)
            init_final = init_path
        # And finally, we copy to the session dir
        if init_final:
            paths.cp(init_final, self.dirpath / f'init{Path(init_final).suffix}')
            # Delete a folder by the name of .init if it exists
            paths.rmtree(self.dirpath / '.init')

        else:
            raise Exception('Init could not be assembled!')


class ScriptResMixin(ResdirMixin):
    def res_script(self, name='script', touch=False) -> Path:
        """
        Get the script resource for this session, or specify by name and auto-detect extension.
        """
        name = name or 'script'
        if not name.endswith('.py'):
            name += '.py'

        path = self.dirpath / name
        if touch:
            paths.touch(path)

        return path

    @property
    def script_backup_path(self):
        return self.dirpath / 'scripts'

    def save_script_backup(self, f=None, compare_checksum=True):
        """
        Save a backup of the script file
        """
        if f is None and hasattr(self, 'f_last'):
            f = self.f_last

        assert f

        script_dir = self.script_backup_path
        script_path = self.res_script()

        paths.mktree(script_dir)

        # Get the checksum of the current script and the last one in the .scripts dir

        if compare_checksum:
            all_backups = list(script_dir.iterdir())
            all_backups = [int(p.stem) for p in all_backups if p.suffix == '.py']
            all_backups.sort()
            if all_backups:
                last_backup = all_backups[-1]
                last_backup_path = script_dir / str(last_backup)
                last_backup_path = last_backup_path.with_suffix('.py')
                import hashlib

                checksum_current = hashlib.md5(script_path.read_bytes()).hexdigest()
                checksum_last = hashlib.md5(last_backup_path.read_bytes()).hexdigest()
                if checksum_current == checksum_last:
                    return

        # Save the script
        script_path = script_dir / str(f)
        script_path = script_path.with_suffix('.py')
        paths.cp(self.res_script(), script_path, True)


class SessionStorage(Storage):
    pass

class SessionMixin(DirMixin):
    @property
    def storagepath(self) -> Path:
        return paths.first_existing(
            self.dirpath / 'storage.json',
            self.dirpath / 'session.json',
            default_index=0
        )

class Session(SessionMixin):
    """
    An Errloom session.
    """

    null: 'Session'


    def __init__(self, name_or_abspath: Pathlike, load=True, fixpad=False, logging=True):
        super().__init__(name_or_abspath)

        self.is_null: bool = False
        self.name: str
        self.dirpath: Path
        self.storage: SessionStorage

        if Path(name_or_abspath).is_absolute():
            self.name = Path(name_or_abspath).stem
            self.dirpath = Path(name_or_abspath)
        elif name_or_abspath is not None:
            self.name = Path(name_or_abspath).stem
            self.dirpath = paths.user_sessions / name_or_abspath

        self.dirpath.mkdir(exist_ok=True, parents=True)
        self.storage = SessionStorage(_path=self.storagepath.as_posix())

        # Proceed with loading the session
        if self.dirpath.exists():
            if load:
                self.load(logging=logging)
        else:
            if logging:
                logger.info(f'New session: {self.name}')

        # Fix the zero-padding on the session directory
        if fixpad:
            if self.dirpath.is_dir() and not is_leadnum_zpadded(self.dirpath):
                logger.info('Session directory is not zero-padded. Migrating...')
                self.make_zpad(leadnum_zpad)

    @staticmethod
    def Now(prefix='', log=True):
        """
        Returns: A new session which is timestamped to now
        """
        name = datetime.now().strftime(paths.session_timestamp_format)
        return Session(f'{prefix}{name}', logging=log)

    @staticmethod
    def RecentOrNow(recent_window=math.inf):
        """
        Returns: The most recent session, or a new session if none exists.
        args:
            recent_window: The number of seconds to consider a session recent. Outside of this window, a new session is created.
        """
        if any(paths.user_sessions.iterdir()):
            latest = max(paths.user_sessions.iterdir(), key=lambda p: p.stat().st_mtime)
            # If the latest session fits into the recent window, use it
            if (
                datetime.now() - datetime.fromtimestamp(latest.stat().st_mtime)
            ).total_seconds() < recent_window:
                return Session(latest)

        return Session.Now()

    @staticmethod
    def Create(
        name_or_path,
        init=None,
        copy_scripts: Path | str = None,
        copy_from=None,
        copy_frame=None,
        increment_name=False,
    ) -> 'Session':
        """Create a new session, optionally copying from another session.

        Args:
            name_or_path: Name or full path for the new session. If just a name, will be created under sessions/
            init: Optional init file to copy
            copy_scripts: Optional script file to copy
            copy_frame: Optional frame range to copy
            copy_from: Optional source session to copy assets from
            @param increment: increment a _x suffix by one.

        Returns:
            The newly created Session
        """
        name_symlist = ['.demucs', '.init']  # paths to create symlinks for
        name_blacklist = [
            'video.', 'video__', 'session', 'stow',
            '.scripts', 'scripts', 'frames',
        ]  # paths to skip
        ext_whitelist = ['.py', '.json', '.npy', *paths.image_exts, *paths.video_exts, *paths.audio_exts]

        path = Path(name_or_path)

        if increment_name:
            path = paths.increment_path(path)

        out = Session(path, load=False)

        if copy_from is not None:
            assert copy_from.dirpath.exists()

            # Create session directory
            paths.mktree(out.dirpath)

            # Copy allowed files/directories
            for copy_file in copy_from.dirpath.iterdir():
                # Skip blacklisted files
                if any(copy_file.name.startswith(b) for b in name_blacklist):
                    print(f'- Skipped {copy_file.name} due to prefix')
                    continue

                # Create symlinks for specified directories
                if copy_file.name in name_symlist:
                    (out.dirpath / copy_file.name).symlink_to(
                        f'../{copy_from.dirpath.name}/{copy_file.name}'
                    )
                    print(f'- Symlinked {copy_file.name} to {copy_from.dirpath.name}')
                    continue

                # Skip non-whitelisted suffixes
                if not copy_file.suffix in ext_whitelist:
                    print(f'- Skipped {copy_file.name} due to suffix')
                    continue

                # Skip init files if init arg provided
                if init and (paths.is_init(copy_file.stem) or copy_file.suffix == '.npy'):
                    print(f'- Skipped {copy_file.name} due to suffix')
                    continue

                # Skip numbered frames (outdated, now handled by blacklisted frames directory)
                try:
                    int(copy_file.stem)
                    continue
                except ValueError:
                    pass

                # Only copy whitelisted extensions
                if copy_file.is_file():
                    paths.cp(copy_file, out.dirpath / copy_file.name)
                    print(f'- Copied file {copy_file.name} to {out.dirpath.name}')
                elif copy_file.is_dir() and not paths.is_temp_extraction_path(copy_file):
                    shutil.copytree(copy_file, out.dirpath / copy_file.name)
                    print(f'- Copied directory {copy_file.name} from {copy_from.name}')

                # Handle init file
                if init:
                    # Remove any existing init/music files
                    for copy_file in ['music', 'init']:
                        existing = out.res(copy_file)
                        if existing:
                            paths.rm(existing)

        if init:
            out.download_init(init)

        # Immediately write the storage to clearly announce this folder as a discore session
        out.storage.write()

        return out

    @staticmethod
    def Unloaded(path):
        return Session(path, load=False)

    @staticmethod
    def Null() -> 'Session':
        ret = Session(is_null=True)
        return ret

    def Child(self, name) -> 'Session':
        if name:
            return Session(self.res(name))
        else:
            return self

    def __str__(self):
        return f'Session({self.name} ({self.dirpath})'


def concat(s1, s2):
    if s1:
        s1 += ','
    return s1 + s2


def vf_rescale(vf, w, h, ow, oh, crop=False):
    """
    Rescale by keeping aspect ratio.
    Args:
        vf:
        h: Target height (or none) can be a percent str or an int
        w: Target width (or none) can be a percent str or an int
        ow: original width (for AR calculation)
        oh: original height (for AR calculation)

    Returns:

    """
    # if isinstance(h, str):
    #     h = int(h.replace('%', '')) / 100
    #     h = f'ih*{str(h)}'
    # if isinstance(w, str):
    #     w = int(w.replace('%', '')) / 100
    #     w = f'iw*{str(w)}'
    # if w is None:
    #     w = f'-1'
    # if h is None:
    #     h = f'-1'

    w = w or ow
    h = h or oh

    if not w:
        w = '-1'
    if not h:
        h = '-1'

    if not crop:
        if w > h:
            vf = concat(vf, f'scale={w}:-1')
        elif h > w:
            vf = concat(vf, f'scale=-1:{h}')
        else:
            vf = concat(vf, f'scale={w}:{h}')
    else:
        # vf = concat(vf, f"scale={w}:-1,pad={w}:{h}")
        ar = w / h
        vf = concat(vf, f'crop=iw*{ar}:ih')

    return vf, w, h


def vf_fade(vf, fade_in, fade_out, frames, fps):
    duration = frames / fps
    if fade_in < duration:
        vf = concat(vf, f'fade=in:st=0:d={fade_in}')
    if fade_out < duration:
        vf = concat(vf, f'fade=out:st={frames / fps - fade_out}:d={fade_out}')

    return vf


def get_video_fps(video_path):
    command = [
        'ffprobe',
        '-v', 'error',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=avg_frame_rate',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        video_path,
    ]

    result = subprocess.run(command, capture_output=True, text=True)
    fps = result.stdout.strip()

    if '/' in fps:
        fps = fps.split('/')
        fps = float(fps[0]) / float(fps[1])

    fps = int(fps)
    return fps


class NullSession(Session):
    def __init__(self):
        self.is_null = True


Session.null = NullSession()

# @property
# def img(self):
# 	return self._img
#
# @img.setter
# def img(self, value):
# 	self._img = convert.load_cv2(value)

# def resize(self, w, h, crop=False):
# 	if self.img is None:
# 		return
#
# 	self.img = convert.resize(self.img, w, h, crop)

# @property
# def w(self):
# 	if self.img is None:
# 		return 0
# 	return self.img.shape[1]
#
# @property
# def h(self):
# 	if self.img is None:
# 		return 0
# 	return self.img.shape[0]

# def det_current_frame_path(self, subdir=''):
# 	return self.det_frame_path(self.f, subdir)
#
# def det_current_frame_exists(self):
# 	return self.det_current_frame_path().is_file()
#
