import logging
import typing
from typing import Optional, List, Tuple

import cv2
import numpy as np
from yachalk import chalk

from src.lib import loglib
from src.lib.loglib import printkw
from errloom.utils import convert
from errloom.utils.convert import save_png

log = logging.getLogger("HUD")

rows_tmp = []

if typing.TYPE_CHECKING:
    from errloom.utils.signal_block import RenderVars


class HUD:
    null: "HUD"

    def __init__(self, enable_logs):
        self.snaps: List[Tuple[str, np.ndarray]] = []
        self.rows: List[Tuple[str, Tuple[int, int, int]]] = []
        self.draw_signals: List["HUD.DrawSignal"] = []
        self.rv: Optional["RenderVars"] = None
        self.enable_logs = enable_logs
        self.name_stack: List[str] = []

    class DrawSignal:
        def __init__(self, name):
            self.name = name
            self.min = 0
            self.max = 0
            self.valid = False

    @property
    def has_snaps(self):
        return len(self.snaps) > 0

    def update_draw_signals(self):
        """
        Update the min and max of all signals
        """
        for s in self.draw_signals:
            signal = self.rv.current_signals.get(s.name)
            if signal is not None:
                s.min = signal.array.min()
                s.max = signal.array.max()
                s.valid = True
            else:
                s.valid = False

    def set_draw_signals(self, *names):
        """
        Set the signals to draw
        """
        self.draw_signals.clear()
        for name in names:
            self.draw_signals.append(self.DrawSignal(name))

    def snap(self, name: str, img: np.ndarray | None):
        if not isinstance(name, str):
            raise ValueError("Name must be a string")
        if img is None:
            return

        # Create full hierarchical name
        full_name = " > ".join(self.name_stack + [name]) if self.name_stack else name

        img = convert.as_cv2(img)
        if img is None:
            img = self.rv.img.copy()
        self.snaps.append((full_name, img))

    def __call__(self, *args, tcolor=(255, 255, 255), **kwargs):
        # Turn args and kwargs into a string like 'a1 a2 x=1 y=2'
        # Format numbers to 3 decimal places (if they are number)
        s = ""
        for a in args:
            s += loglib.value_to_print_str(a)
            s += " "

        # TODO auto-snap if kwargs is ndarray hwc

        for k, v in kwargs.items():
            s += f"{loglib.value_to_print_str(k)}="
            s += loglib.value_to_print_str(v)
            s += " "

        maxlen = 9999
        s = "\n".join([s[i : i + maxlen] for i in range(0, len(s), maxlen)])

        if self.enable_logs:
            printkw(**kwargs, chalk=chalk.magenta, fn_print=log)
        self.rows.append((s, tcolor))

    def clear(self):
        """
        Clear the HUD
        """
        self.snaps.clear()
        self.rows.clear()
        self.name_stack.clear()

    def save(self, session, hud):
        save_png(
            hud,
            session.det_current_frame_path("prompt_hud").with_suffix(".png"),
            with_async=True,
        )

    def hud_base(self):
        assert self.rv

        rv = self.rv
        self(chg=rv.chg, cfg=rv.cfg, seed=rv.seed)
        self.hud_ccg()
        self(prompt=rv.prompt)

    def hud_ccg(self):
        assert self.rv

        ccgs = []
        iccgs = []
        ccgas = []
        ccgbs = []
        i = 1
        while f"ccg{i}" in self.rv:
            ccg = self.rv[f"ccg{i}"]
            iccg = self.rv[f"iccg{i}"]
            ccga = self.rv[f"ccg{i}_a"]
            ccgb = self.rv[f"ccg{i}_b"]
            ccgs.append(ccg)
            iccgs.append(iccg)
            ccgas.append(ccga)
            ccgbs.append(ccgb)
            i += 1
        self(ccgs=tuple(ccgs))
        self(iccgs=tuple(iccgs))
        self(ccgas=tuple(ccgas))
        self(ccgbs=tuple(ccgbs))

    def snap_ccg_imgs(self):
        assert self.rv

        # std.save_guidance_imgs((rv.ccg3_img, rv.ccg2_img, rv.ccg1_img))
        i = 1
        while f"ccg{i}_img" in self.rv:
            img = self.rv[f"ccg{i}_img"]
            self.snap(f"ccg{i}_img", img)
            i += 1

    def normalize_image_size(self, image, target_width, target_height):
        """Resize and pad the image to the target size."""
        h, w = image.shape[:2]

        # Calculate scaling factor
        scale = min(target_width / w, target_height / h)

        # Resize image
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(image, (new_w, new_h))

        # Create a black canvas of target size
        result = np.zeros((target_height, target_width, 3), dtype=np.uint8)

        # Compute positioning
        x_offset = (target_width - new_w) // 2
        y_offset = (target_height - new_h) // 2

        # Place the resized image on the canvas
        result[y_offset : y_offset + new_h, x_offset : x_offset + new_w] = resized

        return result

    def save_tiled_snapshots(self, target_path):
        """
        Save tiled snapshots to the specified path.
        This function is designed to be used within the HUD module.
        """
        if len(self.snaps) <= 0:
            print("No snapshots to save.")
            return

        # Find the maximum dimensions
        max_height = max(img[1].shape[0] for img in self.snaps)
        max_width = max(img[1].shape[1] for img in self.snaps)

        def add_label_to_image(image, label, font_scale=0.5, thickness=1):
            """Add a label below the image."""
            # Create a new canvas with extra space for the label
            label_height = 30  # Adjust this value to change label height
            h, w = image.shape[:2]
            canvas = np.zeros((h + label_height, w, 3), dtype=np.uint8)

            # Place the original image on the canvas
            canvas[:h, :w] = image

            # Add the label
            cv2.putText(
                canvas,
                label,
                (10, h + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                thickness,
            )
            # Normalize image size

            return canvas

        # Normalize all images to the same size & add labels
        normalized_snaps = []
        for i, (name, img) in enumerate(self.snaps):
            normalized_img = self.normalize_image_size(img, max_width, max_height)
            labeled_img = add_label_to_image(normalized_img, name)
            normalized_snaps.append(labeled_img)

        # Tile images vertically
        grid = np.vstack(normalized_snaps)

        # Save the tiled image
        grid = cv2.cvtColor(grid, cv2.COLOR_RGB2BGR)
        cv2.imwrite(target_path, grid)

        print(f"Tiled snapshot saved to: {target_path}")

    def push(self, name: str):
        """Push a name onto the hierarchy stack"""
        self.name_stack.append(name)
        return self  # Enable method chaining

    def pop(self):
        """Pop the last name from the hierarchy stack"""
        if self.name_stack:
            self.name_stack.pop()
        return self  # Enable method chaining

    class NameContext:
        """Context manager for managing the HUD name stack"""

        def __init__(self, hud, name: str):
            self.hud = hud
            self.name = name

        def __enter__(self):
            self.hud.push(self.name)
            return self.hud

        def __exit__(self, exc_type, exc_val, exc_tb):
            self.hud.pop()

    def context(self, name: str):
        """Create a context manager for name hierarchy"""
        return self.NameContext(self, name)

    def scope(self, name: str):
        """Create a context manager for name hierarchy"""
        return self.NameContext(self, name)


HUD.null = HUD(False)
