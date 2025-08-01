import typing
from pathlib import Path
from typing import Union

import cv2
import numpy as np
import requests

if typing.TYPE_CHECKING:
    import torch

TPath = Union[str, Path]
TImage = Union[np.ndarray, "torch.Tensor"]


def resize(img: TImage, w: int, h: int, crop: bool = False) -> TImage:
    if img is None:
        return

    if isinstance(img, np.ndarray):
        # TODO this section without pillow
        if crop:
            # center anchored crop
            im = cv2pil(img)
            im = im.crop(w // 2 - w // 2,
                         h // 2 - h // 2,
                w // 2 + w // 2,
                         h // 2 + h // 2)
            return pil2cv(im)
        else:
            im = cv2pil(img)
            im = im.resize((w, h))
            return pil2cv(im)
    else:
        import torch

        if isinstance(img, torch.Tensor):
            if crop:
                raise NotImplementedError("Crop not implemented for torch.Tensor")
            else:
                ret = torch.nn.functional.interpolate(
                    img.type(torch.float32).unsqueeze(0), (h, w), mode="bilinear"
                )
                return ret

    raise ValueError(f"Unknown type of img: {type(img)}")



def bhwc2cv(bhwc):
    hwc = np.squeeze(bhwc, axis=0)
    # hwc = np.transpose(hwc, (1, 2, 0))
    return hwc.detach().cpu().numpy()


def cv2bhwc(hwc):
    import torch

    bhwc = np.expand_dims(hwc, axis=0)
    # bhwc = np.transpose(bhwc, (2, 0, 1))
    return torch.from_numpy(bhwc).float()


def ensure_extension(path: TPath, ext):
    path = Path(path)
    if path.suffix != ext:
        path = path.with_suffix(ext)
    return path


def save_jpg(pil: TImage, path: TPath, with_async=False):
    save_img(pil, path, with_async=with_async, img_format="JPEG")


def save_png(pil, path, with_async=False):
    save_img(pil, path, with_async=with_async, img_format="PNG")


def save_img(arr, path, with_async=False, img_format="PNG", quality=90):
    if img_format[0] == ".":
        img_format = {".jpg": "JPEG", ".png": "PNG"}.get(img_format, None)
        if img_format is None:
            raise ValueError(f"Unknown format: {img_format}")

    with tracer(f"save_img({Path(path).relative_to(Path.cwd())}, async={with_async}, {arr.shape})"):
        path = Path(path)
        if img_format == "PNG":
            path = ensure_extension(path, ".png")
        elif img_format == "JPEG":
            path = ensure_extension(path, ".jpg")

        if with_async:
            save_async(path, arr)
        else:
            Path(path).parent.mkdir(parents=True, exist_ok=True)

            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
            cv2.imwrite(path.as_posix(), arr)

def save_async(path, arr, format="PNG", quality=90) -> None:
    if isinstance(path, Path):
        path = path.as_posix()
    arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)

    # Use threaded lambda to save image
    def write(im) -> None:
        try:
            # if isinstance(im, np.ndarray):
            #     im = cv2pil(im)
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            if path.endswith(".jpg") or path.endswith(".jpeg"):
                cv2.imwrite(path, im, [cv2.IMWRITE_JPEG_QUALITY, quality])
            else:
                cv2.imwrite(path, im)
        # im.save(path, format='PNG')
        except:
            print(f"Error saving image: {path}")

    import threading

    t = threading.Thread(target=write, args=(arr,))
    t.start()


# def save_jpg(pil, path, quality=90):
#     path = ensure_extension(path, '.jpg')
#     path = Path(path)
#     path.parent.mkdir(parents=True, exist_ok=True)
#     pil.save(path, format='JPEG', quality=quality)


def save_npy(path, nparray):
    path = ensure_extension(path, ".npy")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(path), nparray)


def load_npy(path):
    path = ensure_extension(path, ".npy")
    path = Path(path)
    if not path.is_file():
        return None
    return np.load(str(path))


def save_json(data, path):
    import json

    path = Path(path).with_suffix(".json")

    if isinstance(data, dict) or isinstance(data, list):
        # data = json.dumps(data, indent=4, sort_keys=True)
        data = json.dumps(data)

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path.as_posix(), "w") as w:
        w.write(data)


def load_json(path, default="required"):
    import json

    is_required = default == "required"

    path = Path(path).with_suffix(".json")
    if not path.is_file():
        if is_required:
            raise FileNotFoundError(f"File not found: {path}")
        else:
            return default

    try:
        with open(path.as_posix(), "r") as r:
            return json.load(r)
    except Exception as e:
        if is_required:
            raise e
        else:
            return default



def load_cv2(pil: TImage | Path | str) -> np.ndarray:
    # LOAD IMAGE FROM FILE
    if isinstance(pil, Path) or isinstance(pil, str) and not pil.startswith("http"):
        # img = Image.open(pil.as_posix())
        # ret = pil2cv(img)
        ret = cv2.imread(pil.as_posix() if isinstance(pil, Path) else pil)
        ret = cv2.cvtColor(ret, cv2.COLOR_BGR2RGB)
    # WEB URL
    elif isinstance(pil, str) and pil.startswith("http"):
        # img = Image.open(requests.get(pil, stream=True).raw)
        # ret = pil2cv(img)
        ret = cv2.imread(requests.get(pil, stream=True).raw)
        ret = cv2.cvtColor(ret, cv2.COLOR_BGR2RGB)
    # LOAD IMAGE FROM FILE
    elif isinstance(pil, str) and Path(pil).is_file():
        ret = cv2.imread(pil)
    elif isinstance(pil, np.ndarray) and pil.dtype == np.uint8 and len(pil.shape) == 3 and pil.shape[2] == 3:
        ret = pil
    else:
        raise ValueError(f"Unknown type of path: {type(pil)}")

    if ret is not None and ret.shape[2] == 4:
        ret = ret[:, :, :3]

    return ret


def as_cv2(pil: TImage | Path | str) -> np.ndarray:
    assert pil is not None

    ret = None

    # LOAD IMAGE FROM NUMPY ARRAY
    if isinstance(pil, np.ndarray):
        ret = pil
    else:
        return load_cv2(pil)

    # Remove alpha channel
    if ret is not None and ret.shape[2] == 4:
        ret = ret[:, :, :3]

    return ret


def get_cv2(
    pil: TImage | Path | str, target_size: tuple[int, int]
) -> np.ndarray:
    ret = None

    has_size = isinstance(target_size, tuple) and None not in target_size

    # LOAD IMAGE FROM NUMPY ARRAY
    if isinstance(pil, np.ndarray):
        ret = pil
    # LOAD IMAGE FROM FILE
    elif isinstance(pil, Path):
        # img = Image.open(pil.as_posix())
        # ret = pil2cv(img)
        ret = cv2.imread(pil.as_posix())
        ret = cv2.cvtColor(ret, cv2.COLOR_BGR2RGB)
    # WEB URL
    elif isinstance(pil, str) and pil.startswith("http"):
        # img = Image.open(requests.get(pil, stream=True).raw)
        # ret = pil2cv(img)
        ret = cv2.imread(requests.get(pil, stream=True).raw)
        ret = cv2.cvtColor(ret, cv2.COLOR_BGR2RGB)
    # LOAD IMAGE FROM FILE
    elif isinstance(pil, str) and Path(pil).is_file():
        ret = cv2.imread(pil)
    # LOAD IMAGE FROM COLOR STRING
    elif isinstance(pil, str) and pil.startswith("#"):
        # rgb = Image.new("RGB", target_size or (1, 1), color=pil)
        # ret = np.asarray(rgb)
        raise NotImplementedError # TODO
    # BLACK FRAMEkk
    elif pil == "black":
        ret = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
    # WHITE FRAME
    elif pil == "white":
        ret = np.ones((target_size[1], target_size[0], 3), dtype=np.uint8) * 255
    # LOAD IMAGE FROM COLOR STRING
    elif isinstance(pil, str) and pil.startswith("#"):
        # color string like 'black', etc.
        # TODO
        rgb = Image.new("RGB", target_size or (1, 1), color=pil)
        ret = np.asarray(rgb)
    elif has_size:  # load_cv2 always tries to return some sort of img array no matter what
        ret = np.ones((target_size[1], target_size[0], 3), dtype=np.uint8) * 255
        ret[:, :, 0] = 255
    else:
        raise ValueError(f"Unknown type of path: {type(pil)}")

    # Remove alpha channel
    if ret is not None and ret.shape[2] == 4:
        ret = ret[:, :, :3]

    return ret


def load_torch(path_or_cv2):
    import torch

    img = load_cv2(path_or_cv2)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img


def load_txt(path):
    path = Path(path)
    if not path.is_file():
        return None
    with open(path.as_posix(), "r") as r:
        return r.read().strip()


def save_txt(path, txt):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path.as_posix(), "w") as w:
        w.write(txt)


def crop_or_pad(image, width=None, height=None, bg=(0, 0, 0), anchor=(0.5, 0.5)):
    h_img, w_img, _ = image.shape

    if width == w_img and height == h_img:
        return image  # No need to crop or pad if dimensions are already equal

    # If width or height is None, calculate it from the other dimension (to preserve aspect ratio)
    if width is None and height is None:
        width = int(height * w_img / h_img)
    elif height is None and width is not None:
        height = int(width * h_img / w_img)

    # Calculate the padding sizes
    pad_left = 0
    pad_right = 0
    pad_top = 0
    pad_bottom = 0

    if width > w_img:
        pad_left = int((width - w_img) * anchor[0])
        pad_right = width - w_img - pad_left
    elif width < w_img:
        crop_left = int((w_img - width) * anchor[0])
        crop_right = w_img - width - crop_left
        image = image[:, crop_left: w_img - crop_right]

    if height > h_img:
        pad_top = int((height - h_img) * anchor[1])
        pad_bottom = height - h_img - pad_top
    elif height < h_img:
        crop_top = int((h_img - height) * anchor[1])
        crop_bottom = h_img - height - crop_top
        image = image[crop_top: h_img - crop_bottom, :]

    # Pad the image
    image = cv2.copyMakeBorder(
        image,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        cv2.BORDER_CONSTANT,
        value=(0, 0, 0, 1),
    )

    return image


def is_image(value):
    is_image = isinstance(value, np.ndarray) or isinstance(value, PImage)
    if is_image:
        return True

    import torch

    if isinstance(value, torch.Tensor):
        return True

    return False


# def crop_or_pad(ret, w, h, param):
#     """
#     Crop or pad h,w,c image to fit w and h
#     """
#
#
#     pass
#
#     # with trace(f"res_frame_cv2: crop"):
#
#     #     w = ret.shape[1]
#     #     h = ret.shape[0]
#     #     ow = w
#     #     oh = h
#     #     if w > h:
#     #         # Too wide, crop width
#     #         cropped_span = ow - self.w
#     #         if cropped_span > 0:
#     #             ret = ret[0:oh, cropped_span // 2:ow - cropped_span // 2]
#     #         else:
#     #             # We have to pad with black borders
#     #             w = self.w
#     #             h = self.h
#     #             padded = np.zeros((h, w, 3), dtype=np.uint8)
#     #             padded[:, (w - ow) // 2:(w - ow) // 2 + ow] = ret[0:oh, 0:ow]
#     #             ret = padded
#     #     else:
#     #         # Too tall, crop height
#     #         cropped_span = oh - self.h
#     #         if cropped_span > 0:
#     #             ret = ret[cropped_span // 2:oh - cropped_span // 2, 0:ow]
#     #         else:
#     #             # We have to pad with black borders
#     #             w = self.w
#     #             h = self.h
#     #             padded = np.zeros((h, w, 3), dtype=np.uint8)
#     #             padded[(h - oh) // 2:(h - oh) // 2 + oh, :] = ret[0:oh, 0:ow]
#     #             ret = padded
#     # return None
