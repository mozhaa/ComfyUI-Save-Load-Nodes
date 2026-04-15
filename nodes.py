import datetime
import json
import re
from pathlib import Path

import folder_paths
import numpy as np
import torch
from PIL import Image

import nodes

mapping = {
    "yyyy": "%Y",
    "yy": "%y",
    "MM": "%m",
    "M": "%m",
    "dd": "%d",
    "d": "%d",
    "hh": "%H",
    "h": "%H",
    "mm": "%M",
    "m": "%M",
    "ss": "%S",
    "s": "%S",
}
pattern = re.compile("|".join(sorted(mapping.keys(), key=len, reverse=True)))


def _load_image_and_prompts(image_path: Path):
    img = Image.open(image_path)
    img_rgb = img.convert("RGB")
    img_np = np.array(img_rgb).astype(np.float32) / 255.0
    image_tensor = torch.from_numpy(img_np)[None, ...]

    if "A" in img.getbands():
        mask_np = np.array(img.getchannel("A")).astype(np.float32) / 255.0
        mask_tensor = 1.0 - torch.from_numpy(mask_np)
    else:
        mask_tensor = torch.zeros((64, 64), dtype=torch.float32)

    pos_raw = img.info.get("positive_prompt", "")
    neg_raw = img.info.get("negative_prompt", "")
    try:
        pos = json.loads(pos_raw)
    except:
        pos = pos_raw
    try:
        neg = json.loads(neg_raw)
    except:
        neg = neg_raw

    return image_tensor, mask_tensor, pos, neg


class SaveImageWithPrompt(nodes.SaveImage):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "positive_prompt": ("STRING", {"forceInput": True}),
                "negative_prompt": ("STRING", {"forceInput": True}),
                "filename_prefix": ("STRING", {"default": "ComfyUI"}),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    def save_images(
        self,
        images,
        positive_prompt,
        negative_prompt,
        filename_prefix="ComfyUI",
        prompt=None,
        extra_pnginfo=None,
    ):
        def replace_date(match):
            fmt = match.group(1)
            fmt = pattern.sub(lambda m: mapping[m.group(0)], fmt)
            return datetime.datetime.now().strftime(fmt)

        filename_prefix = re.sub(r"%date:([^%]+)%", replace_date, filename_prefix)

        if extra_pnginfo is None:
            extra_pnginfo = {}
        extra_pnginfo["positive_prompt"] = positive_prompt
        extra_pnginfo["negative_prompt"] = negative_prompt

        return super().save_images(images, filename_prefix, prompt, extra_pnginfo)


class LoadImageFromFolder:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "path": ("STRING", {"default": "output"}),
                "index": ("INT", {"default": 0, "min": 0, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "STRING", "STRING")
    RETURN_NAMES = ("image", "mask", "positive_prompt", "negative_prompt")
    FUNCTION = "load_image"
    CATEGORY = "image"

    def load_image(self, path, index):
        folder = Path(folder_paths.base_path) / path
        valid_exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}
        images = sorted([f for f in folder.iterdir() if f.suffix.lower() in valid_exts])
        return _load_image_and_prompts(images[index])


class LoadImagesFromFolder:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"path": ("STRING", {"default": "output"})}}

    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    FUNCTION = "load_images"
    CATEGORY = "image"

    def load_images(self, path):
        folder = Path(folder_paths.base_path) / path
        if not folder.is_dir():
            raise FileNotFoundError(f"Folder not found: {folder}")
        valid_exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}
        image_files = sorted(
            [f for f in folder.iterdir() if f.suffix.lower() in valid_exts]
        )
        if not image_files:
            raise ValueError(f"No image files in {folder}")

        images, masks, prompts = [], [], []
        for f in image_files:
            img_t, mask_t, pos, neg = _load_image_and_prompts(f)
            images.append(img_t)
            masks.append(mask_t)
            prompts.append({"positive": pos, "negative": neg})

        images_tensor = torch.cat(images, dim=0)
        masks_tensor = torch.stack(masks, dim=0)
        prompts_json = json.dumps(prompts)
        return (images_tensor, masks_tensor, prompts_json)


class LoadImageWithPrompt(nodes.LoadImage):
    RETURN_TYPES = ("IMAGE", "MASK", "STRING", "STRING")
    RETURN_NAMES = ("image", "mask", "positive_prompt", "negative_prompt")
    FUNCTION = "load_image"

    def load_image(self, image):
        img_tensor, mask_tensor = super().load_image(image)
        image_path = folder_paths.get_annotated_filepath(image)
        _, _, pos, neg = _load_image_and_prompts(Path(image_path))
        return (img_tensor, mask_tensor, pos, neg)
