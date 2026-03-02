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


class SaveImageWithPrompt(nodes.SaveImage):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "prompt_text": ("STRING", {"forceInput": True}),
                "filename_prefix": ("STRING", {"default": "ComfyUI"}),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    def save_images(
        self,
        images,
        prompt_text,
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
        extra_pnginfo["custom_prompt"] = prompt_text

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

    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    FUNCTION = "load_image"
    CATEGORY = "image"

    def load_image(self, path, index):
        folder = Path(folder_paths.base_path) / path
        valid_exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}
        images = sorted(
            [
                f
                for f in folder.iterdir()
                if f.is_file() and f.suffix.lower() in valid_exts
            ]
        )
        img = Image.open(images[index])
        prompt_text = img.info.get("custom_prompt", "")
        if prompt_text:
            try:
                prompt_text = json.loads(prompt_text)
            except:
                pass

        image = img.convert("RGB")
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image)[None,]

        if "A" in img.getbands():
            mask = np.array(img.getchannel("A")).astype(np.float32) / 255.0
            mask = 1.0 - torch.from_numpy(mask)
        else:
            mask = torch.zeros((64, 64), dtype=torch.float32, device="cpu")

        return (image, mask, prompt_text)


class LoadImagesFromFolder:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "path": ("STRING", {"default": "output"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    FUNCTION = "load_images"
    CATEGORY = "image"

    def load_images(self, path):
        folder = Path(folder_paths.base_path) / path
        if not folder.exists() or not folder.is_dir():
            raise FileNotFoundError(f"Folder not found: {folder}")

        valid_exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}
        image_files = sorted(
            [
                f
                for f in folder.iterdir()
                if f.is_file() and f.suffix.lower() in valid_exts
            ]
        )
        if not image_files:
            raise ValueError(f"No image files found in {folder}")

        images = []
        masks = []
        prompts = []
        for img_file in image_files:
            img = Image.open(img_file)
            img_rgb = img.convert("RGB")
            img_np = np.array(img_rgb).astype(np.float32) / 255.0
            images.append(torch.from_numpy(img_np))

            if "A" in img.getbands():
                mask_np = np.array(img.getchannel("A")).astype(np.float32) / 255.0
                mask = 1.0 - torch.from_numpy(mask_np)
            else:
                mask = torch.zeros(img_np.shape[:2], dtype=torch.float32)
            masks.append(mask)

            prompt_text = img.info.get("custom_prompt", "")
            if prompt_text:
                try:
                    prompt_text = json.loads(prompt_text)
                except:
                    pass
            prompts.append(prompt_text)

        images_tensor = torch.stack(images, dim=0)
        masks_tensor = torch.stack(masks, dim=0)
        prompts_json = json.dumps(prompts)
        return (images_tensor, masks_tensor, prompts_json)


class LoadImageWithPrompt(nodes.LoadImage):
    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    FUNCTION = "load_image"

    def load_image(self, image):
        img_tensor, mask_tensor = super().load_image(image)
        image_path = folder_paths.get_annotated_filepath(image)
        with Image.open(image_path) as pil_img:
            prompt = pil_img.info.get("custom_prompt", "")
        return (img_tensor, mask_tensor, prompt)
