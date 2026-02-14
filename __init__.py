from .nodes import LoadImageFromFolder, LoadImagesFromFolder, SaveImageWithPrompt

NODE_CLASS_MAPPINGS = {
    "Save Image With Prompt": SaveImageWithPrompt,
    "Load Image From Folder": LoadImageFromFolder,
    "Load Images From Folder": LoadImagesFromFolder,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "Save Image With Prompt": "Save Image With Prompt",
    "Load Image From Folder": "Load Image From Folder",
    "Load Images From Folder": "Load Images From Folder",
}
