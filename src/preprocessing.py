"""Image preprocessing shared by training and inference.

Keeping these functions in one module guarantees that the model sees the same
kind of input at training time and at inference time.
"""
from PIL import Image

IMG_SIZE = 128


def to_hsv_value(img: Image.Image) -> Image.Image:
    """Keep only the HSV value (brightness) channel, returned as 3-channel RGB.

    Color is mostly background and does not define a font, so it is dropped.
    """
    return img.convert("HSV").split()[2].convert("RGB")


def resize_pad(img: Image.Image, size: int = IMG_SIZE, fill: int = 255) -> Image.Image:
    """Resize to fit in a size x size square without distorting the letters."""
    w, h = img.size
    scale = size / max(w, h)
    img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))))
    canvas = Image.new("RGB", (size, size), (fill, fill, fill))
    canvas.paste(img, ((size - img.width) // 2, (size - img.height) // 2))
    return canvas


def upscale(img: Image.Image, factor: int = 2) -> Image.Image:
    """Upscale before text detection (small text is detected more reliably)."""
    return img.resize((img.width * factor, img.height * factor), Image.BICUBIC)
