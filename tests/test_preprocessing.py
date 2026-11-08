from PIL import Image

from src.preprocessing import IMG_SIZE, resize_pad, to_hsv_value, upscale


def test_hsv_value_returns_grayscale_rgb():
    img = Image.new("RGB", (40, 20), (200, 30, 90))
    out = to_hsv_value(img)
    assert out.mode == "RGB"
    r, g, b = out.getpixel((0, 0))
    assert r == g == b  # single channel replicated


def test_resize_pad_output_is_square():
    out = resize_pad(Image.new("RGB", (300, 50), (0, 0, 0)))
    assert out.size == (IMG_SIZE, IMG_SIZE)


def test_resize_pad_keeps_aspect_ratio():
    # a wide black word on a white canvas: top and bottom rows must stay white
    out = resize_pad(Image.new("RGB", (300, 50), (0, 0, 0)))
    assert out.getpixel((IMG_SIZE // 2, 0)) == (255, 255, 255)
    assert out.getpixel((IMG_SIZE // 2, IMG_SIZE // 2)) == (0, 0, 0)


def test_resize_pad_handles_tiny_images():
    out = resize_pad(Image.new("RGB", (1, 1)))
    assert out.size == (IMG_SIZE, IMG_SIZE)


def test_upscale_doubles_size():
    assert upscale(Image.new("RGB", (30, 10))).size == (60, 20)
