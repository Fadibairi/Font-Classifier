"""Generate a synthetic font dataset (one folder per font) from .ttf/.otf files.

Usage:
    python scripts/generate_dataset.py --fonts fonts/ --out data/train --per-font 2000

Put any fonts you are allowed to use in fonts/ (e.g. open-licensed Google Fonts).
Each image is a single word rendered in a random color on a random background,
matching the "folder name = label" layout expected by the training script.
"""
import argparse
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

WORDS = [
    "design", "brand", "market", "product", "studio", "light", "future", "vision",
    "story", "create", "summer", "coffee", "travel", "modern", "classic", "digital",
    "ocean", "garden", "simple", "bold",
]


def rand_color():
    return tuple(random.randint(0, 255) for _ in range(3))


def render_word(font_path: Path, size: int = 128) -> Image.Image:
    word = random.choice(WORDS)
    word = random.choice([word, word.capitalize(), word.upper()])
    font = ImageFont.truetype(str(font_path), random.randint(24, 48))
    bg, fg = rand_color(), rand_color()
    img = Image.new("RGB", (size, size), bg)
    draw = ImageDraw.Draw(img)
    left, top, right, bottom = draw.textbbox((0, 0), word, font=font)
    x = max(0, (size - (right - left)) // 2 - left)
    y = max(0, (size - (bottom - top)) // 2 - top)
    draw.text((x, y), word, font=font, fill=fg)
    return img


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fonts", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--per-font", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    random.seed(args.seed)

    font_files = sorted([*args.fonts.glob("*.ttf"), *args.fonts.glob("*.otf")])
    if not font_files:
        raise SystemExit(f"No .ttf/.otf files found in {args.fonts}")

    for font_path in font_files:
        target = args.out / font_path.stem
        target.mkdir(parents=True, exist_ok=True)
        for i in range(args.per_font):
            render_word(font_path).save(target / f"{i:05d}.png")
        print(f"{font_path.stem}: {args.per_font} images")


if __name__ == "__main__":
    main()
