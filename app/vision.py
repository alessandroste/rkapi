"""Fixed-grid Qwen3.5 preprocessing; normalization is embedded in the RKNN graph."""

import io

import numpy as np
from PIL import Image, UnidentifiedImageError


class VisionEncoder:
    def __init__(self, model_path, library_path, max_pixels=16_000_000, embedding_size=2048):
        from app._native import Vision  # pylint: disable=import-outside-toplevel
        if max_pixels <= 0:
            raise ValueError("max_pixels must be positive")
        self.max_pixels = max_pixels
        self.native = Vision(library_path, model_path, embedding_size)

    def preprocess(self, image_data):
        try:
            with Image.open(io.BytesIO(image_data)) as image:
                if image.format not in ("PNG", "JPEG", "WEBP"):
                    raise ValueError("Only PNG, JPEG and WebP images are supported")
                if image.width * image.height > self.max_pixels:
                    raise ValueError("Decoded image exceeds MAX_IMAGE_PIXELS")
                if getattr(image, "n_frames", 1) != 1:
                    raise ValueError("Animated images are not supported")
                # Qwen uses RGB, bicubic resize, and no MiniCPM-style square padding.
                rgb = image.convert("RGB").resize(
                    (self.native.width, self.native.height), Image.Resampling.BICUBIC
                )
                return np.ascontiguousarray(rgb, dtype=np.uint8)
        except (OSError, UnidentifiedImageError, Image.DecompressionBombError) as error:
            raise ValueError("Invalid or oversized image") from error

    def encode(self, image_data):
        embeddings = self.native.encode(self.preprocess(image_data))
        return {
            "embeddings": embeddings,
            "n_image_tokens": self.native.image_tokens,
            "image_width": self.native.width,
            "image_height": self.native.height,
        }

    def close(self):
        self.native.close()
