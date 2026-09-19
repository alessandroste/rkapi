"""Fixed-grid Qwen3.5 preprocessing; normalization is embedded in the RKNN graph."""

import hashlib
import io

import numpy as np
from PIL import Image, UnidentifiedImageError


class VisionEncoder:
    """Cache one image per loaded encoder; the service serializes encode/close."""

    def __init__(self, model_path, library_path, max_pixels=16_000_000, embedding_size=2048):
        from app._native import Vision  # pylint: disable=import-outside-toplevel
        if max_pixels <= 0:
            raise ValueError("max_pixels must be positive")
        self.max_pixels = max_pixels
        self.native = Vision(library_path, model_path, embedding_size)
        self._cached_image: tuple[bytes, np.ndarray] | None = None

    def preprocess(self, image_data):
        try:
            with Image.open(io.BytesIO(image_data)) as image:
                if image.format not in ("PNG", "JPEG", "WEBP"):
                    raise ValueError("Only PNG, JPEG and WebP images are supported")
                if image.width * image.height > self.max_pixels:
                    raise ValueError("Decoded image exceeds MAX_IMAGE_PIXELS")
                if getattr(image, "n_frames", 1) != 1:
                    raise ValueError("Animated images are not supported")
                # Pad at the encoder size to avoid allocating huge squares for thin images.
                width, height = self.native.width, self.native.height
                scale = min(width / image.width, height / image.height)
                size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
                rgb = image.convert("RGB").resize(size, Image.Resampling.BILINEAR)
                padded = Image.new("RGB", (width, height), (128, 128, 128))
                padded.paste(rgb, ((width - rgb.width) // 2, (height - rgb.height) // 2))
                return np.ascontiguousarray(padded, dtype=np.uint8)
        except (OSError, UnidentifiedImageError, Image.DecompressionBombError) as error:
            raise ValueError("Invalid or oversized image") from error

    def encode(self, image_data):
        key = hashlib.sha256(image_data).digest()
        if self._cached_image is None or self._cached_image[0] != key:
            embeddings = self.native.encode(self.preprocess(image_data))
            embeddings.setflags(write=False)
            self._cached_image = (key, embeddings)
        return {
            "embeddings": self._cached_image[1],
            "n_image_tokens": self.native.image_tokens,
            "image_width": self.native.width,
            "image_height": self.native.height,
        }

    def close(self):
        self._cached_image = None
        self.native.close()
