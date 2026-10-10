"""Shrinks large photos before they go to a vision model.

Phone photos are routinely 3-8 MB. Vision APIs cap the size of an inline image (Groq's is about
4 MB once base64-encoded, which is only ~3 MB of picture), and more pixels than the model can use
only cost time and tokens. Small photos are passed through untouched.
"""
import base64
import io

from PIL import Image, ImageOps, UnidentifiedImageError

MAX_SIDE = 1600           # longest edge after shrinking, in pixels
SHRINK_ABOVE = 1_500_000  # photos at or under this many bytes are sent as they are
Image.MAX_IMAGE_PIXELS = 50_000_000  # refuse decompression bombs rather than decode them


class UnreadableImage(ValueError):
    """The bytes are not an image we can decode."""


def prepare_image(image_b64: str, mime_type: str) -> tuple[str, str]:
    """Returns (base64, mime) ready to send: unchanged if small, else a smaller JPEG."""
    raw = base64.b64decode(image_b64)
    if len(raw) <= SHRINK_ABOVE:
        return image_b64, mime_type
    try:
        with Image.open(io.BytesIO(raw)) as im:
            im = ImageOps.exif_transpose(im)  # apply the camera's rotation before it is lost
            im = im.convert("RGB")            # JPEG has no alpha channel
            im.thumbnail((MAX_SIDE, MAX_SIDE))
            out = io.BytesIO()
            im.save(out, "JPEG", quality=85, optimize=True)
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise UnreadableImage(f"could not read the image: {exc.__class__.__name__}") from exc
    return base64.b64encode(out.getvalue()).decode(), "image/jpeg"
