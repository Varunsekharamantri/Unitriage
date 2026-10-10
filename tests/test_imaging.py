import base64
import io
import os

import pytest
from PIL import Image

from app.services.imaging import MAX_SIDE, SHRINK_ABOVE, UnreadableImage, prepare_image
from app.services.vision_service import MockProvider, TriageError, VisionService

DRAIN = {
    "severity": "minor", "confidence": 0.92, "trade": "plumber",
    "is_safe_for_diy": True, "summary": "Slow kitchen drain",
}


def jpeg_b64(width, height, noisy=True, orientation=None) -> str:
    """A JPEG of random noise (which compresses badly, so it is big) as base64."""
    data = os.urandom(width * height * 3) if noisy else bytes(width * height * 3)
    im = Image.frombytes("RGB", (width, height), data)
    exif = Image.Exif()
    if orientation:
        exif[0x0112] = orientation
    out = io.BytesIO()
    im.save(out, "JPEG", quality=95, exif=exif)
    return base64.b64encode(out.getvalue()).decode()


def decode(b64):
    return Image.open(io.BytesIO(base64.b64decode(b64)))


def test_small_photo_is_passed_through_untouched():
    small = jpeg_b64(200, 150)
    assert len(base64.b64decode(small)) <= SHRINK_ABOVE
    assert prepare_image(small, "image/jpeg") == (small, "image/jpeg")


def test_big_photo_is_shrunk_to_something_every_provider_accepts():
    big = jpeg_b64(2400, 1800)
    assert len(base64.b64decode(big)) > SHRINK_ABOVE
    out, mime = prepare_image(big, "image/jpeg")
    im = decode(out)
    assert mime == "image/jpeg" and max(im.size) == MAX_SIDE
    assert im.size == (1600, 1200)                       # same shape as before
    assert len(out) < 4_000_000                           # under Groq's inline limit once encoded
    assert len(out) < len(big)


def test_the_cameras_rotation_is_applied_before_it_is_lost():
    # a landscape picture marked "rotate 90 degrees" is really a portrait one
    big = jpeg_b64(2400, 1200, orientation=6)
    out, _ = prepare_image(big, "image/jpeg")
    w, h = decode(out).size
    assert h > w


def test_png_with_transparency_becomes_a_plain_jpeg():
    im = Image.frombytes("RGBA", (1500, 1200), os.urandom(1500 * 1200 * 4))
    buf = io.BytesIO()
    im.save(buf, "PNG")
    assert buf.tell() > SHRINK_ABOVE
    out, mime = prepare_image(base64.b64encode(buf.getvalue()).decode(), "image/png")
    assert mime == "image/jpeg" and decode(out).mode == "RGB"


def test_garbage_that_claims_to_be_a_big_photo_is_rejected():
    junk = base64.b64encode(b"not an image at all " * 100_000).decode()
    with pytest.raises(UnreadableImage):
        prepare_image(junk, "image/jpeg")


def test_the_service_sends_providers_the_shrunk_photo_once():
    big = jpeg_b64(2400, 1800)
    provider = MockProvider(["garbage", DRAIN])  # first reply is invalid, so it is retried
    VisionService([provider], retries_per_provider=2).triage(big)
    assert provider.calls == 2
    assert provider.image_sizes[0] == provider.image_sizes[1] < len(big)


def test_an_unreadable_photo_becomes_a_triage_error_so_a_person_reviews_it():
    junk = base64.b64encode(b"x" * 2_000_000).decode()
    provider = MockProvider([DRAIN])
    with pytest.raises(TriageError, match="could not be read"):
        VisionService([provider]).triage(junk)
    assert provider.calls == 0  # never sent to a model
