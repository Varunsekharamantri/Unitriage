"""The safety checks in evals/fetch_photos.py (no network)."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "evals"))

from fetch_photos import UTM, check_host, credit_text, resolve_folder, safe_name  # noqa: E402


def test_names_become_safe_file_names():
    assert safe_name("Scuffed wall!") == "scuffed-wall"
    assert safe_name("../../etc/passwd") == "etc-passwd"          # no path tricks
    assert safe_name("a" * 200) == "a" * 60
    with pytest.raises(ValueError):
        safe_name("///")


def test_folders_are_matched_however_you_spell_them(tmp_path):
    (tmp_path / "Minor_NOT_DIY").mkdir()
    assert resolve_folder(tmp_path, "minor-not-diy").name == "Minor_NOT_DIY"
    assert resolve_folder(tmp_path, "MINOR NOT DIY").name == "Minor_NOT_DIY"
    assert resolve_folder(tmp_path, "emergency") == tmp_path / "emergency"   # new one, canonical name
    with pytest.raises(ValueError):
        resolve_folder(tmp_path, "../../somewhere-else")


def test_only_unsplash_https_addresses_are_ever_fetched():
    assert check_host("https://images.unsplash.com/photo-1") and check_host("https://api.unsplash.com/photos/x")
    for bad in ("http://images.unsplash.com/x", "https://evil.com/x", "https://unsplash.com.evil.com/x",
                "https://notunsplash.com/x", "file:///c:/secrets"):
        with pytest.raises(ValueError):
            check_host(bad)


def test_the_credit_names_the_photographer_and_links_back_with_the_referral_tags():
    photo = {
        "user": {"name": "Jane Doe", "links": {"html": "https://unsplash.com/@jane"}},
        "links": {"html": "https://unsplash.com/photos/abc"},
    }
    text = credit_text("Minor_DIY/unsplash-x.jpg", photo)
    assert "Jane Doe" in text and "Unsplash License" in text
    assert f"https://unsplash.com/@jane?{UTM}" in text and f"https://unsplash.com/photos/abc?{UTM}" in text
