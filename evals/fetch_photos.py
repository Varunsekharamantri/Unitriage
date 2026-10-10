"""Find reference photos on Unsplash (official API) and save the ones you approve.

    # 1. LOOK: lists matches. Downloads nothing.
    .venv/Scripts/python.exe evals/fetch_photos.py search "scuffed wall paint" --per-page 10

    # 2. SAVE one you have approved, into the folder that holds its correct answer:
    .venv/Scripts/python.exe evals/fetch_photos.py download PHOTO_ID --folder minor-diy --name scuffed-wall

Needs UNSPLASH_ACCESS_KEY in .env (free, from unsplash.com/developers).

Following Unsplash's API rules: the "download" step calls their download-tracking endpoint, and
the photographer is credited in evals/images/CREDITS.txt. Photos are Unsplash-licensed, not yours
to redistribute: keep evals/images/ out of any public repository.
"""
import argparse
import re
import sys
from pathlib import Path
from urllib.parse import urlparse

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import config  # noqa: E402
from run_eval import FOLDERS, folder_key  # noqa: E402

API = "https://api.unsplash.com"
UTM = "utm_source=unitriage_eval&utm_medium=referral"   # the referral tags Unsplash asks for
IMAGES = Path(__file__).resolve().parent / "images"
MAX_BYTES = 8 * 1024 * 1024


def safe_name(name: str) -> str:
    """A file-name-safe slug: letters, digits and dashes only."""
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    if not slug:
        raise ValueError("--name needs at least one letter or digit")
    return slug[:60]


def resolve_folder(images_dir: Path, folder: str) -> Path:
    """The existing folder that means the same thing (Minor_DIY, minor-diy, minor diy...)."""
    key = folder_key(folder)
    if key not in FOLDERS:
        raise ValueError(f"folder must be one of: {', '.join(FOLDERS)}")
    for existing in images_dir.iterdir():
        if existing.is_dir() and folder_key(existing.name) == key:
            return existing
    return images_dir / key


def check_host(url: str) -> str:
    """Only ever fetch from Unsplash's own hosts."""
    host = urlparse(url).hostname or ""
    if urlparse(url).scheme != "https" or not (host == "unsplash.com" or host.endswith(".unsplash.com")):
        raise ValueError(f"refusing to fetch from a non-Unsplash address: {url}")
    return url


def credit_text(rel_path: str, photo: dict) -> str:
    user = photo["user"]
    return (
        f"\n{rel_path}\n"
        f"  Photo by {user['name']} ({user['links']['html']}?{UTM}) on Unsplash\n"
        f"  {photo['links']['html']}?{UTM}\n"
        f"  Unsplash License (free to use; not for redistribution as a photo collection)\n"
    )


def call(url_or_path: str, **params):
    if not config.UNSPLASH_ACCESS_KEY:
        sys.exit("UNSPLASH_ACCESS_KEY is not set in .env")
    url = url_or_path if url_or_path.startswith("http") else API + url_or_path
    r = httpx.get(
        check_host(url),
        params=params or None,
        headers={"Authorization": f"Client-ID {config.UNSPLASH_ACCESS_KEY}", "Accept-Version": "v1"},
        timeout=30,
    )
    if r.status_code == 401:
        sys.exit("Unsplash rejected the key (401). Check UNSPLASH_ACCESS_KEY in .env: use the Access Key.")
    if r.status_code in (403, 429):
        sys.exit(f"Unsplash says you are over the rate limit ({r.status_code}). Wait a while and try again.")
    r.raise_for_status()
    return r.json()


def cmd_search(args) -> int:
    data = call("/search/photos", query=args.query, per_page=min(args.per_page, 30),
                orientation=args.orientation) if args.orientation else \
        call("/search/photos", query=args.query, per_page=min(args.per_page, 30))
    results = data["results"]
    if not results:
        print("No matches. Try other words.")
        return 1
    print(f"{data['total']} matches for {args.query!r}; showing {len(results)}. Nothing has been downloaded.\n")
    for i, p in enumerate(results, 1):
        desc = (p.get("alt_description") or p.get("description") or "(no description)").strip()
        print(f"{i:2}. id={p['id']}  {p['width']}x{p['height']}  by {p['user']['name']}")
        print(f"    {desc[:100]}")
        print(f"    preview: {p['urls']['small']}")
        print(f"    page:    {p['links']['html']}?{UTM}")
    print("\nTo save one:  fetch_photos.py download <id> --folder <folder> --name <short-name>")
    return 0


def cmd_download(args) -> int:
    name = safe_name(args.name)
    folder = resolve_folder(IMAGES, args.folder)
    target = folder / f"unsplash-{name}.jpg"
    if target.exists():
        sys.exit(f"{target.name} already exists in {folder.name}: choose another --name")

    photo = call(f"/photos/{args.photo_id}")
    # Unsplash asks apps to call this when a photo is downloaded, so photographers get counted.
    call(photo["links"]["download_location"])
    image_url = check_host(photo["urls"]["regular"])      # ~1080px wide: plenty, and small

    r = httpx.get(image_url, timeout=60, follow_redirects=True)
    r.raise_for_status()
    check_host(str(r.url))
    body = r.content
    if len(body) > MAX_BYTES:
        sys.exit(f"Refusing a {len(body) // 1024} KB file (limit {MAX_BYTES // 1024} KB).")
    if body[:2] != b"\xff\xd8":
        sys.exit("The download is not a JPEG, so I did not save it.")

    folder.mkdir(parents=True, exist_ok=True)
    target.write_bytes(body)
    rel = f"{folder.name}/{target.name}"
    with open(IMAGES / "CREDITS.txt", "a", encoding="utf-8") as f:
        f.write(credit_text(rel, photo))
    print(f"Saved {rel}  ({len(body) // 1024} KB, {photo['width']}x{photo['height']} original)")
    print(f"Credit added to evals/images/CREDITS.txt: photo by {photo['user']['name']}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("search", help="list matching photos (downloads nothing)")
    s.add_argument("query")
    s.add_argument("--per-page", type=int, default=8)
    s.add_argument("--orientation", choices=["landscape", "portrait", "squarish"])
    d = sub.add_parser("download", help="save one approved photo into an evals folder")
    d.add_argument("photo_id")
    d.add_argument("--folder", required=True, help=" / ".join(FOLDERS))
    d.add_argument("--name", required=True, help="short file name, e.g. scuffed-wall")
    args = ap.parse_args()
    try:
        return cmd_search(args) if args.cmd == "search" else cmd_download(args)
    except ValueError as exc:
        sys.exit(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
