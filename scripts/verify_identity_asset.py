#!/usr/bin/env python3
"""Verify the canonical picoLLM identity dataset and optional hosted mirror."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_FILE = REPO_ROOT / "picollm" / "accelerated" / "data" / "identity_conversations.jsonl"
DEFAULT_MANIFEST = REPO_ROOT / "picollm" / "accelerated" / "data" / "identity_conversations.manifest.json"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def count_rows(data: bytes) -> int:
    return sum(1 for line in data.decode("utf-8").splitlines() if line.strip())


def load_manifest(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def verify_local_file(local_file: Path, expected_sha256: str, expected_rows: int) -> bytes:
    data = local_file.read_bytes()
    actual_sha256 = sha256_bytes(data)
    actual_rows = count_rows(data)

    if actual_sha256 != expected_sha256:
        raise SystemExit(
            f"Local identity dataset checksum mismatch: expected {expected_sha256}, got {actual_sha256}"
        )
    if actual_rows != expected_rows:
        raise SystemExit(f"Local identity dataset row-count mismatch: expected {expected_rows}, got {actual_rows}")

    print(f"local: ok ({actual_rows} rows, sha256={actual_sha256})")
    return data


def verify_hosted_asset(url: str, local_data: bytes, expected_sha256: str, expected_rows: int) -> None:
    data = fetch_hosted_asset(url)
    verify_hosted_bytes(data, expected_sha256, expected_rows, url)
    if data != local_data:
        raise SystemExit("Hosted identity asset bytes differ from the canonical repo-local file despite matching metadata")


def fetch_hosted_asset(url: str) -> bytes:
    try:
        request = Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 (compatible; picoLLM-identity-verifier/1.0)",
            },
        )
        with urlopen(request) as response:
            return response.read()
    except HTTPError as exc:
        raise SystemExit(f"Hosted identity asset returned HTTP {exc.code}: {url}") from exc
    except URLError as exc:
        raise SystemExit(f"Unable to fetch hosted identity asset: {url} ({exc.reason})") from exc


def verify_hosted_bytes(data: bytes, expected_sha256: str, expected_rows: int, url: str) -> None:
    actual_sha256 = sha256_bytes(data)
    actual_rows = count_rows(data)

    if actual_sha256 != expected_sha256:
        raise SystemExit(f"Hosted identity asset checksum mismatch: expected {expected_sha256}, got {actual_sha256}")
    if actual_rows != expected_rows:
        raise SystemExit(f"Hosted identity asset row-count mismatch: expected {expected_rows}, got {actual_rows}")
    print(f"hosted: ok ({actual_rows} rows, sha256={actual_sha256}, url={url})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify the canonical picoLLM identity dataset and hosted mirror.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST, help="Path to identity dataset manifest JSON")
    parser.add_argument("--local-file", type=Path, default=DEFAULT_DATA_FILE, help="Path to the canonical local identity dataset")
    parser.add_argument("--hosted-url", type=str, default="", help="Hosted identity asset URL; defaults to the manifest URL")
    parser.add_argument("--download-to", type=Path, default=None, help="Download the hosted asset to this path after verification")
    parser.add_argument("--local-only", action="store_true", help="Only verify the local canonical dataset")
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    expected_sha256 = manifest["sha256"]
    expected_rows = manifest["row_count"]
    hosted_url = args.hosted_url or manifest.get("hosted_mirror", {}).get("url", "")

    local_data = None
    if args.local_file.exists():
        local_data = verify_local_file(args.local_file, expected_sha256, expected_rows)
    elif not args.download_to:
        raise SystemExit(f"Local identity dataset does not exist: {args.local_file}")

    if args.local_only:
        return
    if not hosted_url:
        raise SystemExit("No hosted URL provided and none was found in the manifest")

    if args.download_to:
        data = fetch_hosted_asset(hosted_url)
        verify_hosted_bytes(data, expected_sha256, expected_rows, hosted_url)
        if local_data is not None and data != local_data:
            raise SystemExit("Hosted identity asset bytes differ from the canonical repo-local file despite matching metadata")
        args.download_to.parent.mkdir(parents=True, exist_ok=True)
        args.download_to.write_bytes(data)
        print(f"downloaded: ok ({args.download_to})")
        return

    assert local_data is not None
    verify_hosted_asset(hosted_url, local_data, expected_sha256, expected_rows)


if __name__ == "__main__":
    main()
