"""Verify immutable artifacts in the sanitized research archive."""
from __future__ import annotations
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
manifest = ROOT / 'metadata' / 'SHA256SUMS.txt'
count=0
for line in manifest.read_text(encoding='utf-8').splitlines():
    digest, filename = line.split('  ',1)
    path=ROOT/filename
    if not path.is_file():
        raise SystemExit(f'Missing archive member: {filename}')
    actual=hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != digest:
        raise SystemExit(f'Checksum mismatch for: {filename}')
    count+=1
print(f'PASS: {count} archived files match their recorded SHA-256 digests.')
