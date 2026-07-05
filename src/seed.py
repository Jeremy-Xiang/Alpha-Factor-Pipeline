"""seed.py — collision-resistant deterministic seed from a ticker string,
same utility used across every project in this series."""

import zlib


def stable_seed(s: str) -> int:
    return zlib.crc32(s.encode()) % (2**32)
