"""Sparse vectors of page slugs, for suggesting pages in place of missing ones.

`slug_trigrams` in rust_search/src/similar_paths.rs builds the same vectors for
queries, so the two must change together.
"""

import re
from collections import Counter

SLUG_VECTOR = "slug"


def _trigram_letter(char: str) -> int:
    if "a" <= char <= "z":
        return ord(char) - ord("a") + 1
    if "0" <= char <= "9":
        return ord(char) - ord("0") + 27
    return {"^": 37, "$": 38}.get(char, 0)


def slug_trigrams(page: str) -> tuple[list[int], list[float]]:
    """Each word of the last segment of `page`, padded as `^word$`, split into
    overlapping trigrams. A trigram's index is its letters in base 39, where
    a-z are 1-26, 0-9 are 27-36, `^` is 37, `$` is 38, and anything else is 0.
    Its value is how often it occurs."""
    slug = page.strip("/").rsplit("/", 1)[-1]
    counts: Counter[int] = Counter()
    for word in re.split(r"[-_.]", slug):
        if not word:
            continue
        letters = [_trigram_letter(char) for char in f"^{word.lower()}$"]
        for a, b, c in zip(letters, letters[1:], letters[2:]):
            counts[a * 39 * 39 + b * 39 + c] += 1
    indices = sorted(counts)
    return indices, [float(counts[i]) for i in indices]


if __name__ == "__main__":
    # The same vectors as the `encodes_slugs_as_trigrams` test in Rust.
    assert slug_trigrams("documentation/hy-1") == ([13181, 56614, 57407], [1.0, 1.0, 1.0])
    assert slug_trigrams("/blog/Aa_aa/") == ([1598, 56317], [2.0, 2.0])
    print("ok")
