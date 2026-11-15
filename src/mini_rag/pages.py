"""Page selections such as "3-7,9,14-" for commands that filter by page."""

from typing import NamedTuple


class PageRange(NamedTuple):
    """1-based inclusive page range; last=None means "to the end"."""

    first: int
    last: int | None

    def includes(self, page: int) -> bool:
        return self.first <= page and (self.last is None or page <= self.last)

    def overlaps(self, first: int, last: int) -> bool:
        """True when the range shares at least one page with first..last."""
        return self.first <= last and (self.last is None or first <= self.last)


def parse_page_ranges(spec: str) -> tuple[PageRange, ...]:
    """Parse e.g. "3-7,9,14-" into 1-based inclusive ranges; "-M" starts at page 1."""
    ranges: list[PageRange] = []
    for item in spec.split(","):
        first, sep, last = (part.strip() for part in item.strip().partition("-"))
        if not (first or last) or not all(p.isdecimal() for p in (first, last) if p):
            raise ValueError(f"invalid page range {item.strip()!r} in {spec!r}")
        page_range = PageRange(
            int(first) if first else 1,
            (int(last) if last else None) if sep else int(first),
        )
        if page_range.first < 1 or (
            page_range.last is not None and page_range.last < page_range.first
        ):
            raise ValueError(f"invalid page range {item.strip()!r} in {spec!r}")
        ranges.append(page_range)
    return tuple(ranges)
