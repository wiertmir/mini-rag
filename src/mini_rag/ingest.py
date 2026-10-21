"""Split PDF documents listed in the manifest into chunks."""

import argparse
from collections.abc import Sequence

from mini_rag.config import HelpFormatter, IngestConfig, load_config


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m mini_rag.ingest",
        description=__doc__,
        formatter_class=HelpFormatter,
    )
    config = load_config(IngestConfig, argv, parser)
    print(config.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
