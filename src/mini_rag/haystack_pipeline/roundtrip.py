import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from haystack import Pipeline
from rich_struct import print_json

from mini_rag.haystack_pipeline.components import build_pipeline


def build_save_run(text: str, save_path: Path) -> str:
    pipeline = build_pipeline()
    with open(save_path, mode="w", encoding="utf-8") as yaml_file:
        pipeline.dump(yaml_file)

    result = pipeline.run(
        data={"text_stats": {"text": text}, "normalizer": {"text": text}},
        include_outputs_from={"normalizer"},
    )
    return json.dumps(result, sort_keys=True)


def load_run(text: str, load_path: Path) -> str:
    with open(load_path, mode="r", encoding="utf-8") as yaml_file:
        pipeline = Pipeline.load(
            yaml_file, allowed_modules=["mini_rag.haystack_pipeline.components"], unsafe=False
        )

    result = pipeline.run(
        data={"text_stats": {"text": text}, "normalizer": {"text": text}},
        include_outputs_from={"normalizer"},
    )
    return json.dumps(result, sort_keys=True)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="""
Run in two modes:
    --save - builds and saves the pipeline, then runs it with the provided text
    --load - loads the saved pipeline, then runs it with the provided text
"""
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--load", type=str, help="Where from load pipeline yaml")
    group.add_argument("--save", type=str, help="Where to save pipeline yaml")
    parser.add_argument(
        "text",
        nargs="?",
        default="One   two\nthree",
        help="Text to analyze (default: %(default)s)",
    )

    args = parser.parse_args(argv)
    if args.save is not None:
        print_json(build_save_run(args.text, Path(args.save)))
    elif args.load is not None:
        print_json(load_run(args.text, Path(args.load)))


if __name__ == "__main__":
    main()
