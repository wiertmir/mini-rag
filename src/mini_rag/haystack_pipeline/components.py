import re
from collections.abc import Sequence

from haystack import Pipeline, component
from rich_struct import print_obj


@component
class TextStats:
    """Counts words and lines in text."""

    @component.output_types(
        word_count=int,
        line_count=int,
        char_count=int,
    )
    def run(self, text: str) -> dict:
        return {
            "word_count": len(text.split()),
            "line_count": len(text.strip().splitlines()),
            "char_count": len(text),
        }


multiple_whitespace_pattern = re.compile(r" +")


@component
class WhitespaceNormalizer:
    """Removes duplicate spaces"""

    @component.output_types(normalized_text=str)
    def run(self, text: str) -> dict:
        clean_text = multiple_whitespace_pattern.sub(" ", text)
        return {"normalized_text": clean_text}


def build_pipeline() -> Pipeline:
    pipeline = Pipeline()

    pipeline.add_component("text_stats", TextStats())
    pipeline.add_component("normalizer", WhitespaceNormalizer())
    pipeline.add_component("normalized_text_stats", TextStats())

    pipeline.connect("normalizer", "normalized_text_stats")
    return pipeline


def main(argv: Sequence[str] | None = None) -> None:
    text = """
Nie  losowe — dwa małe komponenty demonstracyjne  z przewidywalnym wynikiem. Na przykład pierwszy normalizuje odstępy  w tekście, drugi
dodaje stały prefiks. Bez  logiki RAG.
"""

    pipeline = build_pipeline()

    result = pipeline.run(
        data={"text_stats": {"text": text}, "normalizer": {"text": text}},
        include_outputs_from={"normalizer"},
    )

    print("\nNon-modified text:")
    print_obj(result["text_stats"])

    print("\nNormalized text:")
    print_obj(result["normalized_text_stats"])


if __name__ == "__main__":
    main()
