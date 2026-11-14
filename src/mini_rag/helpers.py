from types import TracebackType

from rich.console import Console
from rich.highlighter import ReprHighlighter
from rich.panel import Panel
from rich.text import Text


class ErrorHandler:
    def __init__(self, *handled_exceptions: type[Exception]) -> None:
        self._handled_exceptions = handled_exceptions

    def __enter__(self) -> None:
        return None

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        if exc_type is None or exc is None:
            return False
        if not issubclass(exc_type, Exception):
            return False

        console = Console(stderr=True)
        if isinstance(exc, self._handled_exceptions):
            message = ReprHighlighter()(Text(str(exc)))
            console.print(Panel(message, title=exc_type.__name__, border_style="red", expand=False))
        else:
            console.print_exception(show_locals=False)

        raise SystemExit(1) from None
