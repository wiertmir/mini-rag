"""Application settings.

Precedence (highest first): CLI arguments, environment variables, ``.env``,
TOML config file (``config.toml`` or the path in ``MINI_RAG_CONFIG``),
field defaults.

In the TOML file, top-level keys are shared by all commands, and a command's
table (e.g. ``[ingest]``, set via ``toml_table_header``) overrides them.
"""

import argparse
import copy
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any, ClassVar, Self, cast

from pydantic import Field, ValidationError, create_model, model_validator
from pydantic_settings import (
    BaseSettings,
    CliSettingsSource,
    CliSuppress,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)
from rich_argparse import RichHelpFormatter

CONFIG_FILE_ENV_VAR = "MINI_RAG_CONFIG"


class HelpFormatter(RichHelpFormatter):
    """Colored help; pydantic-settings already appends ``(default: ...)``."""

    highlights: ClassVar[list[str]] = [
        *RichHelpFormatter.highlights,
        r"\(default: (?P<default>[^)]*)\)",
    ]


class Config(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
        frozen=True,
        cli_kebab_case=True,
    )

    data_dir: Path = Field(
        default=Path("data"), description="directory with the input data"
    )
    tokenizer_model: str = Field(
        default="text-embedding-3-small", description="model used to tokenize text"
    )
    # Not shown in --help, but still settable via CLI, env or config file.
    log_folder: CliSuppress[Path] = Field(
        default=Path("logs"), description="directory for log files"
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        config_file = os.getenv(CONFIG_FILE_ENV_VAR, "config.toml")
        sources = [init_settings, env_settings, dotenv_settings]
        table = settings_cls.model_config.get("toml_table_header", ())
        if table:
            try:
                sources.append(
                    TomlConfigSettingsSource(settings_cls, config_file, table)
                )
            except KeyError:
                pass  # the file has no table for this command
        sources.append(TomlConfigSettingsSource(settings_cls, config_file, ()))
        return tuple(sources)


class IngestConfig(Config):
    model_config = SettingsConfigDict(toml_table_header=("ingest",))

    manifest: Path = Field(
        default=Path("data/documents.csv"),
        description="CSV manifest describing documents",
    )
    pdf_dir: Path = Field(
        default=Path("data/pdfs"), description="directory with PDF files"
    )
    output: Path = Field(
        default=Path("index/chunks.jsonl"), description="output JSONL file with chunks"
    )
    chunk_size: int = Field(default=800, description="chunk size in tokens")
    chunk_overlap: int = Field(
        default=120, description="overlap between chunks in tokens"
    )

    @model_validator(mode="after")
    def _check_overlap(self) -> Self:
        if not 0 <= self.chunk_overlap < self.chunk_size:
            raise ValueError("chunk_overlap must be >= 0 and < chunk_size")
        return self


def load_config[C: Config](
    config_cls: type[C],
    argv: Sequence[str] | None = None,
    parser: argparse.ArgumentParser | None = None,
) -> C:
    """Load settings, letting CLI arguments override env vars and the config file.

    Pass ``parser`` to add the settings options to a program's own parser.
    ``--help`` shows the values resolved from env vars / config file as defaults.
    """
    resolved = config_cls()

    # Help text takes defaults from the model fields, so build a CLI-only model
    # whose defaults are the already resolved (env / file) values.
    fields: dict[str, Any] = {}
    for name, field in config_cls.model_fields.items():
        # Copy to keep metadata such as CliSuppress; only the default changes.
        info = copy.copy(field)
        info.default = getattr(resolved, name)
        info.default_factory = None
        fields[name] = (field.annotation, info)
    cli_model = cast(
        type[BaseSettings],
        create_model(config_cls.__name__, __base__=config_cls, **fields),
    )
    parser = parser or argparse.ArgumentParser(formatter_class=HelpFormatter)
    cli_source: CliSettingsSource[Any] = CliSettingsSource(
        cli_model, root_parser=parser, cli_parse_args=False
    )
    overrides: dict[str, Any] = cli_source(
        args=list(argv) if argv is not None else True
    )()
    try:
        return config_cls(**overrides)
    except ValidationError as exc:
        parser.error(str(exc))
