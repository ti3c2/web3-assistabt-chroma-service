import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_PATH = Path(__file__).parents[2]

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


@dataclass
class NavigatorMixin:
    path_root: Path = PROJECT_PATH
    path_data: Path = path_root / "data"
    path_data_html: Path = path_data / "html"

    def find_file(
        self,
        fname: Union[str, Path],
        base: Optional[Path] = None,
    ) -> Optional[Path]:
        if (fpath := Path(fname)).exists():
            return fpath
        base = base or self.path_data
        files = list(base.rglob(f"*{fname}*", case_sensitive=False))
        if len(files) > 1:
            logger.warning(f"Multiple files found for '{fname}': {files}")
        for path in files:
            if path.is_file():
                logger.info(f"Found file for '{fname}': {path}")
                return path
        logger.error(f"No file found for '{fname}'")
        return None


class OpenAiConfigMixin:
    openai_api_key: Optional[str] = Field(None)
    openai_api_base: str = Field(default="https://api.openai.com/v1")
    openai_model: str = Field(default="gpt-4o")
    openai_temperature: float = 0.3


class ChromaDbMixin:
    chromadb_host: str = Field(default="localhost")
    chromadb_port: int = Field(default=6300)
    chromadb_api_port: int = Field(default=6400)


class TgParserMixin:
    path_session: Path = PROJECT_PATH / "session"
    tg_parser_host: str = Field(default="http://tg-parser-api")
    tg_parser_port: int = Field(default=8778)


class ProjectSettings(
    BaseSettings, NavigatorMixin, OpenAiConfigMixin, ChromaDbMixin, TgParserMixin
):
    # Model configuration
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # Logging level
    log_level: int = logging.DEBUG


# Create global settings instance
settings = ProjectSettings()  # pyright: ignore - using default arguments
logging.basicConfig(level=settings.log_level)
