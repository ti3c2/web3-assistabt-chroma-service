import datetime as dt
import logging
from functools import cached_property
from typing import List, Optional

from pydantic import BaseModel, ConfigDict

from .clean_text import cleanup_text
from .extract_data import extract_token_mentions

logger = logging.getLogger(__name__)


class TelegramMessage(BaseModel):
    username: str
    message_id: str
    datetime: str = ""
    content: str

    @cached_property
    def parsed_content(self) -> str:
        out = cleanup_text(self.content)
        logger.debug("Parsed content: \n%s", out)
        return out

    # @cached_property
    # def token_mentions(self) -> List[str]:
    #     return extract_token_mentions(self.parsed_content)

    model_config = ConfigDict(
        json_encoders={dt.datetime: lambda v: v.isoformat() if v else None}
    )
