import datetime as dt
from functools import cached_property
from typing import List, Optional

from pydantic import BaseModel, ConfigDict

from .clean_text import cleanup_text
from .extract_data import extract_token_mentions


class TelegramMessage(BaseModel):
    username: str
    message_id: str
    datetime: Optional[dt.datetime] = None
    content: str

    @cached_property
    def parsed_content(self) -> str:
        return cleanup_text(self.content)

    # @cached_property
    # def token_mentions(self) -> List[str]:
    #     return extract_token_mentions(self.parsed_content)

    model_config = ConfigDict(
        json_encoders={dt.datetime: lambda v: v.isoformat() if v else None}
    )
