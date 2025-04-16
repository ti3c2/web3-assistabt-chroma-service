import datetime as dt
import logging
import re
from functools import cached_property
from pathlib import Path
from typing import List, Optional, Union
from zoneinfo import ZoneInfo

from bs4 import BeautifulSoup
from pydantic import BaseModel, ConfigDict, model_validator

from ..config.settings import settings
from .models import TelegramMessage

logger = logging.getLogger(__name__)


def parse_datetime(date_str: str) -> Optional[dt.datetime]:
    """Parse datetime string with timezone information."""
    try:
        # Extract components from string like "02.01.2025 18:43:24 UTC+03:00"
        match = re.match(
            r"(\d{2}\.\d{2}\.\d{4})\s+(\d{2}:\d{2}:\d{2})\s+UTC([+-]\d{2}):(\d{2})",
            date_str,
        )
        if match:
            date_part, time_part, tz_hours, tz_minutes = match.groups()

            # Parse the local datetime
            local_dt = dt.datetime.strptime(
                f"{date_part} {time_part}", "%d.%m.%Y %H:%M:%S"
            )

            # Create timezone offset
            tz_offset = (
                int(tz_hours) * 3600 + int(tz_minutes) * 60
            )  # Convert to seconds
            timezone = ZoneInfo("UTC")

            # Convert to UTC
            utc_dt = local_dt.replace(
                tzinfo=ZoneInfo(f"Etc/GMT{-int(tz_hours):+d}")
            ).astimezone(timezone)

            return utc_dt
    except (ValueError, TypeError):
        pass
    return None


def parse_tg_html(html_content: str, channel: str) -> list[TelegramMessage]:
    soup = BeautifulSoup(html_content, "lxml")
    messages = []

    # Find all message divs (both default and service messages)
    message_divs = soup.find_all(
        "div", class_=["message default clearfix", "message service"]
    )

    for message in message_divs:
        # Get message ID
        message_id = message.get("id")

        # Get datetime
        date_div = message.find("div", class_="pull_right date details")
        message_datetime = None
        if date_div:
            date_str = date_div.get("title", "")
            if date_str:
                message_datetime = parse_datetime(date_str)

        # Get text content
        text_div = message.find("div", class_="text")
        logger.debug(f"Text content: {text_div}")
        if text_div:
            text_div_html = str(text_div)
            # text_content = md.markdownify(text_div_html)
            text_content = str(text_div)
        else:
            # For service messages (dates)
            body_details = message.find("div", class_="body details")
            if body_details:
                text_content = body_details.get_text(strip=True)
            else:
                text_content = ""

        # Create TelegramMessage instance
        telegram_message = TelegramMessage(
            username=channel,
            message_id=message_id,
            datetime=message_datetime,
            content=text_content,
        )
        messages.append(telegram_message)

    return messages


def parse_tg_files(file_paths: List[Union[Path, str]]) -> List[TelegramMessage]:
    messages = []
    for file in file_paths:
        file = settings.find_file(str(file))
        if file:
            content = file.read_text()
            channel_name = file.stem.split("__")[0]
            messages.extend(parse_tg_html(content, channel_name))
        else:
            logger.warning(f"File not found: {file}")
    return messages


def main():
    import argparse

    import markdownify as md

    argparser = argparse.ArgumentParser()
    argparser.add_argument("-d", "--debug", action="store_true")
    argparser.add_argument("-p", "--print-content", action="store_true")
    argparser.add_argument("-f", "--files", nargs="+", type=str)
    args = argparser.parse_args()

    if args.debug:
        logger.setLevel(logging.DEBUG)

    files = args.files or list(settings.path_data_html.glob("*.html"))
    print(files)
    messages = parse_tg_files(files)

    for message in messages:
        if args.debug:
            continue

        print("\nChannel:", message.username)
        print("Message ID:", message.message_id)
        print("Datetime:", message.datetime)
        if args.print_content:
            print("\nText:\n", md.markdownify(message.content))
        print("Tokens:", message.token_mentions)
        print("\nParsed content:\n", message.parsed_content)
        print("-" * 50)


if __name__ == "__main__":
    main()
