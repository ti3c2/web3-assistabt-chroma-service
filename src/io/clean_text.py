import re
from typing import Callable, List

import emoji
import markdownify as md
from bs4 import BeautifulSoup


def remove_emojis(text: str) -> str:
    return emoji.replace_emoji(text, "")


def remove_telegram_links(text: str) -> str:
    # Remove @username mentions
    text = re.sub(r"\s?@[\w\d_]+", "", text)
    return text


def remove_urls(text: str) -> str:
    url_pattern = r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
    return re.sub(url_pattern, "", text)


def remove_hashtags(text: str) -> str:
    return re.sub(r"#[\S]+", "", text)


def remove_cashtags(text: str) -> str:
    """Remove $TAG style cashtags."""
    return re.sub(r"\$[A-Z]+", "", text)


def remove_whitespace(text: str) -> str:
    # Remove leading/trailing whitespace
    text = "\n".join(line.strip() for line in text.splitlines())
    # Remove multiple spaces
    text = re.sub(r"\w\s\s+", " ", text)
    # Strip
    text = text.strip()
    return text


def parse_html(text: str) -> str:
    return md.markdownify(text)


def parse_html_soup(text: str) -> str:  # NOTE: Probably not needed
    text = re.sub(r"<br\s*/>", "\n", text)
    text = BeautifulSoup(text, "lxml").text
    text = text.strip()
    return text


def get_cleanup_text(funcs: List[Callable[[str], str]]) -> Callable[[str], str]:
    def cleanup_text(text: str) -> str:
        for func in funcs:
            text = func(text)
        return text

    return cleanup_text


cleanup_text: Callable[[str], str] = get_cleanup_text(
    [
        parse_html_soup,
        remove_emojis,
        remove_telegram_links,
        remove_urls,
        remove_hashtags,
        remove_whitespace,
    ]
)


def test_sample_text():
    # Example usage and testing
    sample_text = """
    ✨ Hello @username! Check out   https://example.com my website
    and t.me/channel $BTC 👋
    Привет мир! 123 #hashtag
    """
    cleanup_text = get_cleanup_text(
        [
            remove_emojis,
            # remove_telegram_links,
            # remove_urls,
            # remove_hashtags,
            remove_whitespace,
        ]
    )

    print("Original text:")
    print(sample_text)
    print("\nCleaned text:")
    print(cleanup_text(sample_text))


if __name__ == "__main__":
    test_sample_text()
