import re
from typing import List, Optional, Set

from typing_extensions import Tuple

TICKER_PATTERN = r"(?:st|w)?[A-Z]{2,10}"


def extract_token_single(text: str) -> List[str]:
    out = set()
    cashtag_pattern = rf"\$({TICKER_PATTERN})"
    for match in re.finditer(cashtag_pattern, text):
        symbol = match.group(1)
        out.add(symbol)
    return list(out)


def extract_token_pairs(text: str) -> List[Tuple[str, str]]:
    out = set()
    pair_pattern = rf"({TICKER_PATTERN})\s?[-/_]\s?({TICKER_PATTERN})"
    for match in re.finditer(pair_pattern, text):
        base_currency = match.group(1)
        quote_currency = match.group(2)
        out.add((base_currency, quote_currency))
    return list(out)


def extract_token_mentions(text: str) -> List[str]:
    out = set()
    out.update(extract_token_single(text))
    for base, quote in extract_token_pairs(text):
        out.update([base, quote])
    return list(out)


def test_extraction():
    sample_texts = [
        "Looking at $BTC and $ETH prices",
        "BTC/USDT is looking bullish while ETH-USDT is consolidating and APT / USDC is trading",
        "Trading $SOL and SOL_USDT pairs",
        "As for wrapped tokens, I have $wETH and $stTON",
        "$BTC $BTC BTC/USDT BTC-USDT",  # Test duplicate handling
        "Random text with no tokens",
    ]

    for text in sample_texts:
        print("\nOriginal text:", text)
        token_mentions = extract_token_mentions(text)
        print("Token mentions:")
        for mention in token_mentions:
            print(f"  Symbol: {mention}")


if __name__ == "__main__":
    test_extraction()
