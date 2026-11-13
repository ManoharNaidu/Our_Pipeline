"""Tokenizers used to count and truncate text."""

from __future__ import annotations

from types import SimpleNamespace


class WordTokenizer:
    """Counts words instead of model tokens, for tests with mock models."""

    def __call__(self, text: str, add_special_tokens: bool = False):
        return SimpleNamespace(input_ids=text.split())

    def encode(self, text: str, add_special_tokens: bool = False) -> list[str]:
        return text.split()

    def decode(self, tokens: list[str], skip_special_tokens: bool = True) -> str:
        return " ".join(tokens)


def load_tokenizer(cfg: dict):
    """The classifier's tokenizer, so token counts match what the classifier reads."""
    if cfg["classifier"].get("backend") == "mock":
        return WordTokenizer()
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(cfg["classifier"]["model_id"], revision=cfg["classifier"].get("revision"))
