import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Set

from dolma.core.data_types import DocResult, Document, Span
from dolma.core.registry import TaggerRegistry
from dolma.core.taggers import BaseTagger

logger = logging.getLogger(__name__)

def load_naughty_words(language: str) -> Set[str]:
    naughty_words_file = Path(__file__).parent / f"../../../data/naughty_words_{language}.txt"
    if language in {"default", "agnostic"} or not naughty_words_file.exists():
        if language not in {"default", "agnostic"}:
            logger.warning(
                f"No naughty words file found for language '{language}'. "
                f"Falling back to default language-agnostic list."
            )
        naughty_words_file = Path(__file__).parent / f"../../../data/naughty_words.txt"
    
    naughty_lines = naughty_words_file.absolute().open().read().splitlines()
    words = set(w for w in naughty_lines if " " not in w)
    phrases = set(w for w in naughty_lines if " " in w)

    return words, phrases

@dataclass
class C4Attributes:
    has_naughty_word: bool = False
    has_javascript: bool = False
    has_lorem_ipsum: bool = False
    has_curly_brace: bool = False
    line_count: int = 0
    character_count: int = 0

    def as_spans(self) -> List[Span]:
        spans = []
        if self.has_naughty_word:
            spans.append(Span(0, self.character_count, type="has_naughty_word"))
        if self.has_javascript:
            spans.append(Span(0, self.character_count, type="has_javascript"))
        if self.has_lorem_ipsum:
            spans.append(Span(0, self.character_count, type="has_lorem_ipsum"))
        if self.has_curly_brace:
            spans.append(Span(0, self.character_count, type="has_curly_brace"))
        spans.append(Span(0, self.character_count, type="line_count", score=self.line_count))
        return spans

@TaggerRegistry.add("mc4")
class MC4Tagger(BaseTagger):
    def __init__(self, language: str = "en"):
        super().__init__()
        self.naughty_words, self.naughty_phrases = load_naughty_words(language)
    
    def predict(self, doc: Document) -> DocResult:
        spans: List[Span] = []
        text = doc.text.lower()
        lines = text.split("\n")
        valid_line_count = sum(len(line) >= 200 for line in lines)

        if valid_line_count < 3:
            spans.append(Span(0, len(doc.text), type="filtered_by_line_length"))

        # if "{" in text:
        #     spans.append(Span(0, len(doc.text), type="has_curly_brace"))

        # if "lorem ipsum" in text:
        #     spans.append(Span(0, len(doc.text), type="has_lorem_ipsum"))

        # if "javascript" in text:
        #     spans.append(Span(0, len(doc.text), type="has_javascript"))

        if any(word in self.naughty_words for word in text.split()) or any(
            phrase in text for phrase in self.naughty_phrases
        ):
            spans.append(Span(0, len(doc.text), type="has_naughty_word"))

        spans.append(Span(0, len(doc.text), type="line_count", score=valid_line_count))
        return DocResult(doc=doc, spans=spans)