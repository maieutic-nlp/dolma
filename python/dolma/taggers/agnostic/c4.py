import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Set

from dolma.core.data_types import DocResult, Document, Span
from dolma.core.registry import TaggerRegistry
from dolma.core.taggers import BaseTagger
from dolma.utils.language_config import get_language_config, get_spaceless_languages

logger = logging.getLogger(__name__)

def load_naughty_words(language: str) -> tuple[Set[str], Set[str]]:
    """
    Load a list of "naughty" words and phrases to flag as sensitive content.
    Tries to load a language-specific file; falls back to a language-agnostic version.
    """
    naughty_words_file = Path(__file__).parent / f"../../data/naughty_words_{language}.txt"
    if language == "agnostic" or not naughty_words_file.exists():
        if language != "agnostic":
            logger.warning(
                f"No naughty words file found for language '{language}'. "
                f"Falling back to default language-agnostic list."
            )
        naughty_words_file = Path(__file__).parent / f"../../data/naughty_words.txt"
    
    naughty_lines = naughty_words_file.absolute().open().read().splitlines()
    words = set(w for w in naughty_lines if " " not in w)
    phrases = set(w for w in naughty_lines if " " in w)

    return words, phrases

@dataclass
class C4Attributes:
    lines_with_no_ending_punctuation: List[Span]
    lines_with_too_few_words: List[Span]
    has_naughty_word: bool = False
    has_javascript: bool = False
    has_lorem_ipsum: bool = False
    has_curly_brace: bool = False
    line_count: int = 0
    character_count: int = 0

    def as_spans(self) -> List[Span]:
        spans = []
        spans.extend(self.lines_with_no_ending_punctuation)
        spans.extend(self.lines_with_too_few_words)
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
        self.language = language
        self.naughty_words, self.naughty_phrases = load_naughty_words(language)

        config = get_language_config(language)
        self.min_words_per_line = config["min_words_per_line"]
        self.eol_punctuation = config["eol_punctuation"]
    
    def predict(self, doc: Document) -> DocResult:
        spans: List[Span] = []
        text = doc.text.lower()

        if "{" in text:
            spans.append(Span(0, len(doc.text), type="has_curly_brace"))

        if "lorem ipsum" in text:
            spans.append(Span(0, len(doc.text), type="has_lorem_ipsum"))

        if "javascript" in text:
            spans.append(Span(0, len(doc.text), type="has_javascript"))

        if any(word in self.naughty_words for word in text.split()) or any(
            phrase in text for phrase in self.naughty_phrases
        ):
            spans.append(Span(0, len(doc.text), type="has_naughty_word"))
        
        start = count = 0
        for sent in text.split("\n"):
            end = start + len(sent)
            if end != len(text):
                # account for the newline
                end += 1

            # strip any trailing whitespace
            sent = sent.strip()

            if not sent.endswith(tuple(self.eol_punctuation)):
                spans.append(Span(start, end, type="lines_with_no_ending_punctuation"))
            
            # check word count if language uses spaces
            if self.language not in get_spaceless_languages():
                if len(sent.split()) < self.min_words_per_line:
                    spans.append(Span(start, end, type="lines_with_too_few_words"))

            count += 1
            start = end

        spans.append(Span(0, len(doc.text), type="line_count", score=count))
        return DocResult(doc=doc, spans=spans)
