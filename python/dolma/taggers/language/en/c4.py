import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Set

from dolma.core.data_types import DocResult, Document, Span
from dolma.core.registry import TaggerRegistry
from dolma.core.taggers import BaseTagger

MIN_WORDS_PER_LINE = 3
EOL_PUNCTUATION = {".", "?", "!", '"'}

def load_naughty_words(language: str) -> Set[str]:
    naughty_words_file = Path(__file__).parent / f"../data/naughty_words_{language}.txt"
    if not naughty_words_file.exists():
        raise ValueError(f"Naughty words file for language '{language}' not found: {naughty_words_file}")
    
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


def get_attributes(text: str, naughty_words: Set[str], naughty_phrases: Set[str]) -> C4Attributes:
    attrs = C4Attributes([], [])
    attrs.character_count = len(text)
    try:
        lines = text.split("\n")
        attrs.line_count = len(lines)
        offset = 0
        for line_no in range(0, len(lines)):
            original_line = lines[line_no]
            end_offset = offset + len(original_line)
            if line_no < len(lines) - 1:
                end_offset += 1
            line = original_line.lower().strip()
            if not line.endswith((".", "?", "!", '"')):
                attrs.lines_with_no_ending_punctuation.append(
                    Span(offset, end_offset, type="lines_with_no_ending_punctuation")
                )
            words = line.split()
            if len(words) < MIN_WORDS_PER_LINE:
                attrs.lines_with_too_few_words.append(Span(offset, end_offset, type="lines_with_too_few_words"))
            if any(word in naughty_words for word in words) or any(phrase in line for phrase in naughty_phrases):
                attrs.has_naughty_word = True
            if any(word == "javascript" for word in words):
                attrs.has_javascript = True
            if "lorem ipsum" in line:
                attrs.has_lorem_ipsum = True
            if "{" in line:
                attrs.has_curly_brace = True
            offset = end_offset
    except Exception:
        logging.exception(f"Error parsing text: {text[:200]}")

    return attrs


@TaggerRegistry.add("c4_v1")
class C4Tagger(BaseTagger):
    def __init__(self, language: str = "en"):
        super().__init__()
        self.naughty_words, self.naughty_phrases = load_naughty_words(language)

    def predict(self, doc: Document) -> DocResult:
        attrs = get_attributes(doc.text, self.naughty_words, self.naughty_phrases)
        result = DocResult(doc=doc, spans=attrs.as_spans())
        return result


@TaggerRegistry.add("c4_v2")
class FasterC4Tagger(BaseTagger):
    def __init__(self, language: str = "en"):
        super().__init__()
        self.naughty_words, self.naughty_phrases = load_naughty_words(language)

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

            if not sent.endswith((".", "?", "!", '"')):
                spans.append(Span(start, end, type="lines_with_no_ending_punctuation"))

            if len(sent.split()) < MIN_WORDS_PER_LINE:
                spans.append(Span(start, end, type="lines_with_too_few_words"))

            count += 1
            start = end

        spans.append(Span(0, len(doc.text), type="line_count", score=count))
        return DocResult(doc=doc, spans=spans)

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