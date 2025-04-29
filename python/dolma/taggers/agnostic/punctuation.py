import logging
import regex

from dolma.core.data_types import DocResult, Document, Span
from dolma.core.registry import TaggerRegistry
from dolma.core.taggers import BaseTagger
from dolma.core.utils import split_paragraphs
from dolma.utils.language_config import get_language_config

logger = logging.getLogger(__name__)

@TaggerRegistry.add("not_alphanum_paragraph_agnostic")
class NotAlphanumParagraphMonolingual(BaseTagger):
    def __init__(self, language: str = "en") -> None:
        config = get_language_config(language)
        self.re_has_alphanum = regex.compile(config["alphanum_regex"], regex.UNICODE)
        
        self.re_all_punctuation = regex.compile(
            r"^("
            r"\p{P}|"
            r"\s|"
            r"["
            "\U0001f300-\U0001f64f"
            "\U0001f680-\U0001f6ff"
            "\u2600-\u26ff\u2700-\u27bf"
            r"]+"
            r")+$",
            regex.UNICODE,
        )

    def predict(self, doc: Document) -> DocResult:
        spans = []

        for para in split_paragraphs(text=doc.text):
            if self.re_has_alphanum.search(para.text):
                continue

            if self.re_all_punctuation.search(para.text):
                spans.append(Span(start=para.start, end=para.end, type="all_punct", score=1))

        if not spans:
            spans.append(Span(start=0, end=len(doc.text), type="all_punct", score=0))

        return DocResult(doc=doc, spans=spans)
