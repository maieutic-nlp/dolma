"""

Filters.

@kylel, @soldni

"""

from typing import Generator

import regex
import uniseg.wordbreak
from tokenizers import Regex, Tokenizer, pre_tokenizers

from dolma.core.data_types import DocResult, Document, Span, TextSlice
from dolma.core.registry import TaggerRegistry
from dolma.core.taggers import BaseTagger
from dolma.core.utils import split_paragraphs


@TaggerRegistry.add("bytes_length_v1")
class BytesLengthV1(BaseTagger):
    def predict(self, doc: Document) -> DocResult:
        score = len(doc.text.encode("utf-8"))
        return DocResult(doc=doc, spans=[Span(start=0, end=len(doc.text), type="bytes", score=score)])


@TaggerRegistry.add("doc_count_v1")
class DocCountLengthV1(BaseTagger):
    def predict(self, doc: Document) -> DocResult:
        return DocResult(doc=doc, spans=[Span(start=0, end=len(doc.text), type="docs", score=1)])


@TaggerRegistry.add("char_length_v1")
class CharLengthV1(BaseTagger):
    def predict(self, doc: Document) -> DocResult:
        score = len(doc.text)
        return DocResult(doc=doc, spans=[Span(start=0, end=len(doc.text), type="length", score=score)])


@TaggerRegistry.add("char_length_strip_ws_v1")
class CharLengthStripWsV1(BaseTagger):
    def predict(self, doc: Document) -> DocResult:
        score = len(doc.text.strip())
        return DocResult(doc=doc, spans=[Span(start=0, end=len(doc.text), type="length_no_ws", score=score)])


@TaggerRegistry.add("char_length_with_paragraphs_v1")
class CharLengthWithParagraphsV1(BaseTagger):
    def predict(self, doc: Document) -> DocResult:
        spans = [
            Span(start=p.start, end=p.end, type="paragraph", score=len(p.text)) for p in split_paragraphs(doc.text)
        ]
        spans.append(Span(start=0, end=len(doc.text), type="document", score=len(doc.text)))
        return DocResult(doc=doc, spans=spans)


@TaggerRegistry.add("whitespace_tokenizer_v1")
class WhitespaceLengthV1(BaseTagger):
    WHITESPACE_REGEX = regex.compile(r"\w+|[^\w\s]+")

    def predict(self, doc: Document) -> DocResult:
        score = len(self.WHITESPACE_REGEX.split(doc.text))
        return DocResult(doc=doc, spans=[Span(start=0, end=len(doc.text), type="length", score=score)])


@TaggerRegistry.add("whitespace_tokenizer_with_paragraphs_v1")
class WhitespaceLengthParagraphsV1(WhitespaceLengthV1):
    def predict(self, doc: Document) -> DocResult:
        spans = [
            Span(start=p.start, end=p.end, type="paragraph", score=len(self.WHITESPACE_REGEX.split(p.text)))
            for p in split_paragraphs(doc.text)
        ]
        spans.append(Span(start=0, end=len(doc.text), type="document", score=sum(s.score for s in spans)))
        return DocResult(doc=doc, spans=spans)


@TaggerRegistry.add("uniseg_length_v1")
class UnisegLengthV1(BaseTagger):
    def predict(self, doc: Document) -> DocResult:
        score = sum(1 for _ in uniseg.wordbreak.words(text)) if (text := doc.text.strip()) else 0
        return DocResult(doc=doc, spans=[Span(start=0, end=len(doc.text), type="length", score=score)])


@TaggerRegistry.add("uniseg_length_paragraphs_v1")
class UnisegParagraphsV1(BaseTagger):
    def do_split_paragraphs(self, text: str) -> Generator[TextSlice, None, None]:
        for paragraph in split_paragraphs(text, remove_empty=True):
            yield paragraph

    def predict(self, doc: Document) -> DocResult:
        spans = []

        for para in self.do_split_paragraphs(doc.text):
            # we ignore whitespace-only tokens when counting words
            para_length = sum(1 for w in uniseg.wordbreak.words(para.text.strip()) if w.strip())
            spans.append(Span(start=para.start, end=para.end, type="paragraph", score=para_length))

            # we have to record negative length because mixer can only work on greater than operations,
            # and we might want to drop paragraphs that are shorter than a certain length n, so we need
            # to filter on >= n
            spans.append(Span(start=para.start, end=para.end, type="negative_paragraph", score=-para_length))
        return DocResult(doc=doc, spans=spans)


@TaggerRegistry.add("uniseg_length_paragraphs_with_empty_v1")
class UnisegParagraphsWithEmptyV1(UnisegParagraphsV1):
    def do_split_paragraphs(self, text: str) -> Generator[TextSlice, None, None]:
        for paragraph in split_paragraphs(text, remove_empty=False):
            yield paragraph


@TaggerRegistry.add("uniseg_length_paragraphs_with_doc_length_v1")
class UnisegParagraphsWithDocLengthV1(UnisegParagraphsV1):
    def predict(self, doc: Document) -> DocResult:
        doc_results = super().predict(doc)
        pos_len = sum(s.score for s in doc_results.spans if s.type == "paragraph")
        neg_len = sum(s.score for s in doc_results.spans if s.type == "negative_paragraph")
        doc_results.spans.append(Span(start=0, end=len(doc.text), type="document", score=pos_len))
        doc_results.spans.append(Span(start=0, end=len(doc.text), type="negative_document", score=neg_len))
        return doc_results


@TaggerRegistry.add("olmo_pretokenizer_v1")
class OlmoPreTokenizerV1(BaseTagger):
    def __init__(self) -> None:
        self.pre_tokenizer = pre_tokenizers.Sequence(
            [
                # Split on all punctuation.
                pre_tokenizers.Split(
                    pattern=Regex(" ?[[:punct:]]"),
                    behavior="isolated",
                    invert=False,
                ),
                # Split up digits.
                pre_tokenizers.Split(
                    pattern=Regex(" ?\\d"),
                    behavior="isolated",
                    invert=False,
                ),
                pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True),
            ]
        )

    def predict(self, doc: Document) -> DocResult:
        score = len(self.pre_tokenizer.pre_tokenize_str(doc.text))
        return DocResult(doc=doc, spans=[Span(start=0, end=len(doc.text), type="length", score=score)])


@TaggerRegistry.add("olmo_pretokenizer_with_paragraphs_v1")
class OlmoPreTokenizerParagraphsV1(OlmoPreTokenizerV1):
    def predict(self, doc: Document) -> DocResult:
        spans = [
            Span(
                start=p.start, end=p.end, type="paragraph", score=len(self.pre_tokenizer.pre_tokenize_str(p.text))
            )
            for p in split_paragraphs(doc.text)
        ]
        spans.append(Span(start=0, end=len(doc.text), type="document", score=sum(s.score for s in spans)))
        return DocResult(doc=doc, spans=spans)


@TaggerRegistry.add("dolma_v1_tokenizer")
class DolmaV1Tokenizer(BaseTagger):
    TOKENIZER_NAME_OR_PATH = "allenai/gpt-neox-olmo-dolma-v1_5"

    def __init__(self) -> None:
        self.tokenizer = Tokenizer.from_pretrained(self.TOKENIZER_NAME_OR_PATH)
        super().__init__()

    def predict(self, doc: Document) -> DocResult:
        score = len(self.tokenizer.encode(text)) if (text := doc.text.strip()) else 0
        return DocResult(doc=doc, spans=[Span(start=0, end=len(doc.text), type="length", score=score)])


@TaggerRegistry.add("dolma_v2_tokenizer")
class DolmaV2Tokenizer(DolmaV1Tokenizer):
    TOKENIZER_NAME_OR_PATH = "allenai/dolma2-tokenizer"
