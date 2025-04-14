import logging
import re
from collections import Counter
from dataclasses import dataclass
from statistics import median
from tokenizers import Tokenizer
from typing import Counter as CounterType
from typing import List, Tuple, Union

from dolma.core.data_types import DocResult, Document, Span
from dolma.core.registry import TaggerRegistry
from dolma.core.taggers import BaseTagger

def robust_median(values: List[Union[int, float]]) -> float:
    if not values:
        return 0.0
    return float(median(values))

@dataclass
class GopherAttributes:
    fraction_of_characters_in_most_common_ngram: List[Tuple[int, float]]
    fraction_of_characters_in_duplicate_ngrams: List[Tuple[int, float]]
    character_count: int = 0
    token_count: int = 0
    median_token_length: float = 0.0
    fraction_of_tokens_with_alpha_character: float = 0.0
    fraction_of_duplicate_lines: float = 0.0
    fraction_of_characters_in_duplicate_lines: float = 0.0

    def as_spans(self) -> List[Span]:
        spans = []
        spans.extend(
            [
                Span(
                    0,
                    self.character_count,
                    f"fraction_of_characters_in_most_common_{n}grams",
                    v,
                )
                for n, v in self.fraction_of_characters_in_most_common_ngram
            ]
        )
        spans.extend(
            [
                Span(
                    0,
                    self.character_count,
                    f"fraction_of_characters_in_duplicate_{n}grams",
                    v,
                )
                for n, v in self.fraction_of_characters_in_duplicate_ngrams
            ]
        )
        spans.append(
            Span(
                0,
                self.character_count,
                type="character_count",
                score=self.character_count,
            )
        )
        spans.append(Span(0, self.character_count, type="token_count", score=self.token_count))
        spans.append(
            Span(
                0,
                self.character_count,
                type="median_token_length",
                score=self.median_token_length,
            )
        )
        spans.append(
            Span(
                0,
                self.character_count,
                type="fraction_of_tokens_with_alpha_character",
                score=self.fraction_of_tokens_with_alpha_character,
            )
        )
        spans.append(
            Span(
                0,
                self.character_count,
                type="fraction_of_duplicate_lines",
                score=self.fraction_of_duplicate_lines,
            )
        )
        spans.append(
            Span(
                0,
                self.character_count,
                type="fraction_of_characters_in_duplicate_lines",
                score=self.fraction_of_characters_in_duplicate_lines,
            )
        )
        return spans

def all_ngram_counts(tokens) -> List[Tuple[int, CounterType[Tuple[str, ...]]]]:
    return [(n, Counter(list(zip(*[tokens[i:] for i in range(n)])))) for n in range(2, 11)]

@TaggerRegistry.add("gopher_agnostic")
class AgnosticGopherTagger(BaseTagger):
    def __init__(self, tokenizer: str = "xlm-roberta-base"):
        super().__init__()
        self.tokenizer = Tokenizer.from_pretrained(tokenizer)

    def predict(self, doc: Document) -> DocResult:
        attrs = self.get_agnostic_attributes(doc.text, ignore_empty_lines=True)
        return DocResult(doc=doc, spans=attrs.as_spans())

    def get_agnostic_attributes(self, text: str, ignore_empty_lines: bool = False) -> GopherAttributes:
        attrs = GopherAttributes([], [])
        attrs.character_count = len(text)
        if attrs.character_count == 0:
            return attrs

        tokens = self.tokenizer.encode(sequence=text, add_special_tokens=False).tokens
        token_count = len(tokens)
        character_count = sum(len(token) for token in tokens)

        attrs.token_count = token_count
        attrs.median_token_length = robust_median([len(token) for token in tokens])

        attrs.fraction_of_tokens_with_alpha_character = sum(
            1 for token in tokens if any(c.isalnum() for c in token)
        ) / max(token_count, 1)

        all_counts = all_ngram_counts(tokens)
        count_most_common_ngrams = {2, 3, 4}
        for n, ngram_counts in all_counts:
            if not ngram_counts:
                continue
            if n in count_most_common_ngrams:
                most_common_ngram, count = ngram_counts.most_common(1)[0]
                value = count * sum(len(w) for w in most_common_ngram) / max(character_count, 1)
                attrs.fraction_of_characters_in_most_common_ngram.append((n, value))
            else:
                ng_char_count = sum(count * sum(len(w) for w in ng) for ng, count in ngram_counts.items())
                value = sum(
                    count * sum(len(w) for w in ng) for ng, count in ngram_counts.items() if count > 1
                ) / max(ng_char_count, 1)
                attrs.fraction_of_characters_in_duplicate_ngrams.append((n, value))

        # NOTE: This assumes newlines are meaningful. In some languages (e.g. Chinese, Japanese),
        # newline characters may not correspond to natural sentence/paragraph boundaries.
        if ignore_empty_lines:
            lines = re.split(r"\n+", text)
        else:
            lines = text.split("\n")
        
        line_count = len(lines)
        line_counts = Counter(lines)
        attrs.fraction_of_duplicate_lines = sum(count for line, count in line_counts.items() if count > 1) / max(
            line_count, 1
        )
        attrs.fraction_of_characters_in_duplicate_lines = sum(
            len(line) * count for line, count in line_counts.items() if count > 1
        ) / max(character_count, 1)

        return attrs