from typing import List, Tuple

from dolma.core.data_types import DocResult, Document, Span
from dolma.core.ft_tagger import BaseFastTextTagger
from dolma.core.registry import TaggerRegistry
from dolma.core.taggers import BaseTagger
from dolma.core.utils import split_paragraphs
from dolma.utils.language_config import get_language_config


class BaseLanguageTagger(BaseTagger):
    INCLUDE_NEGATIVE = True
    PREDICT_ON_PARAGRAPHS = False

    def __init__(self, language: str = "en"):
        self.language = language
        config = get_language_config(language)
        self.iso639_1 = config["iso639_1"]

    def predict_text(self, text: str) -> List[Tuple[str, float]]:
        return []

    def make_negative(self, spans: List[Span]) -> List[Span]:
        return [
            Span(start=span.start, end=span.end, type=f"not_{span.type}", score=1.0 - span.score) for span in spans
        ]

    def predict_doc(self, doc: Document) -> DocResult:
        spans = [
            Span(start=0, end=len(doc.text), type=str(lang), score=score)
            for lang, score in self.predict_text(doc.text)
        ]
        return DocResult(doc=doc, spans=spans)

    def predict_paragraph(self, doc: Document) -> DocResult:
        paragraphs = split_paragraphs(doc.text)
        spans: List[Span] = []
        for paragraph in paragraphs:
            spans.extend(
                Span(start=paragraph.start, end=paragraph.end, type=lang, score=score)
                for lang, score in self.predict_text(paragraph.text)
            )
        return DocResult(doc=doc, spans=spans)

    def predict(self, doc: Document) -> DocResult:
        doc_result = self.predict_paragraph(doc) if self.PREDICT_ON_PARAGRAPHS else self.predict_doc(doc)
        if self.INCLUDE_NEGATIVE:
            doc_result.spans.extend(self.make_negative(doc_result.spans))
        return doc_result


class FastTextAllLanguagesDocumentTagger(BaseLanguageTagger, BaseFastTextTagger):
    MODEL_PATH = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
    INCLUDE_NEGATIVE = False
    PREDICT_ON_PARAGRAPHS = False

    def __init__(self, language: str = "en"):
        BaseLanguageTagger.__init__(self, language=language)
        BaseFastTextTagger.__init__(self, model_path=self.MODEL_PATH, model_mode=self.DOCUMENT_LEVEL_TAGGER)

    def predict_text(self, text: str) -> List[Tuple[str, float]]:
        preds = self.classifier.predict(text.lower().replace("\n", " ").strip(), k=-1)
        return [(label.replace("__label__", ""), float(score)) for label, score in zip(*preds)]


class FastTextAgnosticLanguageDocumentTagger(FastTextAllLanguagesDocumentTagger):
    INCLUDE_NEGATIVE = True
    PREDICT_ON_PARAGRAPHS = False

    def predict_text(self, text: str) -> List[Tuple[str, float]]:
        preds = super().predict_text(text)
        filtered_preds = [(lang, score) for lang, score in preds if lang == self.iso639_1] or [(self.iso639_1, 0.0)]
        return filtered_preds  # pyright: ignore


class FastTextAgnosticLanguageParagraphTagger(FastTextAgnosticLanguageDocumentTagger):
    INCLUDE_NEGATIVE = True
    PREDICT_ON_PARAGRAPHS = True


def add_global_language_score_from_slice_score(result: DocResult, iso639_1: str) -> DocResult:
    """
    Compute the overall document-level score for the target language by
    weighting each matching span's confidence score by its character length,
    then normalizing by the total document length.
    """
    try:
        doc_lang_score = sum((s.end - s.start) * s.score for s in result.spans if s.type == iso639_1) / len(
            result.doc.text
        )
        doc_not_lang_score = 1 - doc_lang_score
    except ZeroDivisionError:
        doc_lang_score = doc_not_lang_score = 0.0

    doc_level = (
        Span(start=0, end=len(result.doc.text), type=f"doc_{iso639_1}", score=doc_lang_score),
        Span(start=0, end=len(result.doc.text), type=f"doc_not_{iso639_1}", score=doc_not_lang_score),
    )
    result.spans.extend(doc_level)
    return result


@TaggerRegistry.add("ft_lang_id_agnostic_paragraph_with_doc_score")
class FastTextAgnosticLanguageParagraphWithDocScoreTagger(FastTextAgnosticLanguageParagraphTagger):
    def __init__(self, language: str = "en"):
        super().__init__(language=language)
    
    def predict(self, doc: Document) -> DocResult:
        doc_result = super().predict(doc)
        doc_result = add_global_language_score_from_slice_score(doc_result, self.iso639_1)
        return doc_result
