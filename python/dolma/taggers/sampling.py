import random
from multiprocessing import current_process

from dolma.core.data_types import DocResult, Document, Span
from dolma.core.registry import TaggerRegistry
from dolma.core.taggers import BaseTagger


@TaggerRegistry.add("random_number_v1")
class RandomNumberTagger(BaseTagger):
    def __init__(self, seed: int = 1) -> None:
        assert seed > 0
        # we multiply the seed by the current process id to ensure that each
        # process has a different seed
        self.seed = ((current_process().pid or 0) + 1) * seed
        random.seed(self.seed)

    def predict(self, doc: Document) -> DocResult:
        score = random.random()
        return DocResult(doc=doc, spans=[Span(start=0, end=len(doc.text), type="random", score=score)])
