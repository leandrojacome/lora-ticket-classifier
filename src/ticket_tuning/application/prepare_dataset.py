import hashlib
from dataclasses import dataclass

from ticket_tuning.domain.ticket import TicketExample


@dataclass(frozen=True)
class DatasetSplit:
    train: tuple[TicketExample, ...]
    evaluation: tuple[TicketExample, ...]


class DeterministicSplitStrategy:
    def __init__(self, evaluation_percent: int = 20) -> None:
        if evaluation_percent not in range(1, 100):
            raise ValueError("Evaluation percentage must be between 1 and 99")
        self._threshold = evaluation_percent

    def is_evaluation(self, example: TicketExample) -> bool:
        bucket = int(hashlib.sha256(example.text.encode()).hexdigest()[:8], 16) % 100
        return bucket < self._threshold


class PrepareDataset:
    def __init__(self, strategy: DeterministicSplitStrategy) -> None:
        self._strategy = strategy

    def execute(self, examples: list[TicketExample]) -> DatasetSplit:
        unique: dict[str, TicketExample] = {}
        for example in sorted(examples, key=lambda item: (item.text.lower(), item.text)):
            unique.setdefault(example.text.strip().lower(), example)
        ordered = sorted(unique.values(), key=lambda item: item.text)
        return DatasetSplit(
            tuple(item for item in ordered if not self._strategy.is_evaluation(item)),
            tuple(item for item in ordered if self._strategy.is_evaluation(item)),
        )
