from dataclasses import dataclass

ALLOWED_LABELS = frozenset({"billing", "technical", "account"})


@dataclass(frozen=True)
class TicketExample:
    text: str
    label: str

    def __post_init__(self) -> None:
        if not self.text.strip() or self.label not in ALLOWED_LABELS:
            raise ValueError("A non-empty ticket and an allowed label are required")

    def prompt(self) -> str:
        return f"Classify the support ticket.\nTicket: {self.text}\nCategory: {self.label}"
