from ticket_tuning.application.prepare_dataset import DeterministicSplitStrategy, PrepareDataset
from ticket_tuning.domain.ticket import TicketExample


def test_removes_duplicates_and_produces_stable_split() -> None:
    examples = [
        TicketExample("Invoice is wrong", "billing"),
        TicketExample("invoice is wrong", "billing"),
        TicketExample("Cannot sign in", "account"),
    ]
    use_case = PrepareDataset(DeterministicSplitStrategy(50))
    first = use_case.execute(examples)
    second = use_case.execute(list(reversed(examples)))
    assert first == second
    assert len(first.train) + len(first.evaluation) == 2


def test_rejects_unknown_label() -> None:
    try:
        TicketExample("Help", "sales")
    except ValueError as error:
        assert "allowed label" in str(error)
    else:
        raise AssertionError("Unknown labels must be rejected")
