import argparse
import json
from pathlib import Path

from ticket_tuning.application.prepare_dataset import DeterministicSplitStrategy, PrepareDataset
from ticket_tuning.domain.ticket import TicketExample


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and split ticket-classification JSONL")
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    examples = [TicketExample(**json.loads(line)) for line in args.source.read_text().splitlines()]
    split = PrepareDataset(DeterministicSplitStrategy()).execute(examples)
    args.output.mkdir(parents=True, exist_ok=True)
    for name, values in (("train", split.train), ("evaluation", split.evaluation)):
        (args.output / f"{name}.jsonl").write_text(
            "\n".join(json.dumps({"text": item.prompt()}) for item in values) + "\n"
        )


if __name__ == "__main__":
    main()
