# LoRA Fine-Tuning Pipeline for Support Tickets

This project prepares auditable support-ticket data and provides an optional PEFT/LoRA training adapter for a causal language model. The production problem is ticket triage into billing, technical, and account categories.

## Architecture and patterns

The Ticket Classification bounded context validates labeled examples in the domain. `PrepareDataset` deduplicates and splits data through a deterministic Strategy. Hugging Face, PEFT, and PyTorch are infrastructure details imported only by the training adapter. Visitor is not used because the stable variation is splitting/training policy, not operations over a heterogeneous object tree.

```bash
python -m pip install -e '.[dev]'
pytest
ruff check .

# Optional real training dependencies
python -m pip install -e '.[tuning]'
```

The repository intentionally ships no customer data, model weights, tokens, or experiment secrets. See [architecture](docs/architecture.md), [ADR-001](docs/adr/001-lora-adapter.md), and [model card](docs/model-card.md).
