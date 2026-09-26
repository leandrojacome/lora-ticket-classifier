# Architecture

The domain owns label and text invariants. The application owns deterministic deduplication and evaluation splitting. Infrastructure owns JSONL and the optional training framework. This boundary keeps data-quality tests fast and prevents model downloads in CI. Real experiments must pin a base-model revision, record dataset provenance, evaluate per-class precision/recall, and review privacy and licensing before deployment.
