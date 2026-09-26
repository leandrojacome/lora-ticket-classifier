# ADR-001: Fine-tune with a LoRA adapter

## Status
Accepted for experiments.

## Decision
Train small low-rank adapters while keeping base-model weights unchanged.

## Consequences
Experiments need less memory and produce small artifacts. Quality still depends on the base model and representative data; an adapter is not evidence of production fitness.
