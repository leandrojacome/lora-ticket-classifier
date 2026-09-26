"""Optional LoRA adapter. Install the `tuning` extra before running this module."""

from pathlib import Path


def train(model_name: str, train_file: Path, output_dir: Path) -> None:
    from datasets import load_dataset
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    dataset = load_dataset("json", data_files=str(train_file), split="train")
    tokenized = dataset.map(lambda row: tokenizer(row["text"], truncation=True, padding="max_length", max_length=128))
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model = get_peft_model(model, LoraConfig(task_type=TaskType.CAUSAL_LM, r=8, lora_alpha=16, lora_dropout=0.05))
    trainer = Trainer(model=model, args=TrainingArguments(output_dir=str(output_dir), num_train_epochs=1, report_to=[]), train_dataset=tokenized)
    trainer.train()
    model.save_pretrained(output_dir)
