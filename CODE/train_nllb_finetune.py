import argparse
from pathlib import Path

import torch
from datasets import Dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)


SRC_LANG = "deu_Latn"
TGT_LANG = "hsb_Latn"
BASE_DIR = Path(__file__).resolve().parent


def resolve_path(path: str) -> Path:
    path_obj = Path(path)
    if path_obj.is_absolute():
        return path_obj
    return BASE_DIR / path_obj


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"true", "1", "yes", "y"}:
        return True
    if value in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("Expected a boolean value.")


def read_lines(path: Path) -> list[str]:
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f]


def load_parallel_data(src_path: str, tgt_path: str, debug: bool) -> Dataset:
    src_path_obj = resolve_path(src_path)
    tgt_path_obj = resolve_path(tgt_path)
    sources = read_lines(src_path_obj)
    targets = read_lines(tgt_path_obj)

    if len(sources) != len(targets):
        raise ValueError(f"Source/target length mismatch: {len(sources)} vs {len(targets)}")

    if debug:
        sources = sources[:10]
        targets = targets[:10]

    print(f"Loaded {len(sources)} sentence pairs from {src_path_obj} and {tgt_path_obj}")
    return Dataset.from_dict({"source": sources, "target": targets})


def tokenize_dataset(dataset: Dataset, tokenizer, max_length: int) -> Dataset:
    tokenizer.src_lang = SRC_LANG

    def preprocess(batch):
        return tokenizer(
            batch["source"],
            text_target=batch["target"],
            max_length=max_length,
            truncation=True,
        )

    return dataset.map(preprocess, batched=True, remove_columns=["source", "target"])


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune extended NLLB for German -> Upper Sorbian.")
    parser.add_argument("--model_path", default="../models/nllb-600M-hsb-init")
    parser.add_argument("--train_src_path", default="../data/splits/train.de")
    parser.add_argument("--train_tgt_path", default="../data/splits/train.hsb")
    parser.add_argument("--dev_src_path", default="../data/splits/dev.de")
    parser.add_argument("--dev_tgt_path", default="../data/splits/dev.hsb")
    parser.add_argument("--debug", type=str2bool, default=True)
    parser.add_argument("--num_train_epochs", type=float, default=5)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--output_dir", default="../models/nllb-600M-hsb-finetuned")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = resolve_path(args.output_dir)
    resolve_path("../outputs").mkdir(parents=True, exist_ok=True)
    resolve_path("../models").mkdir(parents=True, exist_ok=True)
    resolve_path("../data/splits").mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.debug:
        print("DEBUG mode is ON: using first 10 train/dev pairs, 1 epoch, batch size 1.")
        args.num_train_epochs = 1
        args.per_device_train_batch_size = 1
        args.per_device_eval_batch_size = 1
        args.gradient_accumulation_steps = 1

    if not torch.cuda.is_available():
        print("WARNING: CUDA is not available. CPU debug is allowed, but training will be slow.")

    print("\nLoading tokenizer and model...")
    model_path = resolve_path(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

    hsb_id = tokenizer.convert_tokens_to_ids(TGT_LANG)
    print("hsb_Latn token id:", hsb_id)
    print("unk token id:", tokenizer.unk_token_id)
    if hsb_id == tokenizer.unk_token_id:
        raise RuntimeError("hsb_Latn maps to unk_token_id. Run CODE/extend_nllb_hsb.py first.")

    model.config.forced_bos_token_id = hsb_id

    train_dataset = load_parallel_data(args.train_src_path, args.train_tgt_path, args.debug)
    dev_dataset = load_parallel_data(args.dev_src_path, args.dev_tgt_path, args.debug)
    train_dataset = tokenize_dataset(train_dataset, tokenizer, args.max_length)
    dev_dataset = tokenize_dataset(dev_dataset, tokenizer, args.max_length)

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_steps=1 if args.debug else 50,
        predict_with_generate=False,
        report_to="none",
        fp16=torch.cuda.is_available(),
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    print("\nStarting NLLB fine-tuning...")
    trainer.train()

    print("\nSaving fine-tuned model and tokenizer...")
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(output_dir)
    print("Saved to:", output_dir)


if __name__ == "__main__":
    main()
