import argparse
import importlib.util
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)


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


def load_parallel_pairs(src_path: str, tgt_path: str, debug: bool):
    src_path_obj = resolve_path(src_path)
    tgt_path_obj = resolve_path(tgt_path)
    sources = read_lines(src_path_obj)
    targets = read_lines(tgt_path_obj)

    if len(sources) != len(targets):
        raise ValueError(f"Source/target length mismatch: {len(sources)} vs {len(targets)}")

    if debug:
        sources = sources[:10]
        targets = targets[:10]

    pairs = list(zip(sources, targets))
    print(f"Loaded {len(pairs)} sentence pairs from {src_path_obj} and {tgt_path_obj}")
    return pairs


def build_messages(german: str, upper_sorbian: str | None = None):
    user_content = (
        "Translate the following German sentence into Upper Sorbian.\n"
        f"German: {german}"
    )
    messages = [{"role": "user", "content": user_content}]
    if upper_sorbian is not None:
        messages.append({"role": "assistant", "content": upper_sorbian})
    return messages


class QwenTranslationDataset(Dataset):
    def __init__(self, pairs, tokenizer, max_length: int):
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index: int):
        german, upper_sorbian = self.pairs[index]

        prompt_messages = build_messages(german)
        full_messages = build_messages(german, upper_sorbian)

        prompt_ids = self.tokenizer.apply_chat_template(
            prompt_messages,
            add_generation_prompt=True,
            truncation=True,
            max_length=self.max_length,
        )
        full_ids = self.tokenizer.apply_chat_template(
            full_messages,
            add_generation_prompt=False,
            truncation=True,
            max_length=self.max_length,
        )

        labels = full_ids.copy()
        prompt_len = min(len(prompt_ids), len(labels))
        labels[:prompt_len] = [-100] * prompt_len

        return {"input_ids": full_ids, "attention_mask": [1] * len(full_ids), "labels": labels}


class CausalLMCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        max_len = max(len(feature["input_ids"]) for feature in features)
        pad_id = self.tokenizer.pad_token_id

        batch = {"input_ids": [], "attention_mask": [], "labels": []}
        for feature in features:
            pad_len = max_len - len(feature["input_ids"])
            batch["input_ids"].append(feature["input_ids"] + [pad_id] * pad_len)
            batch["attention_mask"].append(feature["attention_mask"] + [0] * pad_len)
            batch["labels"].append(feature["labels"] + [-100] * pad_len)

        return {key: torch.tensor(value, dtype=torch.long) for key, value in batch.items()}


def load_qwen_model(args):
    if not torch.cuda.is_available():
        print("WARNING: CUDA is not available. Qwen 7B LoRA debug may be very slow or impossible on CPU.")

    quantization_config = None
    if args.load_in_4bit:
        if importlib.util.find_spec("bitsandbytes") is None:
            raise RuntimeError(
                "bitsandbytes is not installed, so --load_in_4bit true cannot be used. "
                "Install bitsandbytes or rerun with --load_in_4bit false."
            )
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        print("4-bit loading enabled with bitsandbytes.")
    else:
        print("4-bit loading disabled. Use --load_in_4bit true when bitsandbytes is available.")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            quantization_config=quantization_config,
        )
    except Exception as exc:
        raise RuntimeError(
            "Failed to load Qwen for LoRA training. Check GPU memory, model access, "
            "and whether bitsandbytes is installed when using 4-bit loading."
        ) from exc

    if args.load_in_4bit:
        model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return tokenizer, model


def parse_args():
    parser = argparse.ArgumentParser(description="LoRA fine-tuning for Qwen German -> Upper Sorbian.")
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--train_src_path", default="../data/splits/train.de")
    parser.add_argument("--train_tgt_path", default="../data/splits/train.hsb")
    parser.add_argument("--dev_src_path", default="../data/splits/dev.de")
    parser.add_argument("--dev_tgt_path", default="../data/splits/dev.hsb")
    parser.add_argument("--debug", type=str2bool, default=True)
    parser.add_argument("--num_train_epochs", type=float, default=3)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--load_in_4bit", type=str2bool, default=False)
    parser.add_argument("--output_dir", default="../models/qwen2.5-7b-hsb-lora")
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

    tokenizer, model = load_qwen_model(args)

    train_pairs = load_parallel_pairs(args.train_src_path, args.train_tgt_path, args.debug)
    dev_pairs = load_parallel_pairs(args.dev_src_path, args.dev_tgt_path, args.debug)

    train_dataset = QwenTranslationDataset(train_pairs, tokenizer, args.max_length)
    dev_dataset = QwenTranslationDataset(dev_pairs, tokenizer, args.max_length)
    data_collator = CausalLMCollator(tokenizer)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_steps=1 if args.debug else 50,
        report_to="none",
        bf16=torch.cuda.is_available(),
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    print("\nStarting Qwen LoRA fine-tuning...")
    trainer.train()

    print("\nSaving LoRA adapter and tokenizer...")
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(output_dir)
    print("Saved to:", output_dir)


if __name__ == "__main__":
    main()
