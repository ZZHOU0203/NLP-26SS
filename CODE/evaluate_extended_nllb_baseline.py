import argparse
from pathlib import Path

import torch
from sacrebleu import corpus_bleu, corpus_chrf
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


SRC_LANG = "deu_Latn"
TGT_LANG = "hsb_Latn"
BASE_DIR = Path(__file__).resolve().parent


def resolve_path(path: str) -> Path:
    path_obj = Path(path)
    if path_obj.is_absolute():
        return path_obj
    return BASE_DIR / path_obj


def read_lines(path: Path) -> list[str]:
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f]


def load_dataset(src_path: str, ref_path: str, num_test: int):
    src_path_obj = resolve_path(src_path)
    ref_path_obj = resolve_path(ref_path)
    src_sentences = read_lines(src_path_obj)
    references = read_lines(ref_path_obj)

    if len(src_sentences) != len(references):
        raise ValueError(f"Source/reference length mismatch: {len(src_sentences)} vs {len(references)}")

    test_size = min(num_test, len(src_sentences))
    print("\nDataset:")
    print("Source:", src_path_obj)
    print("Reference:", ref_path_obj)
    print("Available pairs:", len(src_sentences))
    print("Evaluation pairs:", test_size)

    return src_sentences[:test_size], references[:test_size]


def load_model(model_path: str):
    model_path_obj = resolve_path(model_path)
    print("\nLoading extended NLLB model...")
    print("Model path:", model_path_obj)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("WARNING: CUDA is not available. Running NLLB evaluation on CPU debug mode.")

    tokenizer = AutoTokenizer.from_pretrained(model_path_obj)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path_obj).to(device)
    model.eval()

    hsb_id = tokenizer.convert_tokens_to_ids(TGT_LANG)
    print("hsb_Latn token id:", hsb_id)
    print("unk token id:", tokenizer.unk_token_id)

    if hsb_id == tokenizer.unk_token_id:
        raise RuntimeError(
            "hsb_Latn still maps to unk_token_id. Please run CODE/extend_nllb_hsb.py first."
        )

    tokenizer.src_lang = SRC_LANG
    return tokenizer, model, device, hsb_id


def translate_with_nllb(source: str, tokenizer, model, device, forced_bos_token_id: int) -> str:
    inputs = tokenizer(source, return_tensors="pt").to(device)
    with torch.no_grad():
        generated_tokens = model.generate(
            **inputs,
            forced_bos_token_id=forced_bos_token_id,
            num_beams=5,
            max_new_tokens=128,
        )
    return tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0].strip()


def evaluate(tokenizer, model, device, forced_bos_token_id: int, src_sentences, references, output_path: str):
    predictions = []
    test_size = len(src_sentences)

    print(f"\nRunning extended NLLB baseline on {test_size} sentences...")
    for i, source in enumerate(src_sentences):
        prediction = translate_with_nllb(source, tokenizer, model, device, forced_bos_token_id)
        predictions.append(prediction)

        if i < 5:
            print("=" * 40)
            print("DE:", source)
            print("REF:", references[i])
            print("PRED:", prediction)

        if (i + 1) % 100 == 0 or (i + 1) == test_size:
            print(f"[{i + 1}/{test_size}] done")

    output_path_obj = resolve_path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path_obj, "w", encoding="utf-8") as f:
        for prediction in predictions:
            f.write(prediction + "\n")

    bleu = corpus_bleu(predictions, [references])
    chrf = corpus_chrf(predictions, [references])

    print("\nFinal scores:")
    print("BLEU:", bleu.score)
    print("chrF++:", chrf.score)
    print("Predictions saved to:", output_path_obj)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate extended NLLB German -> Upper Sorbian baseline.")
    parser.add_argument("--model_path", default="../models/nllb-600M-hsb-init")
    parser.add_argument("--src_path", default="../data/splits/test.de")
    parser.add_argument("--ref_path", default="../data/splits/test.hsb")
    parser.add_argument("--num_test", type=int, default=10)
    parser.add_argument("--output_path", default="../outputs/extended_nllb_predictions.txt")
    return parser.parse_args()


def main():
    args = parse_args()
    resolve_path("../outputs").mkdir(parents=True, exist_ok=True)
    resolve_path("../models").mkdir(parents=True, exist_ok=True)
    resolve_path("../data/splits").mkdir(parents=True, exist_ok=True)

    src_sentences, references = load_dataset(args.src_path, args.ref_path, args.num_test)
    tokenizer, model, device, hsb_id = load_model(args.model_path)
    evaluate(tokenizer, model, device, hsb_id, src_sentences, references, args.output_path)


if __name__ == "__main__":
    main()
