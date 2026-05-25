import torch
from sacrebleu import corpus_bleu, corpus_chrf
from transformers import AutoModelForCausalLM, AutoTokenizer


# =====================
# 1. Config
# =====================

de_path = "../data/KDE4.de-hsb.de"
hsb_path = "../data/KDE4.de-hsb.hsb"
num_test = 2000
qwen_name = "Qwen/Qwen2.5-7B-Instruct"
predictions_path = "qwen_predictions.txt"


# =====================
# 2. Data
# =====================

def load_dataset():
    with open(de_path, encoding="utf-8") as f:
        de_sentences = [line.strip() for line in f]

    with open(hsb_path, encoding="utf-8") as f:
        hsb_references = [line.strip() for line in f]

    print("\nDataset size:")
    print("DE:", len(de_sentences))
    print("HSB:", len(hsb_references))

    assert len(de_sentences) == len(hsb_references)

    print("\nExample:")
    print("DE:", de_sentences[0])
    print("HSB:", hsb_references[0])

    return de_sentences, hsb_references


# =====================
# 3. Model
# =====================

def load_model():
    print("\nLoading Qwen...")
    print("Model:", qwen_name)

    tokenizer = AutoTokenizer.from_pretrained(qwen_name)
    model = AutoModelForCausalLM.from_pretrained(
        qwen_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print("Qwen loaded")
    print("Device map:", getattr(model, "hf_device_map", "single device"))

    return tokenizer, model


# =====================
# 4. Translation
# =====================

def extract_translation(decoded_text: str, prompt: str) -> str:
    text = decoded_text.strip()

    if "Upper Sorbian:" in text:
        text = text.rsplit("Upper Sorbian:", 1)[-1].strip()
    else:
        text = text.replace(prompt, "").strip()

        cleanup_prefixes = (
            "Translate the following German sentence into Upper Sorbian.",
            "Only output the translation.",
            "German:",
        )

        lines = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            if any(line.startswith(prefix) for prefix in cleanup_prefixes):
                continue
            lines.append(line)

        if lines:
            text = lines[-1].strip()

    return text.strip().strip("\"'")


def translate_with_qwen(german: str, tokenizer, model) -> str:
    prompt = f"""Translate the following German sentence into Upper Sorbian.
Only output the translation.

German: {german}
Upper Sorbian:"""

    model_device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(model_device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=30,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_ids = outputs[0][inputs.input_ids.shape[-1]:]
    result = tokenizer.decode(generated_ids, skip_special_tokens=True)

    return extract_translation(result, prompt)


# =====================
# 5. Evaluation
# =====================

def evaluate(tokenizer, model, de_sentences, hsb_references):
    predictions = []
    test_size = min(num_test, len(de_sentences))

    print(f"\nRunning Qwen baseline on {test_size} sentences...")

    for i in range(test_size):
        german = de_sentences[i]
        pred = translate_with_qwen(german, tokenizer, model)
        predictions.append(pred)

        if i < 5:
            print("=" * 40)
            print("DE:", german)
            print("REF:", hsb_references[i])
            print("PRED:", pred)

        if (i + 1) % 100 == 0 or (i + 1) == test_size:
            print(f"[{i + 1}/{test_size}] done")

    with open(predictions_path, "w", encoding="utf-8") as f:
        for pred in predictions:
            f.write(pred + "\n")

    references = hsb_references[:len(predictions)]
    bleu = corpus_bleu(predictions, [references])
    chrf = corpus_chrf(predictions, [references])

    print("\nFinal scores:")
    print("BLEU:", bleu.score)
    print("chrF++:", chrf.score)
    print("Predictions saved to:", predictions_path)


# =====================
# 6. Main
# =====================

def main():
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("CUDA device:", torch.cuda.get_device_name(0))

    de_sentences, hsb_references = load_dataset()
    tokenizer, model = load_model()
    evaluate(tokenizer, model, de_sentences, hsb_references)


if __name__ == "__main__":
    main()
