import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
from sacrebleu import corpus_bleu, corpus_chrf


# =====================
# 1. Paths
# =====================

de_path = "../data/KDE4.de-hsb.de"
hsb_path = "../data/KDE4.de-hsb.hsb"

num_test = 2000


# =====================
# 2. Device
# =====================

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)


# =====================
# 3. NLLB check
# =====================

print("\nLoading NLLB...")

nllb_name = "facebook/nllb-200-distilled-600M"

nllb_tokenizer = AutoTokenizer.from_pretrained(nllb_name)
nllb_model = AutoModelForSeq2SeqLM.from_pretrained(nllb_name).to(device)

print("NLLB loaded")
print("hsb_Latn token id:", nllb_tokenizer.convert_tokens_to_ids("hsb_Latn"))
print("unk token id:", nllb_tokenizer.unk_token_id)

if nllb_tokenizer.convert_tokens_to_ids("hsb_Latn") == nllb_tokenizer.unk_token_id:
    print("NLLB does not support hsb_Latn directly.")


# =====================
# 4. Load Qwen
# =====================

print("\nLoading Qwen...")

qwen_name = "Qwen/Qwen2.5-0.5B-Instruct"

qwen_tokenizer = AutoTokenizer.from_pretrained(qwen_name)
qwen_model = AutoModelForCausalLM.from_pretrained(
    qwen_name,
    torch_dtype="auto"
).to(device)

qwen_model.eval()

print("Qwen loaded")


# =====================
# 5. Load KDE4 data
# =====================

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


# =====================
# 6. Qwen translation function
# =====================

def translate_with_qwen(german: str) -> str:
    prompt = f"""Translate the following German sentence into Upper Sorbian.
Only output the translation.

German: {german}
Upper Sorbian:"""

    inputs = qwen_tokenizer(
        prompt,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = qwen_model.generate(
            **inputs,
            max_new_tokens=30,
            do_sample=False,
            pad_token_id=qwen_tokenizer.eos_token_id
        )

    result = qwen_tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    if "Upper Sorbian:" in result:
        result = result.split("Upper Sorbian:")[-1].strip()

    return result


# =====================
# 7. Run evaluation
# =====================

predictions = []

test_size = min(num_test, len(de_sentences))

print(f"\nRunning Qwen baseline on {test_size} sentences...")

for i in range(test_size):
    german = de_sentences[i]
    pred = translate_with_qwen(german)
    predictions.append(pred)

    if i < 5:
        print("=" * 40)
        print("DE:", german)
        print("REF:", hsb_references[i])
        print("PRED:", pred)

    if (i + 1) % 100 == 0:
        print(f"Finished {i + 1}/{test_size}")


# =====================
# 8. Metrics
# =====================
with open("qwen_predictions.txt", "w", encoding="utf-8") as f:
    for p in predictions:
        f.write(p + "\n")
        
references = hsb_references[:len(predictions)]

bleu = corpus_bleu(predictions, [references])
chrf = corpus_chrf(predictions, [references])

print("\nFinal scores:")
print("BLEU:", bleu.score)
print("chrF++:", chrf.score)