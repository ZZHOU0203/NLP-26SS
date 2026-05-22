import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


# =====================
# 1. Config
# =====================

model_dir = "../models/nllb-600M-hsb-init"
src_lang = "deu_Latn"
tgt_lang = "hsb_Latn"

test_sentences = [
    "Guten Morgen.",
    "Wie geht es dir?",
    "Das ist ein Test.",
]


# =====================
# 2. Load extended tokenizer/model
# =====================

print("Loading extended tokenizer and model...")

tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
model.eval()

print("Loaded:", model_dir)


# =====================
# 3. Verify hsb_Latn token
# =====================

hsb_id = tokenizer.convert_tokens_to_ids(tgt_lang)

print("\nTokenizer check:")
print("hsb_Latn token id:", hsb_id)
print("tokenizer.unk_token_id:", tokenizer.unk_token_id)

if hsb_id == tokenizer.unk_token_id:
    raise RuntimeError("hsb_Latn still maps to unk_token_id. Run extend_nllb_hsb.py first.")


# =====================
# 4. Inference smoke test
# =====================

# NLLB uses tokenizer.src_lang for the source language and forced_bos_token_id
# for the target language. Here hsb_Latn is the newly added target language tag.

tokenizer.src_lang = src_lang
forced_bos_token_id = hsb_id

print("\nRunning German -> Upper Sorbian smoke test...")

for source in test_sentences:
    inputs = tokenizer(
        source,
        return_tensors="pt",
    )

    with torch.no_grad():
        generated_tokens = model.generate(
            **inputs,
            forced_bos_token_id=forced_bos_token_id,
            num_beams=5,
            max_new_tokens=128,
        )

    output = tokenizer.batch_decode(
        generated_tokens,
        skip_special_tokens=True,
    )[0]

    print("=" * 60)
    print("German source:", source)
    print("Generated Upper Sorbian output:", output)
