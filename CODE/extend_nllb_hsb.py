import os

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


# =====================
# 1. Config
# =====================

model_name = "facebook/nllb-200-distilled-600M"
new_lang_tag = "hsb_Latn"
preferred_init_lang = "ces_Latn"
fallback_init_lang = "deu_Latn"
output_dir = "../models/nllb-600M-hsb-init"


# =====================
# 2. Load tokenizer/model
# =====================

print("Loading tokenizer and model...")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

original_vocab_size = len(tokenizer)

print("Loaded:", model_name)
print("Original vocab size:", original_vocab_size)


# =====================
# 3. Check hsb_Latn support
# =====================

hsb_id_before = tokenizer.convert_tokens_to_ids(new_lang_tag)

print("\nBefore extension:")
print("hsb_Latn token id:", hsb_id_before)
print("tokenizer.unk_token_id:", tokenizer.unk_token_id)


# =====================
# 4. Language tag extension
# =====================

# NLLB controls the target language with special language tags such as
# deu_Latn or ces_Latn. If hsb_Latn is unknown, we add it as an additional
# special token, resize the model embeddings, and initialize the new embedding
# from a related language tag instead of leaving it randomly initialized.

if hsb_id_before == tokenizer.unk_token_id:
    print("\nhsb_Latn is not in the tokenizer vocabulary.")
    print("Adding hsb_Latn to additional_special_tokens...")

    existing_special_tokens = tokenizer.additional_special_tokens or []
    tokenizer.add_special_tokens(
        {
            "additional_special_tokens": existing_special_tokens + [new_lang_tag],
        }
    )
else:
    print("\nhsb_Latn is already supported by the tokenizer.")

new_vocab_size = len(tokenizer)
hsb_id = tokenizer.convert_tokens_to_ids(new_lang_tag)

print("\nAfter tokenizer extension:")
print("New vocab size:", new_vocab_size)
print("hsb_Latn id:", hsb_id)

if hsb_id == tokenizer.unk_token_id:
    raise RuntimeError("Failed to add hsb_Latn to the tokenizer.")

if new_vocab_size != original_vocab_size:
    print("\nResizing model token embeddings...")
    model.resize_token_embeddings(new_vocab_size)
else:
    print("\nModel embedding resize is not needed.")


# =====================
# 5. Initialize hsb_Latn embedding
# =====================

source_lang = preferred_init_lang
source_id = tokenizer.convert_tokens_to_ids(source_lang)

if source_id == tokenizer.unk_token_id:
    source_lang = fallback_init_lang
    source_id = tokenizer.convert_tokens_to_ids(source_lang)

if source_id == tokenizer.unk_token_id:
    raise RuntimeError(
        f"Could not find {preferred_init_lang} or {fallback_init_lang} in tokenizer."
    )

print("\nInitializing hsb_Latn embedding...")
print("Initialization source language:", source_lang)
print(f"{source_lang} id:", source_id)

with torch.no_grad():
    input_embeddings = model.get_input_embeddings().weight
    input_embeddings[hsb_id].copy_(input_embeddings[source_id])

    output_embedding_layer = model.get_output_embeddings()
    if output_embedding_layer is not None:
        output_embeddings = output_embedding_layer.weight

        if output_embeddings.data_ptr() != input_embeddings.data_ptr():
            output_embeddings[hsb_id].copy_(output_embeddings[source_id])
            print("Output embedding is separate; copied it as well.")
        else:
            print("Output embedding is tied with input embedding.")


# =====================
# 6. Save extended model/tokenizer
# =====================

os.makedirs(output_dir, exist_ok=True)

print("\nSaving extended tokenizer and model...")
tokenizer.save_pretrained(output_dir)
model.save_pretrained(output_dir)

print("\nDone.")
print("Original vocab size:", original_vocab_size)
print("New vocab size:", new_vocab_size)
print("hsb_Latn id:", hsb_id)
print("Initialization source language:", source_lang)
print("Saved to:", output_dir)
