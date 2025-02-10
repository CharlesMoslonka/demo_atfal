# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "datasets",
#     "llmcompressor",
#     "transformers",
# ]
# ///
import torch
from datasets import load_dataset
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.transformers import SparseAutoModelForCausalLM, oneshot
from llmcompressor.transformers.compression.helpers import (
    calculate_offload_device_map,
    custom_offload_device_map,
)
from transformers import AutoModelForCausalLM, AutoTokenizer

recipe = QuantizationModifier(targets="Linear", sheme="FP8_DYNAMIC", ignore=["lm_head"])

model_stub = "mistralai/Ministral-8B-Instruct-2410"
model_name = model_stub.split("/")[-1]

device_map = calculate_offload_device_map(model_stub, reserve_for_hessians=False, num_gpus=1, torch_dtype=torch.float16)

model = SparseAutoModelForCausalLM.from_pretrained(model_stub, torch_dtype=torch.float16, device_map=device_map)
tokenizer = AutoTokenizer.from_pretrained(model_stub)

output_dir = f"./{model_name}-FP8-Dynamic"

DATASET_ID = "HuggingFaceH4/ultrachat_200k"
DATASET_SPLIT = "train_sft"
NUM_CALIBRATION_SAMPLES = 512
MAX_SEQUENCE_LENGTH = 4096

ds = load_dataset(DATASET_ID, split=DATASET_SPLIT)
ds = ds.shuffle(seed=42).select(range(NUM_CALIBRATION_SAMPLES))


def preprocess(example):
    return {
        "text": tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
        )
    }


ds = ds.map(preprocess)


def tokenize(sample):
    return tokenizer(
        sample["text"],
        padding=False,
        max_length=MAX_SEQUENCE_LENGTH,
        truncation=True,
        add_special_tokens=False,
    )


ds = ds.map(tokenize, remove_columns=ds.column_names)

oneshot(
    model=model,
    output_dir=output_dir,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    save_compressed=True,
)
