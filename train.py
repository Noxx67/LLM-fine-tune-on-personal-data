import os

os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"

import torch
from datasets import load_dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import SFTConfig, SFTTrainer

# ==========================================
# PATH CONFIGURATION
# ==========================================
MODEL_PATH = (
    "./model/Qwen3.5-2B/snapshots/15852e8c16360a2fea060d615a32b45270f8a8fc"
)
DATASET_PATH = "./dataset/train_dataV2.65.jsonl"
OUTPUT_DIR = "./qwen3_5_finetunedV2.65"
# ==========================================


def main():
    # 1. Quantization Setup
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # 2. Load Tokenizer & Model
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        padding_side="right",
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map="auto",
        trust_remote_code=True,
    )

    if not hasattr(model.config, "is_vlm"):
        model.config.is_vlm = False

    model = prepare_model_for_kbit_training(model)

    # 3. Regularized LoRA Setup
    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # 4. Load Dataset
    raw_dataset = load_dataset(
        "json", data_files={"train": DATASET_PATH}
    )["train"]

    # 5. Training Configurations
    sft_config = SFTConfig(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=16,  # Increased back to 8
        gradient_accumulation_steps=1,  # Effective batch size = 16
        learning_rate=1e-5,
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        logging_steps=5,
        max_length=256,  # Reduced back to 256
        num_train_epochs=3,
        dataloader_num_workers=0,  # Set to 2-4 to speed up CPU data collator processing
        dataloader_pin_memory=True,
        bf16=True,
        fp16=False,
        save_strategy="epoch",
        optim="adamw_torch_fused",
        remove_unused_columns=False,
        report_to="none",
        dataset_text_field="text",
    )

    # 6. Initialize Trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=raw_dataset,
        peft_config=peft_config,
        args=sft_config,
        processing_class=tokenizer,
    )

    print("Starting training execution...")
    trainer.train()

    # 7. Save
    trainer.model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("Model adapters and tokenizer saved successfully.")


if __name__ == "__main__":
    main()