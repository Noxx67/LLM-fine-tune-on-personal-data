from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base_model_path = "./model/Qwen3.5-2B/snapshots/15852e8c16360a2fea060d615a32b45270f8a8fc"  # e.g., "meta-llama/Llama-3.2-3B"
lora_adapter_path = "./qwen3_5_finetunedV2.65"  # output_dir from training
output_dir = "./merged_model"

# Load base model & tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model_path)
base_model = AutoModelForCausalLM.from_pretrained(base_model_path)

# Merge LoRA weights into base weights
model = PeftModel.from_pretrained(base_model, lora_adapter_path)
merged_model = model.merge_and_unload()

# Save merged weights
merged_model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)