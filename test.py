import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# 1. Paths to your base model and saved LoRA checkpoint/folder
base_model_id = "./model/Qwen3.5-2B/snapshots/15852e8c16360a2fea060d615a32b45270f8a8fc"  # e.g., "meta-llama/Llama-3.2-3B"
lora_path = "./qwen3_5_finetuned"  # output_dir from training

# 2. Load tokenizer and base model
tokenizer = AutoTokenizer.from_pretrained(base_model_id)
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id, torch_dtype=torch.bfloat16, device_map="auto"
)

# 3. Attach LoRA adapter
model = PeftModel.from_pretrained(base_model, lora_path)
model.eval()

# 4. Format prompt and run inference
prompt = "Hi can you present yourself"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
    )

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)