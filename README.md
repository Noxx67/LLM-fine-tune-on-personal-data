# LLM Fine-Tuning on Personal Data

Dumb side project to try and test LoRA/QLoRA training on local LLMs.

---

## Overview

This project provides an end-to-end pipeline to parse, clean, format, and fine-tune Large Language Models on personal chat exports. The goal is to align a local LLM to replicate specific conversational writing styles, tone, and personal context securely on local hardware.

Project so far is not organized well and since I doubt anyone will try to use it I won't bother unless asked to. The dataset used is my personal DMs on Instagram for now and the base model is **Qwen3.5-2B**. With my RTX 3060 (6GB VRAM), it allows me to train on the entire dataset with batch size 16 for 3 hours on 1 epoch.

I also keep a "journal" in the repo called [Notes_to_self.txt](Notes_to_self.txt) which I will detail more on later if I ever bother.

---

## How it works

__Most of the functions work with Qwen3.5 for the preprocessing. if youre using another model you probably should change the export function to match the model__

- **Data Parsing:** Convert raw Instagram `your_instagram_activity/messages` JSON exports into JSONL format for `system`, `input` (other person), and `target` (me). Messages are truncated by message reply delay and message length for consistent data. You can also visualize average message length and get a list of the most used words, which can be interesting and funny to look at.
- **Privacy Filtering:** Lol there is none.
- **Fast Fine-Tuning:** Uses **QLoRA (4-bit quantization)** for maximum speed and minimal GPU memory usage.
- **Local Deployment:** Export trained LoRA adapters to Safetensors. Since I use `llama.cpp`, I already have the conversion script ready to get it into GGUF format.

All preprocessing is done in the notebook.

To train the model:
1. Put the base model `.safetensors` files in the `model/` folder.
2. Put your selected Insta DMs inside `dataset/datafolders/` (I recommend selecting specific DMs you want to use for better results instead of tiny 3-message threads which ruin the data). The `data/` folder just holds all your extra raw junk.
3. Run `python train.py` and you should see an adapter appear (your LoRA). 
4. If you need it to be merged, run `python merge.py` and you'll get a final `merged_model/` folder ready to use.

---

## Yap & Next Steps

So far with the preprocessing, the model doesn't perform very well and suffers from repetition loops or inconsistent replies. Increasing the length of the message window keeps some consistency.

- Next time I will use a local LLM as a judge during preprocessing to check if the data sample is good to be trained on or not, since raw data quality is pretty bad.
- Adding Discord message exports is the best improvement I can think of for now.

*(And no, you're not getting access to the model weight files).*