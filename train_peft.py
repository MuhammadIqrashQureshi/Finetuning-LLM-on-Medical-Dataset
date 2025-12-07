#!/usr/bin/env python3
"""
Medical LLM Fine-tuning with PEFT (WSL)
Standard approach without unsloth - more stable
Optimized for RTX 3060 6GB with 15k samples
Estimated time: ~2-3 hours
"""

import os
import json
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

print("=" * 60)
print("🏥 Medical LLM Fine-tuning with PEFT")
print("=" * 60)

# Check GPU
print(f"\n🖥️ CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"📊 GPU: {torch.cuda.get_device_name(0)}")
    print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# ============ CONFIGURATION ============
# Using NousResearch's Llama 3 (no login required)
MODEL_NAME = "NousResearch/Meta-Llama-3-8B-Instruct"
DATASET_PATH = "/mnt/d/semester/GenAI/Project/data/merged/medical_merged_15k.jsonl"
OUTPUT_DIR = "/mnt/d/semester/GenAI/Project/outputs/medical_llama3"
MAX_SEQ_LENGTH = 1024

print(f"\n📋 Configuration:")
print(f"   Model: {MODEL_NAME}")
print(f"   Dataset: {DATASET_PATH}")
print(f"   Max sequence length: {MAX_SEQ_LENGTH}")

# ============ LOAD DATASET ============
print("\n📂 Loading dataset...")
data = []
with open(DATASET_PATH, "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        data.append({"text": item["prompt"]})

dataset = Dataset.from_list(data)
print(f"   ✅ Loaded {len(dataset)} samples")

# ============ LOAD MODEL ============
print("\n🔄 Loading model (this may take a few minutes)...")

# 4-bit quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

# Load model
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

print("✅ Model loaded!")

# ============ ADD LORA ============
print("\n🔧 Adding LoRA adapters...")

model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                   "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ============ TRAINING ============
print("\n🚀 Setting up training...")

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=25,
    save_strategy="steps",
    save_steps=500,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    optim="adamw_8bit",
    weight_decay=0.01,
    max_grad_norm=0.3,
    report_to="none",
    gradient_checkpointing=True,
    seed=42,
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LENGTH,
    args=training_args,
)

# Train!
print("\n" + "=" * 60)
print("🏋️ Training in progress...")
print("   Expected time: ~2-3 hours")
print("=" * 60 + "\n")

trainer.train()

# ============ SAVE MODEL ============
print("\n💾 Saving fine-tuned model...")
final_dir = os.path.join(OUTPUT_DIR, "final")
model.save_pretrained(final_dir)
tokenizer.save_pretrained(final_dir)
print(f"✅ Model saved to: {final_dir}")

# ============ QUICK TEST ============
print("\n🧪 Quick test of the fine-tuned model...")

model.eval()

test_prompt = """<|start_header_id|>system<|end_header_id|>

You are a helpful medical assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>

What are the common symptoms of diabetes?<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""

inputs = tokenizer(test_prompt, return_tensors="pt").to("cuda")
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=200, temperature=0.7, do_sample=True)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("\n📝 Test Response:")
print("-" * 40)
parts = response.split("assistant")
if len(parts) > 1:
    print(parts[-1].strip()[:500])
else:
    print(response[-500:])
print("-" * 40)

print("\n" + "=" * 60)
print("🎉 Training complete!")
print("=" * 60)
print(f"\n📁 Model saved to: {final_dir}")
print("\n🚀 Next: Export to GGUF format for Ollama")
