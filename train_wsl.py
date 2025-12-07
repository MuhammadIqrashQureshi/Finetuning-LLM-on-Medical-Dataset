#!/usr/bin/env python3
"""
Medical LLM Fine-tuning with Unsloth (WSL)
Optimized for RTX 3060 6GB with 15k samples
Estimated time: ~1.5-2 hours with Unsloth
"""

import os
import json
import torch
from datasets import Dataset

print("=" * 60)
print("🏥 Medical LLM Fine-tuning with Unsloth")
print("=" * 60)

# Check GPU
print(f"\n🖥️ CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"📊 GPU: {torch.cuda.get_device_name(0)}")
    print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# Import unsloth
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments

# ============ CONFIGURATION ============
max_seq_length = 1024
dtype = None  # Auto-detect
load_in_4bit = True

print("\n📋 Configuration:")
print(f"   Max sequence length: {max_seq_length}")
print(f"   4-bit quantization: {load_in_4bit}")

# ============ LOAD MODEL ============
print("\n🔄 Loading Llama 3 8B model...")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/llama-3-8b-Instruct-bnb-4bit",
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)

print("✅ Model loaded!")

# Add LoRA adapters
print("\n🔧 Adding LoRA adapters...")
model = FastLanguageModel.get_peft_model(
    model,
    r=8,  # Reduced for speed
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                   "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=42,
    max_seq_length=max_seq_length,
)
print("✅ LoRA adapters added!")

# ============ LOAD DATASET ============
print("\n📂 Loading dataset...")
dataset_path = "/mnt/d/semester/GenAI/Project/data/merged/medical_merged_15k.jsonl"

data = []
with open(dataset_path, "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        data.append({"text": item["prompt"]})

dataset = Dataset.from_list(data)
print(f"   ✅ Loaded {len(dataset)} samples")

# ============ TRAINING ============
print("\n🚀 Setting up training...")

training_args = TrainingArguments(
    output_dir="/mnt/d/semester/GenAI/Project/outputs/medical_llama3",
    num_train_epochs=1,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    logging_steps=25,
    save_strategy="steps",
    save_steps=500,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    optim="adamw_8bit",
    weight_decay=0.01,
    max_grad_norm=0.3,
    report_to="none",
    seed=42,
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    args=training_args,
)

# Show GPU memory before training
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"\n📊 GPU Memory: {start_gpu_memory} GB / {max_memory} GB")

# Train!
print("\n" + "=" * 60)
print("🏋️ Training in progress...")
print("   Expected time: ~1.5-2 hours with Unsloth")
print("=" * 60 + "\n")

trainer_stats = trainer.train()

# Show final stats
used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
print(f"\n📊 Peak GPU Memory used: {used_memory} GB")
print(f"⏱️ Training time: {trainer_stats.metrics['train_runtime']:.0f} seconds")

# ============ SAVE MODEL ============
print("\n💾 Saving fine-tuned model...")
output_dir = "/mnt/d/semester/GenAI/Project/outputs/medical_llama3/final"
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"✅ Model saved to: {output_dir}")

# ============ SAVE FOR OLLAMA (GGUF) ============
print("\n📦 Exporting to GGUF format for Ollama...")
gguf_dir = "/mnt/d/semester/GenAI/Project/outputs/medical_llama3/gguf"

# Save as GGUF Q4_K_M
model.save_pretrained_gguf(
    gguf_dir,
    tokenizer,
    quantization_method="q4_k_m",
)

print(f"✅ GGUF model saved to: {gguf_dir}")

# ============ QUICK TEST ============
print("\n🧪 Quick test of the fine-tuned model...")

FastLanguageModel.for_inference(model)

test_prompt = """<|start_header_id|>system<|end_header_id|>

You are a helpful medical assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>

What are the common symptoms of diabetes?<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""

inputs = tokenizer(test_prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=200, temperature=0.7, do_sample=True)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("\n📝 Test Response:")
print("-" * 40)
# Extract just the assistant's response
parts = response.split("assistant")
if len(parts) > 1:
    print(parts[-1].strip())
else:
    print(response[-500:])
print("-" * 40)

print("\n" + "=" * 60)
print("🎉 Training complete!")
print("=" * 60)
print("\n📁 Output files:")
print(f"   - LoRA adapters: {output_dir}")
print(f"   - GGUF model: {gguf_dir}")
print("\n🚀 To use with Ollama, run:")
print(f"   ollama create medical-llama3 -f {gguf_dir}/Modelfile")
