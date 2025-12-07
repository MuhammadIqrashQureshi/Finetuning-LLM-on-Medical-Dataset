"""
Medical LLM Fine-tuning - Windows Version
Quick training on RTX 3060 6GB
"""
import os
import json
import torch
from datasets import Dataset

print("=" * 50)
print("🏥 Medical LLM Fine-tuning")
print("=" * 50)

# Check GPU
print(f"\nCUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

# Config
MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # Small model, fast training
DATASET = "data/merged/medical_merged_15k.jsonl"
OUTPUT = "outputs/medical_tinyllama"
MAX_LEN = 512
SAMPLES = 2000  # Quick training with fewer samples

print(f"\nModel: {MODEL}")
print(f"Samples: {SAMPLES}")

# Load dataset
print("\n📂 Loading dataset...")
data = []
with open(DATASET, "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if i >= SAMPLES:
            break
        item = json.loads(line)
        # Convert to TinyLlama format
        text = item["prompt"].replace("<|start_header_id|>system<|end_header_id|}>\n\n", "<|system|>\n")
        text = text.replace("<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n", "</s>\n<|user|>\n")
        text = text.replace("<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n", "</s>\n<|assistant|>\n")
        text = text.replace("<|eot_id|>", "</s>")
        data.append({"text": text})

dataset = Dataset.from_list(data)
print(f"✅ Loaded {len(dataset)} samples")

# Load model with 4-bit quantization
print("\n🔄 Loading model...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

print("✅ Model loaded!")

# Add LoRA
print("\n🔧 Adding LoRA...")
model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Training
print("\n🚀 Starting training...")
os.makedirs(OUTPUT, exist_ok=True)

training_args = TrainingArguments(
    output_dir=OUTPUT,
    num_train_epochs=1,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
    save_steps=200,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    optim="adamw_8bit",
    weight_decay=0.01,
    max_grad_norm=0.3,
    report_to="none",
    gradient_checkpointing=True,
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=MAX_LEN,
    args=training_args,
)

print("\n" + "=" * 50)
print("🏋️ Training... (Est: 20-30 mins)")
print("=" * 50)

trainer.train()

# Save
print("\n💾 Saving model...")
model.save_pretrained(f"{OUTPUT}/final")
tokenizer.save_pretrained(f"{OUTPUT}/final")

print("\n✅ Training complete!")
print(f"📁 Saved to: {OUTPUT}/final")

# Quick test
print("\n🧪 Testing...")
model.eval()
prompt = "<|system|>\nYou are a medical assistant.</s>\n<|user|>\nWhat are symptoms of diabetes?</s>\n<|assistant|>\n"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=150, temperature=0.7, do_sample=True)
print(tokenizer.decode(out[0], skip_special_tokens=True))
