"""
Medical LLM Fine-tuning Script
Optimized for RTX 3060 6GB with 15k samples
Estimated time: ~2-3 hours
"""

import os
import json
import torch
from datasets import Dataset
from transformers import TrainingArguments
from trl import SFTTrainer

print("=" * 60)
print("🏥 Medical LLM Fine-tuning")
print("=" * 60)

# Check GPU
print(f"\n🖥️ CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"📊 GPU: {torch.cuda.get_device_name(0)}")
    print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# Try to import unsloth (for faster training)
try:
    from unsloth import FastLanguageModel
    UNSLOTH_AVAILABLE = True
    print("\n✅ Unsloth is available - using optimized training!")
except ImportError:
    UNSLOTH_AVAILABLE = False
    print("\n⚠️ Unsloth not available - using standard transformers")
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# ============ CONFIGURATION ============
CONFIG = {
    "base_model": "unsloth/llama-3-8b-Instruct-bnb-4bit",
    "dataset_path": "data/merged/medical_merged_15k.jsonl",
    "output_dir": "outputs/medical_llama3",
    "max_seq_length": 1024,  # Reduced for speed
    "lora_r": 8,  # Reduced for speed
    "lora_alpha": 16,
    "batch_size": 2,
    "gradient_accumulation": 4,
    "epochs": 1,
    "learning_rate": 2e-4,
}

print(f"\n📋 Configuration:")
print(f"   Dataset: {CONFIG['dataset_path']}")
print(f"   Max sequence length: {CONFIG['max_seq_length']}")
print(f"   LoRA rank: {CONFIG['lora_r']}")
print(f"   Epochs: {CONFIG['epochs']}")

# ============ LOAD DATASET ============
print("\n📂 Loading dataset...")
data = []
with open(CONFIG["dataset_path"], "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        data.append({"text": item["prompt"]})

dataset = Dataset.from_list(data)
print(f"   ✅ Loaded {len(dataset)} samples")

# ============ LOAD MODEL ============
print("\n🔄 Loading model (this may take a few minutes)...")

if UNSLOTH_AVAILABLE:
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=CONFIG["base_model"],
        max_seq_length=CONFIG["max_seq_length"],
        dtype=torch.float16,
        load_in_4bit=True,
    )
    
    # Add LoRA adapters
    model = FastLanguageModel.get_peft_model(
        model,
        r=CONFIG["lora_r"],
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],
        lora_alpha=CONFIG["lora_alpha"],
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )
else:
    # Standard transformers loading
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    
    base_model = CONFIG["base_model"].replace("unsloth/", "meta-llama/").replace("-bnb-4bit", "")
    
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Prepare for training
    model = prepare_model_for_kbit_training(model)
    
    # Add LoRA
    lora_config = LoraConfig(
        r=CONFIG["lora_r"],
        lora_alpha=CONFIG["lora_alpha"],
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

print("✅ Model loaded!")
model.print_trainable_parameters()

# ============ TRAINING ============
print("\n🚀 Starting training...")

training_args = TrainingArguments(
    output_dir=CONFIG["output_dir"],
    num_train_epochs=CONFIG["epochs"],
    per_device_train_batch_size=CONFIG["batch_size"],
    gradient_accumulation_steps=CONFIG["gradient_accumulation"],
    learning_rate=CONFIG["learning_rate"],
    fp16=True,
    logging_steps=25,
    save_strategy="steps",
    save_steps=500,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    optim="adamw_8bit",
    weight_decay=0.01,
    max_grad_norm=0.3,
    report_to="none",  # Disable wandb/tensorboard
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=CONFIG["max_seq_length"],
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
trainer.save_model(os.path.join(CONFIG["output_dir"], "final"))
tokenizer.save_pretrained(os.path.join(CONFIG["output_dir"], "final"))

print("\n" + "=" * 60)
print("✅ Training complete!")
print(f"📁 Model saved to: {CONFIG['output_dir']}/final")
print("=" * 60)

# ============ QUICK TEST ============
print("\n🧪 Quick test of the fine-tuned model...")

if UNSLOTH_AVAILABLE:
    FastLanguageModel.for_inference(model)

test_prompt = """<|start_header_id|>system<|end_header_id|>

You are a helpful medical assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>

What are the common symptoms of diabetes?<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""

inputs = tokenizer(test_prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=200, temperature=0.7)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("\n📝 Test Response:")
print("-" * 40)
print(response.split("assistant")[-1].strip())
print("-" * 40)

print("\n🎉 All done! Your medical LLM is ready.")
print("   Next step: Export to Ollama using export_to_ollama.py")
