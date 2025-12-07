"""
Download Clinical/Diagnostic Medical Datasets
"""
import os
import json
from datasets import load_dataset

# Create data directory
os.makedirs("data/clinical", exist_ok=True)

print("📥 Downloading Clinical Medical Datasets...\n")

# 1. Download the pre-formatted Llama3 medical dataset (short version for testing)
print("1️⃣ Downloading medical_llama3_instruct_dataset_short (2,000 samples)...")
dataset_short = load_dataset("Shekswess/medical_llama3_instruct_dataset_short", split="train")
dataset_short.to_json("data/clinical/medical_llama3_short.jsonl")
print(f"   ✅ Saved: {len(dataset_short)} samples\n")

# 2. Download MedQuAD - Medical Question Answering Dataset
print("2️⃣ Downloading MedQuAD dataset...")
try:
    medquad = load_dataset("keivalya/MedQuad-MedicalQnADataset", split="train")
    medquad.to_json("data/clinical/medquad.jsonl")
    print(f"   ✅ Saved: {len(medquad)} samples\n")
except Exception as e:
    print(f"   ⚠️ MedQuAD not available: {e}\n")

# 3. Download ChatDoctor dataset (clinical conversations)
print("3️⃣ Downloading ChatDoctor dataset...")
try:
    chatdoctor = load_dataset("lavita/ChatDoctor-HealthCareMagic-100k", split="train")
    # Take a subset for manageable size
    chatdoctor_subset = chatdoctor.select(range(min(10000, len(chatdoctor))))
    chatdoctor_subset.to_json("data/clinical/chatdoctor_10k.jsonl")
    print(f"   ✅ Saved: {len(chatdoctor_subset)} samples\n")
except Exception as e:
    print(f"   ⚠️ ChatDoctor not available: {e}\n")

# 4. Download Medical Meadow WikiDoc
print("4️⃣ Downloading Medical Meadow WikiDoc...")
try:
    wikidoc = load_dataset("medalpaca/medical_meadow_wikidoc", split="train")
    wikidoc.to_json("data/clinical/medical_meadow_wikidoc.jsonl")
    print(f"   ✅ Saved: {len(wikidoc)} samples\n")
except Exception as e:
    print(f"   ⚠️ WikiDoc not available: {e}\n")

print("=" * 50)
print("✅ Download complete!")
print("\nDatasets saved in: data/clinical/")
print("\nFiles:")
for f in os.listdir("data/clinical"):
    size = os.path.getsize(f"data/clinical/{f}") / (1024*1024)
    print(f"   📄 {f} ({size:.1f} MB)")
