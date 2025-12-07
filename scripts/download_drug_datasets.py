"""
Download Additional Drug Information Datasets
"""
import os
from datasets import load_dataset

os.makedirs("data/drug_info", exist_ok=True)

print("💊 Downloading Additional Drug Datasets...\n")

# 1. Medical Meadow MediQA (drug-related Q&A)
print("1️⃣ Downloading Medical Meadow MediQA...")
try:
    mediqa = load_dataset("lavita/medical-qa-datasets", "medical_meadow_mediqa", split="train")
    mediqa.to_json("data/drug_info/medical_meadow_mediqa.jsonl")
    print(f"   ✅ Saved: {len(mediqa)} samples")
except Exception as e:
    print(f"   ⚠️ Error: {e}")

# 2. Medical Flashcards (includes drug info)
print("\n2️⃣ Downloading Medical Flashcards...")
try:
    flashcards = load_dataset("lavita/medical-qa-datasets", "medical_meadow_medical_flashcards", split="train")
    flashcards.to_json("data/drug_info/medical_flashcards.jsonl")
    print(f"   ✅ Saved: {len(flashcards)} samples")
except Exception as e:
    print(f"   ⚠️ Error: {e}")

# 3. Health Advice (includes medications)
print("\n3️⃣ Downloading Health Advice dataset...")
try:
    health_advice = load_dataset("lavita/medical-qa-datasets", "medical_meadow_health_advice", split="train")
    health_advice.to_json("data/drug_info/health_advice.jsonl")
    print(f"   ✅ Saved: {len(health_advice)} samples")
except Exception as e:
    print(f"   ⚠️ Error: {e}")

# 4. iCliniq dataset (doctor consultations with prescriptions)
print("\n4️⃣ Downloading iCliniq dataset...")
try:
    icliniq = load_dataset("lavita/medical-qa-datasets", "chatdoctor-icliniq", split="train")
    icliniq.to_json("data/drug_info/icliniq.jsonl")
    print(f"   ✅ Saved: {len(icliniq)} samples")
except Exception as e:
    print(f"   ⚠️ Error: {e}")

# Summary
print("\n" + "=" * 50)
print("✅ Download complete!")
print("=" * 50)

print("\n📁 Drug Information (data/drug_info/):")
for f in os.listdir("data/drug_info"):
    size = os.path.getsize(f"data/drug_info/{f}") / (1024*1024)
    print(f"   📄 {f} ({size:.1f} MB)")
