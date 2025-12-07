"""
Download General Medicine and Drug Information Datasets
"""
import os
from datasets import load_dataset

# Create data directories
os.makedirs("data/general_medicine", exist_ok=True)
os.makedirs("data/drug_info", exist_ok=True)

print("📥 Downloading General Medicine & Drug Information Datasets...\n")

# ============ GENERAL MEDICINE ============
print("=" * 50)
print("🏥 GENERAL MEDICINE DATASETS")
print("=" * 50)

# 1. Full medical instruction dataset
print("\n1️⃣ Downloading medical_llama3_instruct_dataset (FULL - 26k+ samples)...")
try:
    dataset_full = load_dataset("Shekswess/medical_llama3_instruct_dataset", split="train")
    dataset_full.to_json("data/general_medicine/medical_llama3_full.jsonl")
    print(f"   ✅ Saved: {len(dataset_full)} samples")
except Exception as e:
    print(f"   ⚠️ Error: {e}")

# 2. PubMedQA - Research-based medical Q&A
print("\n2️⃣ Downloading PubMedQA dataset...")
try:
    pubmedqa = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
    pubmedqa.to_json("data/general_medicine/pubmedqa.jsonl")
    print(f"   ✅ Saved: {len(pubmedqa)} samples")
except Exception as e:
    print(f"   ⚠️ Error: {e}")

# 3. Medical Q&A dataset
print("\n3️⃣ Downloading Medical Q&A dataset...")
try:
    med_qa = load_dataset("medmcqa", split="train")
    # Take subset for manageable size
    med_qa_subset = med_qa.select(range(min(20000, len(med_qa))))
    med_qa_subset.to_json("data/general_medicine/medmcqa_20k.jsonl")
    print(f"   ✅ Saved: {len(med_qa_subset)} samples")
except Exception as e:
    print(f"   ⚠️ Error: {e}")

# 4. Health advice dataset  
print("\n4️⃣ Downloading Health Advice dataset...")
try:
    health = load_dataset("lavita/medical-qa-datasets", "all", split="train")
    health_subset = health.select(range(min(15000, len(health))))
    health_subset.to_json("data/general_medicine/health_qa_15k.jsonl")
    print(f"   ✅ Saved: {len(health_subset)} samples")
except Exception as e:
    print(f"   ⚠️ Error: {e}")

# ============ DRUG INFORMATION ============
print("\n" + "=" * 50)
print("💊 DRUG INFORMATION DATASETS")
print("=" * 50)

# 5. Drug information dataset
print("\n5️⃣ Downloading Drug Information dataset...")
try:
    drugs = load_dataset("mpingale/medical-meadow-mediqa", split="train")
    drugs.to_json("data/drug_info/mediqa.jsonl")
    print(f"   ✅ Saved: {len(drugs)} samples")
except Exception as e:
    print(f"   ⚠️ MediQA not available, trying alternative...")
    try:
        drugs = load_dataset("truehealth/healthqa", split="train")
        drugs.to_json("data/drug_info/healthqa.jsonl")
        print(f"   ✅ Saved: {len(drugs)} samples")
    except Exception as e2:
        print(f"   ⚠️ Error: {e2}")

# 6. Pharmacology dataset
print("\n6️⃣ Downloading Pharmacology/Drug QA dataset...")
try:
    pharma = load_dataset("openlifescienceai/medmcqa", split="train")
    # Filter for pharmacology questions
    pharma_subset = pharma.select(range(min(10000, len(pharma))))
    pharma_subset.to_json("data/drug_info/pharma_qa_10k.jsonl")
    print(f"   ✅ Saved: {len(pharma_subset)} samples")
except Exception as e:
    print(f"   ⚠️ Error: {e}")

# 7. FDA Drug Labels (if available)
print("\n7️⃣ Downloading FDA Drug dataset...")
try:
    fda = load_dataset("axiong/pmc_oa_drqa", split="train")
    fda_subset = fda.select(range(min(5000, len(fda))))
    fda_subset.to_json("data/drug_info/drug_qa_5k.jsonl")
    print(f"   ✅ Saved: {len(fda_subset)} samples")
except Exception as e:
    print(f"   ⚠️ Error: {e}")

# ============ SUMMARY ============
print("\n" + "=" * 50)
print("✅ Download complete!")
print("=" * 50)

print("\n📁 General Medicine (data/general_medicine/):")
if os.path.exists("data/general_medicine"):
    for f in os.listdir("data/general_medicine"):
        size = os.path.getsize(f"data/general_medicine/{f}") / (1024*1024)
        print(f"   📄 {f} ({size:.1f} MB)")

print("\n📁 Drug Information (data/drug_info/):")
if os.path.exists("data/drug_info"):
    for f in os.listdir("data/drug_info"):
        size = os.path.getsize(f"data/drug_info/{f}") / (1024*1024)
        print(f"   📄 {f} ({size:.1f} MB)")
