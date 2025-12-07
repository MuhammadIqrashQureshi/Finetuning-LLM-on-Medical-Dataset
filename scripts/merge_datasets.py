"""
Merge and Optimize Datasets for Fast Training
Creates a balanced 25k sample dataset from all sources
"""
import os
import json
import random

random.seed(42)

output_dir = "data/merged"
os.makedirs(output_dir, exist_ok=True)

print("🔀 Creating Optimized Merged Dataset...\n")

all_samples = []

# Define datasets and how many samples to take from each
datasets_to_merge = [
    # Clinical/Diagnostic
    ("data/clinical/medical_llama3_short.jsonl", 2000, "prompt"),  # All - pre-formatted
    ("data/clinical/medquad.jsonl", 3000, None),
    ("data/clinical/chatdoctor_10k.jsonl", 2000, None),
    ("data/clinical/medical_meadow_wikidoc.jsonl", 1500, None),
    
    # General Medicine
    ("data/general_medicine/medical_llama3_full.jsonl", 3000, "prompt"),  # Pre-formatted
    ("data/general_medicine/medmcqa_20k.jsonl", 1500, None),
    
    # Drug Info
    ("data/drug_info/medical_flashcards.jsonl", 1000, None),
    ("data/drug_info/pharma_qa_10k.jsonl", 1000, None),
]

def format_to_llama3(item):
    """Convert various formats to Llama 3 instruction format"""
    # If already has prompt field, use it
    if "prompt" in item and item["prompt"]:
        return item["prompt"]
    
    # Extract question and answer from various formats
    question = ""
    answer = ""
    
    # Common field names for questions
    for q_field in ["question", "input", "instruction", "query", "Question"]:
        if q_field in item and item[q_field]:
            question = item[q_field]
            break
    
    # Common field names for answers
    for a_field in ["answer", "output", "response", "Answer", "text"]:
        if a_field in item and item[a_field]:
            answer = item[a_field]
            break
    
    # Handle ChatDoctor format
    if "instruction" in item and "input" in item and "output" in item:
        question = item["instruction"]
        if item["input"]:
            question += f"\n{item['input']}"
        answer = item["output"]
    
    # Handle MCQA format (multiple choice)
    if "opa" in item:  # medmcqa format
        question = item.get("question", "")
        options = []
        for opt in ["opa", "opb", "opc", "opd"]:
            if opt in item:
                options.append(item[opt])
        if options:
            question += "\nOptions: " + ", ".join(options)
        correct = item.get("cop", 0)  # correct option position
        if options and correct < len(options):
            answer = f"The correct answer is: {options[correct]}"
            if "exp" in item and item["exp"]:
                answer += f"\n\nExplanation: {item['exp']}"
    
    if not question or not answer:
        return None
    
    # Format as Llama 3 instruction
    prompt = f"""<|start_header_id|>system<|end_header_id|>

You are a helpful medical assistant. Answer the question accurately and professionally.<|eot_id|><|start_header_id|>user<|end_header_id|>

{question.strip()}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{answer.strip()}<|eot_id|>"""
    
    return prompt


# Process each dataset
for filepath, max_samples, prompt_field in datasets_to_merge:
    if not os.path.exists(filepath):
        print(f"   ⚠️ Not found: {filepath}")
        continue
    
    print(f"📄 Processing: {filepath}")
    
    samples = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                item = json.loads(line)
                
                if prompt_field:
                    # Already formatted
                    prompt = item.get(prompt_field, "")
                else:
                    # Need to format
                    prompt = format_to_llama3(item)
                
                if prompt and len(prompt) > 100:  # Filter too short
                    samples.append({"prompt": prompt})
            except:
                continue
    
    # Random sample
    if len(samples) > max_samples:
        samples = random.sample(samples, max_samples)
    
    all_samples.extend(samples)
    print(f"   ✅ Added: {len(samples)} samples")

# Shuffle all samples
random.shuffle(all_samples)

# Save merged dataset
output_file = os.path.join(output_dir, "medical_merged_15k.jsonl")
with open(output_file, 'w', encoding='utf-8') as f:
    for sample in all_samples:
        f.write(json.dumps(sample, ensure_ascii=False) + "\n")

size_mb = os.path.getsize(output_file) / (1024 * 1024)

print(f"\n{'='*50}")
print(f"✅ Merged dataset created!")
print(f"{'='*50}")
print(f"📄 File: {output_file}")
print(f"📊 Total samples: {len(all_samples)}")
print(f"💾 Size: {size_mb:.1f} MB")
print(f"⏱️ Estimated training time: ~2-3 hours (1 epoch)")
