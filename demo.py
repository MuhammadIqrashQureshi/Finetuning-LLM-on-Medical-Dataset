"""
Quick Demo Script for Medical LLM
Run: python demo.py
"""
import requests
import json

def query_medical_llm(question, model="medical-llama3"):
    """Query the medical LLM via Ollama API"""
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model,
                "prompt": question,
                "stream": False,
                "options": {"temperature": 0.7}
            },
            timeout=120
        )
        if response.status_code == 200:
            return response.json().get("response", "No response")
        return f"Error: {response.status_code}"
    except Exception as e:
        return f"Error: {e}"

def main():
    print("=" * 60)
    print("🏥 Medical LLM Assistant - Demo")
    print("=" * 60)
    print("Model: medical-llama3 (Llama 3 8B fine-tuned on medical data)")
    print("-" * 60)
    
    # Demo questions
    questions = [
        "What are the common symptoms of Type 2 Diabetes?",
        "What is the mechanism of action of Metformin?",
        "How is hypertension typically treated?",
    ]
    
    for i, q in enumerate(questions, 1):
        print(f"\n📋 Question {i}: {q}")
        print("-" * 40)
        answer = query_medical_llm(q)
        print(f"🤖 Answer: {answer[:500]}...")
        print()
    
    # Interactive mode
    print("\n" + "=" * 60)
    print("Interactive Mode (type 'quit' to exit)")
    print("=" * 60)
    
    while True:
        question = input("\n👤 Your question: ").strip()
        if question.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        if question:
            print("\n🤖 Answer:", query_medical_llm(question))

if __name__ == "__main__":
    main()
