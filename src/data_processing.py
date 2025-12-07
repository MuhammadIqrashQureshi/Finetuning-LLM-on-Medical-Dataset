"""
Data Processing Utilities for Medical LLM Fine-Tuning
"""

import json
from typing import Dict, List, Optional
from datasets import Dataset, load_dataset


def format_prompt_llama3(question: str, answer: str = "") -> str:
    """
    Format a prompt for Llama 3 Instruct model.
    
    Args:
        question: The medical question
        answer: The answer (empty for inference)
    
    Returns:
        Formatted prompt string
    """
    system_message = "Answer the question truthfully, you are a medical professional."
    
    prompt = f"""<|start_header_id|>system<|end_header_id|>

{system_message}<|eot_id|><|start_header_id|>user<|end_header_id|}

{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{answer}"""
    
    return prompt


def format_prompt_llama2(question: str, answer: str = "") -> str:
    """
    Format a prompt for Llama 2 Chat model.
    
    Args:
        question: The medical question
        answer: The answer (empty for inference)
    
    Returns:
        Formatted prompt string
    """
    system_message = "Answer the question truthfully, you are a medical professional."
    
    prompt = f"""<s>[INST] <<SYS>>
{system_message}
<</SYS>>

{question} [/INST] {answer}"""
    
    return prompt


def format_prompt_mistral(question: str, answer: str = "") -> str:
    """
    Format a prompt for Mistral Instruct model.
    
    Args:
        question: The medical question
        answer: The answer (empty for inference)
    
    Returns:
        Formatted prompt string
    """
    prompt = f"""<s>[INST] Answer the question truthfully, you are a medical professional.

{question} [/INST] {answer}"""
    
    return prompt


def format_prompt_gemma(question: str, answer: str = "") -> str:
    """
    Format a prompt for Gemma Instruct model.
    
    Args:
        question: The medical question
        answer: The answer (empty for inference)
    
    Returns:
        Formatted prompt string
    """
    prompt = f"""<start_of_turn>user
Answer the question truthfully, you are a medical professional.

{question}<end_of_turn>
<start_of_turn>model
{answer}"""
    
    return prompt


# Mapping of model types to their format functions
FORMAT_FUNCTIONS = {
    "llama3": format_prompt_llama3,
    "llama2": format_prompt_llama2,
    "mistral": format_prompt_mistral,
    "gemma": format_prompt_gemma,
}


def format_prompt(question: str, answer: str = "", model_type: str = "llama3") -> str:
    """
    Format a prompt based on model type.
    
    Args:
        question: The medical question
        answer: The answer (empty for inference)
        model_type: Type of model ("llama3", "llama2", "mistral", "gemma")
    
    Returns:
        Formatted prompt string
    """
    format_func = FORMAT_FUNCTIONS.get(model_type, format_prompt_llama3)
    return format_func(question, answer)


class MedicalDataProcessor:
    """
    Process medical datasets for fine-tuning.
    """
    
    def __init__(self, model_type: str = "llama3"):
        """
        Initialize the data processor.
        
        Args:
            model_type: Type of model for prompt formatting
        """
        self.model_type = model_type
        self.format_func = FORMAT_FUNCTIONS.get(model_type, format_prompt_llama3)
    
    def load_hf_dataset(
        self, 
        dataset_name: str, 
        split: str = "train",
        sample_size: Optional[int] = None
    ) -> Dataset:
        """
        Load a dataset from HuggingFace.
        
        Args:
            dataset_name: Name of the HuggingFace dataset
            split: Dataset split to load
            sample_size: Optional number of samples to use
        
        Returns:
            HuggingFace Dataset object
        """
        dataset = load_dataset(dataset_name, split=split)
        
        if sample_size is not None and sample_size < len(dataset):
            dataset = dataset.select(range(sample_size))
        
        return dataset
    
    def load_json_dataset(self, file_path: str) -> Dataset:
        """
        Load a dataset from a JSON file.
        
        Args:
            file_path: Path to the JSON file
        
        Returns:
            HuggingFace Dataset object
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return Dataset.from_list(data)
    
    def process_qa_dataset(
        self, 
        dataset: Dataset,
        question_field: str = "question",
        answer_field: str = "answer",
        output_field: str = "prompt"
    ) -> Dataset:
        """
        Process a Q&A dataset into instruction format.
        
        Args:
            dataset: Input dataset
            question_field: Name of the question column
            answer_field: Name of the answer column
            output_field: Name for the output prompt column
        
        Returns:
            Processed dataset with formatted prompts
        """
        def format_example(example):
            question = example[question_field]
            answer = example[answer_field]
            example[output_field] = self.format_func(question, answer)
            return example
        
        return dataset.map(format_example)
    
    def get_dataset_statistics(self, dataset: Dataset) -> Dict:
        """
        Get statistics about a dataset.
        
        Args:
            dataset: Input dataset
        
        Returns:
            Dictionary with dataset statistics
        """
        stats = {
            "num_samples": len(dataset),
            "columns": list(dataset.column_names),
        }
        
        # Calculate average lengths if text columns exist
        for col in dataset.column_names:
            if isinstance(dataset[0][col], str):
                lengths = [len(example[col]) for example in dataset]
                stats[f"{col}_avg_length"] = sum(lengths) / len(lengths)
                stats[f"{col}_max_length"] = max(lengths)
                stats[f"{col}_min_length"] = min(lengths)
        
        return stats


def load_medical_meadow_wikidoc(sample_size: Optional[int] = None) -> Dataset:
    """
    Load the Medical Meadow WikiDoc dataset.
    
    Args:
        sample_size: Optional number of samples to load
    
    Returns:
        HuggingFace Dataset
    """
    dataset = load_dataset("medalpaca/medical_meadow_wikidoc", split="train")
    
    if sample_size is not None and sample_size < len(dataset):
        dataset = dataset.select(range(sample_size))
    
    return dataset


def load_preprocessed_medical_dataset(
    model_type: str = "llama3",
    short: bool = True
) -> Dataset:
    """
    Load pre-processed medical instruction datasets from HuggingFace.
    
    Args:
        model_type: Type of model ("llama3", "llama2", "mistral", "gemma")
        short: Whether to load the short version (2000 samples)
    
    Returns:
        HuggingFace Dataset
    """
    dataset_mapping = {
        "llama3": "Shekswess/medical_llama3_instruct_dataset",
        "llama2": "Shekswess/medical_llama2_instruct_dataset",
        "mistral": "Shekswess/medical_mistral_instruct_dataset",
        "gemma": "Shekswess/medical_gemma_instruct_dataset",
    }
    
    dataset_name = dataset_mapping.get(model_type, dataset_mapping["llama3"])
    
    if short:
        dataset_name = dataset_name + "_short"
    
    return load_dataset(dataset_name, split="train")


if __name__ == "__main__":
    # Example usage
    processor = MedicalDataProcessor(model_type="llama3")
    
    # Load sample data
    sample_dataset = processor.load_json_dataset("data/sample_medical_data.json")
    print(f"Loaded {len(sample_dataset)} samples")
    
    # Process the dataset
    processed = processor.process_qa_dataset(sample_dataset)
    print(f"\nSample formatted prompt:\n{processed[0]['prompt'][:500]}...")
    
    # Get statistics
    stats = processor.get_dataset_statistics(processed)
    print(f"\nDataset statistics: {stats}")
