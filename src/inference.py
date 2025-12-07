"""
Inference Utilities for Medical LLM
"""

import os
from typing import Optional, List, Dict
import torch


class MedicalInference:
    """
    Inference class for the fine-tuned medical LLM.
    """
    
    def __init__(
        self,
        model_path: str,
        max_seq_length: int = 2048,
        load_in_4bit: bool = True,
        device: str = "cuda"
    ):
        """
        Initialize the inference engine.
        
        Args:
            model_path: Path to the fine-tuned model
            max_seq_length: Maximum sequence length
            load_in_4bit: Whether to load in 4-bit quantization
            device: Device to use ("cuda" or "cpu")
        """
        self.model_path = model_path
        self.max_seq_length = max_seq_length
        self.load_in_4bit = load_in_4bit
        self.device = device
        
        self.model = None
        self.tokenizer = None
    
    def load_model(self):
        """
        Load the fine-tuned model for inference.
        """
        try:
            from unsloth import FastLanguageModel
        except ImportError:
            raise ImportError(
                "Unsloth is not installed. Please install it with:\n"
                "pip install 'unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git'"
            )
        
        print(f"Loading model from {self.model_path}...")
        
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.model_path,
            max_seq_length=self.max_seq_length,
            dtype=torch.float16,
            load_in_4bit=self.load_in_4bit,
        )
        
        # Enable fast inference mode
        FastLanguageModel.for_inference(self.model)
        
        print("Model loaded and ready for inference!")
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
        use_cache: bool = True,
    ) -> str:
        """
        Generate a response for a given prompt.
        
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            do_sample: Whether to use sampling
            use_cache: Whether to use KV cache
        
        Returns:
            Generated response
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        inputs = self.tokenizer(
            [prompt],
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                use_cache=use_cache,
            )
        
        response = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        
        return response
    
    def answer_medical_question(
        self,
        question: str,
        model_type: str = "llama3",
        max_new_tokens: int = 512,
        **kwargs
    ) -> str:
        """
        Answer a medical question using the appropriate prompt format.
        
        Args:
            question: Medical question to answer
            model_type: Type of model for prompt formatting
            max_new_tokens: Maximum tokens in response
            **kwargs: Additional generation parameters
        
        Returns:
            Model's answer
        """
        from .data_processing import format_prompt
        
        prompt = format_prompt(question, answer="", model_type=model_type)
        response = self.generate(prompt, max_new_tokens=max_new_tokens, **kwargs)
        
        # Extract just the answer part (after the prompt)
        if model_type == "llama3":
            if "<|start_header_id|>assistant<|end_header_id|>" in response:
                response = response.split("<|start_header_id|>assistant<|end_header_id|>")[-1]
        elif model_type == "llama2":
            if "[/INST]" in response:
                response = response.split("[/INST]")[-1]
        elif model_type == "mistral":
            if "[/INST]" in response:
                response = response.split("[/INST]")[-1]
        elif model_type == "gemma":
            if "<start_of_turn>model" in response:
                response = response.split("<start_of_turn>model")[-1]
        
        return response.strip()
    
    def batch_generate(
        self,
        prompts: List[str],
        max_new_tokens: int = 256,
        **kwargs
    ) -> List[str]:
        """
        Generate responses for multiple prompts.
        
        Args:
            prompts: List of input prompts
            max_new_tokens: Maximum tokens per response
            **kwargs: Additional generation parameters
        
        Returns:
            List of generated responses
        """
        responses = []
        for prompt in prompts:
            response = self.generate(prompt, max_new_tokens=max_new_tokens, **kwargs)
            responses.append(response)
        return responses


def create_ollama_modelfile(
    model_path: str,
    output_path: str = "Modelfile",
    model_name: str = "medical-llama3",
    system_prompt: Optional[str] = None
) -> str:
    """
    Create an Ollama Modelfile for the fine-tuned model.
    
    Args:
        model_path: Path to the GGUF model file
        output_path: Path to save the Modelfile
        model_name: Name for the Ollama model
        system_prompt: Custom system prompt
    
    Returns:
        Path to the created Modelfile
    """
    if system_prompt is None:
        system_prompt = """You are a helpful medical assistant trained to answer medical questions accurately and professionally. 

Important: Always recommend consulting a qualified healthcare professional for actual medical advice. 
The information provided is for educational purposes only and should not be used for self-diagnosis or treatment."""
    
    modelfile_content = f'''# Modelfile for {model_name}
# Medical Fine-tuned LLM

FROM {model_path}

# Set the system prompt
SYSTEM """
{system_prompt}
"""

# Model parameters
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER repeat_penalty 1.1
PARAMETER num_ctx 2048

# Template for Llama 3 format
TEMPLATE """<|start_header_id|>system<|end_header_id|>

{{{{ .System }}}}<|eot_id|><|start_header_id|>user<|end_header_id|>

{{{{ .Prompt }}}}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
'''
    
    with open(output_path, 'w') as f:
        f.write(modelfile_content)
    
    print(f"Modelfile created at {output_path}")
    print(f"\nTo create the Ollama model, run:")
    print(f"  ollama create {model_name} -f {output_path}")
    
    return output_path


def export_to_gguf(
    model_path: str,
    output_path: str,
    quantization: str = "q4_k_m"
) -> str:
    """
    Export a model to GGUF format for Ollama.
    
    Args:
        model_path: Path to the fine-tuned model
        output_path: Path for the GGUF output
        quantization: Quantization method
    
    Returns:
        Path to the exported GGUF file
    """
    try:
        from unsloth import FastLanguageModel
    except ImportError:
        raise ImportError("Unsloth is required for GGUF export.")
    
    print(f"Loading model from {model_path}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=2048,
        dtype=torch.float16,
        load_in_4bit=True,
    )
    
    print(f"Exporting to GGUF with {quantization} quantization...")
    
    # Save as GGUF
    model.save_pretrained_gguf(
        output_path,
        tokenizer,
        quantization_method=quantization
    )
    
    gguf_file = os.path.join(output_path, f"model-{quantization}.gguf")
    print(f"Model exported to {gguf_file}")
    
    return gguf_file


if __name__ == "__main__":
    # Example: Create a Modelfile template
    create_ollama_modelfile(
        model_path="./outputs/llama-3-8b-medical-q4_k_m.gguf",
        output_path="Modelfile",
        model_name="medical-llama3"
    )
