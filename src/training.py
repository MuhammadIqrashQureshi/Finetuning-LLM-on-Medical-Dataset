"""
Training Utilities for Medical LLM Fine-Tuning
"""

import json
import os
from typing import Dict, Optional, Any
from pathlib import Path

import torch
import yaml
from datasets import Dataset


def load_config(config_path: str = "config/training_config.yaml") -> Dict:
    """
    Load training configuration from YAML file.
    
    Args:
        config_path: Path to the configuration file
    
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def get_device_info() -> Dict:
    """
    Get information about available compute devices.
    
    Returns:
        Dictionary with device information
    """
    info = {
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }
    
    if torch.cuda.is_available():
        info["cuda_device_name"] = torch.cuda.get_device_name(0)
        gpu_props = torch.cuda.get_device_properties(0)
        info["total_memory_gb"] = round(gpu_props.total_memory / 1024**3, 2)
        info["bf16_supported"] = torch.cuda.is_bf16_supported()
    
    return info


class MedicalTrainer:
    """
    Trainer class for medical LLM fine-tuning using Unsloth.
    """
    
    def __init__(self, config: Optional[Dict] = None, config_path: Optional[str] = None):
        """
        Initialize the trainer.
        
        Args:
            config: Configuration dictionary (optional)
            config_path: Path to configuration file (optional)
        """
        if config is None and config_path is not None:
            config = load_config(config_path)
        elif config is None:
            config = load_config()
        
        self.config = config
        self.model = None
        self.tokenizer = None
        self.trainer = None
        self.trainer_stats = None
        
        # Auto-detect fp16/bf16 support
        if torch.cuda.is_available():
            bf16_supported = torch.cuda.is_bf16_supported()
            self.config["training_config"]["fp16"] = not bf16_supported
            self.config["training_config"]["bf16"] = bf16_supported
    
    def load_model(self):
        """
        Load the base model and tokenizer using Unsloth.
        """
        try:
            from unsloth import FastLanguageModel
        except ImportError:
            raise ImportError(
                "Unsloth is not installed. Please install it with:\n"
                "pip install 'unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git'"
            )
        
        model_config = self.config["model_config"]
        
        print(f"Loading model: {model_config['base_model']}")
        
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_config["base_model"],
            max_seq_length=model_config["max_seq_length"],
            dtype=getattr(torch, model_config.get("dtype", "float16")),
            load_in_4bit=model_config.get("load_in_4bit", True),
        )
        
        print("Model loaded successfully!")
        return self.model, self.tokenizer
    
    def setup_peft(self):
        """
        Setup PEFT (LoRA) for the model.
        """
        try:
            from unsloth import FastLanguageModel
        except ImportError:
            raise ImportError("Unsloth is not installed.")
        
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        lora_config = self.config["lora_config"]
        
        print("Setting up LoRA/QLoRA...")
        
        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=lora_config["r"],
            target_modules=lora_config["target_modules"],
            lora_alpha=lora_config["lora_alpha"],
            lora_dropout=lora_config["lora_dropout"],
            bias=lora_config["bias"],
            use_gradient_checkpointing=lora_config.get("use_gradient_checkpointing", True),
            random_state=42,
            use_rslora=lora_config.get("use_rslora", False),
            use_dora=lora_config.get("use_dora", False),
            loftq_config=lora_config.get("loftq_config", None),
        )
        
        print("PEFT setup complete!")
        return self.model
    
    def create_trainer(self, dataset: Dataset):
        """
        Create the SFTTrainer for fine-tuning.
        
        Args:
            dataset: Training dataset
        """
        try:
            from trl import SFTTrainer
            from transformers import TrainingArguments
        except ImportError:
            raise ImportError("trl and transformers are required.")
        
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        training_config = self.config["training_config"]
        dataset_config = self.config["training_dataset"]
        model_config = self.config["model_config"]
        
        # Create output directory
        os.makedirs(training_config["output_dir"], exist_ok=True)
        
        print("Creating trainer...")
        
        self.trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=dataset,
            dataset_text_field=dataset_config["input_field"],
            max_seq_length=model_config["max_seq_length"],
            dataset_num_proc=2,
            packing=False,
            args=TrainingArguments(
                per_device_train_batch_size=training_config["per_device_train_batch_size"],
                gradient_accumulation_steps=training_config["gradient_accumulation_steps"],
                warmup_steps=training_config["warmup_steps"],
                max_steps=training_config["max_steps"] if training_config["max_steps"] > 0 else -1,
                num_train_epochs=training_config["num_train_epochs"],
                learning_rate=training_config["learning_rate"],
                fp16=training_config["fp16"],
                bf16=training_config["bf16"],
                logging_steps=training_config["logging_steps"],
                optim=training_config["optim"],
                weight_decay=training_config["weight_decay"],
                lr_scheduler_type=training_config["lr_scheduler_type"],
                seed=training_config.get("seed", 42),
                output_dir=training_config["output_dir"],
            ),
        )
        
        print("Trainer created!")
        return self.trainer
    
    def train(self) -> Dict:
        """
        Run the training loop.
        
        Returns:
            Training statistics
        """
        if self.trainer is None:
            raise ValueError("Trainer not created. Call create_trainer() first.")
        
        print("Starting training...")
        print(self._get_memory_stats())
        
        self.trainer_stats = self.trainer.train()
        
        print("\nTraining complete!")
        print(self._get_memory_stats())
        
        return self.trainer_stats
    
    def _get_memory_stats(self) -> str:
        """Get GPU memory statistics."""
        if not torch.cuda.is_available():
            return "CUDA not available"
        
        gpu_stats = torch.cuda.get_device_properties(0)
        reserved_memory = round(torch.cuda.max_memory_reserved() / 1024**3, 2)
        max_memory = round(gpu_stats.total_memory / 1024**3, 2)
        used_memory = round(torch.cuda.max_memory_allocated() / 1024**3, 2)
        
        return (
            f"GPU Memory: {used_memory}GB used / {max_memory}GB total "
            f"({round(used_memory/max_memory*100, 1)}%)"
        )
    
    def save_model(self, save_path: Optional[str] = None):
        """
        Save the fine-tuned model.
        
        Args:
            save_path: Path to save the model
        """
        if save_path is None:
            save_path = os.path.join(
                self.config["training_config"]["output_dir"],
                self.config["model_config"]["finetuned_model"]
            )
        
        print(f"Saving model to {save_path}...")
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        print("Model saved!")
    
    def save_trainer_stats(self, save_path: Optional[str] = None):
        """
        Save training statistics to JSON.
        
        Args:
            save_path: Path to save the stats
        """
        if self.trainer_stats is None:
            print("No training stats to save.")
            return
        
        if save_path is None:
            save_path = os.path.join(
                self.config["training_config"]["output_dir"],
                "trainer_stats.json"
            )
        
        # Convert trainer stats to serializable format
        stats_dict = {
            "training_loss": self.trainer_stats.training_loss,
            "global_step": self.trainer_stats.global_step,
            "metrics": self.trainer_stats.metrics if hasattr(self.trainer_stats, 'metrics') else {},
        }
        
        with open(save_path, 'w') as f:
            json.dump(stats_dict, f, indent=4)
        
        print(f"Training stats saved to {save_path}")
    
    def push_to_hub(self, repo_name: Optional[str] = None):
        """
        Push the model to HuggingFace Hub.
        
        Args:
            repo_name: Repository name on HuggingFace
        """
        if repo_name is None:
            username = self.config.get("hugging_face_username", "")
            model_name = self.config["model_config"]["finetuned_model"]
            repo_name = f"{username}/{model_name}" if username else model_name
        
        print(f"Pushing to HuggingFace Hub: {repo_name}")
        self.model.push_to_hub(repo_name, tokenizer=self.tokenizer)
        print("Model pushed to Hub!")


def quick_train(
    dataset: Dataset,
    config_path: str = "config/training_config.yaml",
    save_model: bool = True,
    push_to_hub: bool = False
) -> MedicalTrainer:
    """
    Quick training function that handles the full training pipeline.
    
    Args:
        dataset: Training dataset
        config_path: Path to configuration file
        save_model: Whether to save the model after training
        push_to_hub: Whether to push to HuggingFace Hub
    
    Returns:
        Trained MedicalTrainer instance
    """
    trainer = MedicalTrainer(config_path=config_path)
    
    # Load and setup model
    trainer.load_model()
    trainer.setup_peft()
    
    # Create trainer and train
    trainer.create_trainer(dataset)
    trainer.train()
    
    # Save results
    trainer.save_trainer_stats()
    
    if save_model:
        trainer.save_model()
    
    if push_to_hub:
        trainer.push_to_hub()
    
    return trainer


if __name__ == "__main__":
    # Print device info
    device_info = get_device_info()
    print("Device Information:")
    for key, value in device_info.items():
        print(f"  {key}: {value}")
