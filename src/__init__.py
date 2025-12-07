# LLM Medical Fine-Tuning Package
"""
A package for fine-tuning Large Language Models on medical data.
"""

__version__ = "1.0.0"
__author__ = "Your Name"

from .data_processing import MedicalDataProcessor, format_prompt
from .training import MedicalTrainer, load_config
from .inference import MedicalInference
