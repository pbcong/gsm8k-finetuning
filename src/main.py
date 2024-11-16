import torch
import time
import evaluate  ## for calculating rouge score
import pandas as pd
import numpy as np

from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    GenerationConfig,
    TrainingArguments,
    Trainer,
)
from train import train_and_evaluate
from utils import print_number_of_trainable_model_parameters
from peft import PeftModel

"""### Load Dataset and LLM"""

huggingface_dataset_name = "knkarthick/dialogsum"
model_name = "google/flan-t5-small"

dataset = load_dataset(huggingface_dataset_name)

train_and_evaluate(model_name, dataset)

