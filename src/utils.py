from datetime import datetime
import pandas as pd
import torch
from typing import Any, Dict, List, Optional, Tuple, Union
from torch import nn
from transformers import DataCollatorForSeq2Seq
from transformers import Seq2SeqTrainer

def get_time():
    # Get the current date and time
    now = datetime.now()

    # Format the date and time
    formatted_date_time = now.strftime("%d%m%y-%H%M%S")
    return formatted_date_time

def print_number_of_trainable_model_parameters(model):
    trainable_model_params = 0
    all_model_params = 0
    for _, param in model.named_parameters():
        all_model_params += param.numel()
        if param.requires_grad:
            trainable_model_params += param.numel()
    print(f'trainable model parameters: {trainable_model_params}\n \
            all model parameters: {all_model_params} \n \
            percentage of trainable model parameters: {(trainable_model_params / all_model_params) * 100} %')
    return

