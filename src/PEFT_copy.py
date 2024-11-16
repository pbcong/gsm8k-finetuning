import torch
import time
import evaluate  ## for calculating rouge score
import pandas as pd
import numpy as np

from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, GenerationConfig, TrainingArguments, Trainer

"""### Load Dataset and LLM"""

huggingface_dataset_name = "knkarthick/dialogsum"

dataset = load_dataset(huggingface_dataset_name)

dataset

model_name = 'google/flan-t5-small'

# bfloat16 mean we are using the small version of flan-t5
original_model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(model_name)

"""It is possible to pull out the number of parameters from the model and find out how many of them are trainable."""

def print_number_of_trainable_model_parameters(model):
    trainable_model_params = 0
    all_model_params = 0
    for _, param in model.named_parameters():
        all_model_params += param.numel()
        if param.requires_grad:
            trainable_model_params += param.numel()
    return f'trainable model parameters: {trainable_model_params}\n \
            all model parameters: {all_model_params} \n \
            percentage of trainable model parameters: {(trainable_model_params / all_model_params) * 100} %'

print(print_number_of_trainable_model_parameters(original_model))

"""# 1. Preprocess the Dialog-Summary Dataset

We need to convert dialog-summary into explicit instruction for LLM. Prepend an instruction to the start of the dialog with Summarize the following conversation and to the start of the summary with summary as follows:
<br>
Training prompt (dialogue):

```
Summarize the following conversation.

    Chris: This is his part of conversation
    Antje: This is her part of conversation
    
Summary:
```

Training response (summary):

```
Both Chrish and Antje participated in the conversation.
```

Then preprocess the prompt-response dataset into token and pull out their input_ids (1 per token)
"""

def tokeninze_function(example):
    start_prompt = 'Summarize the following conversation. \n\n'
    end_prompt = '\n\nSummary: '
    prompt = [start_prompt + dialogue + end_prompt for dialogue in example["dialogue"]]
    example['input_ids'] = tokenizer(prompt, padding='max_length', truncation=True,
                                     return_tensors='pt').input_ids
    example['labels'] = tokenizer(example['summary'], padding='max_length', truncation=True,
                                 return_tensors='pt').input_ids

    return example

# The Dataseta ctually contains 3 diff splits: train, validation, and test.
# The tokenize_function code is handling all data across all splits in batches
tokenize_datasets = dataset.map(tokeninze_function, batched=True)
tokenize_datasets = tokenize_datasets.remove_columns(['id', 'topic', 'dialogue',
                                                     'summary'])

"""To save some time in the lab, you will subsample the dataset:"""

# tokenize_datasets = tokenize_datasets.filter(lambda exmaple, index: index % 100 == 0,
#                                             with_indices=True)

"""Check the shape of all three dataset"""

print(f'Training: {tokenize_datasets["train"].shape}')
print(f'Valdiation: {tokenize_datasets["validation"].shape}')
print(f'Test: {tokenize_datasets["test"].shape}')
print(tokenize_datasets)

"""Use the Hugging face builtin `Trainer` library. Pass the preprocessed dataset with reference to original model."""

output_dir = f'./dialogue-summary-training-{str(int(time.time()))}'

# import os
# os.environ['WANDB_DISABLED'] = 'true'

"""## 2. Parameter Efficient Fine Tuning

### 2.1. PEFT using LoRA technique
LoRA required setting up a new layer of adapter. PEFT freezes the underlying LLM parameters and train only adapter.
"""

lora_config = LoraConfig(r=32, #rank 32,
                         lora_alpha=32, ## LoRA Scaling factor
                         target_modules=['q', 'v'], ## The modules(for example, attention blocks) to apply the LoRA update matrices.
                         lora_dropout = 0.05,
                         bias='none',
                         task_type=TaskType.SEQ_2_SEQ_LM ## flan-t5
)

# ## target_modules='q', This represents the value projection layer in the transformer model. The value projection layer transforms input tokens into value vectors,
# # which are the actual values that are attended to based on the attention scores computed from query and key vectors.

# ## target_modules='v',This typically refers to the query projection layer in a transformer-based model. The query projection layer is responsible for transforming
# # input tokens into query vectors, which are used to attend to other tokens in the sequence during self-attention mechanism.

"""Add LoRA parameter to original model"""

peft_model = get_peft_model(original_model, lora_config)
peft_model = peft_model.to('cuda')
print(print_number_of_trainable_model_parameters(peft_model))

"""### 2.2. Train PEFT Adapter"""

# model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

output_dir = f'./dialogue-summary-training-{str(int(time.time()))}'
## this is we are again back to the hugging face trainer module
peft_training_args = TrainingArguments(output_dir=output_dir,
                                       auto_find_batch_size=True,
                                       learning_rate=1e-3,
                                       num_train_epochs=1,
                                       logging_steps=1,
                                       max_steps=1,
                                        report_to='wandb' ## can be wandb, but we are reporint to noe
                )

## this is same except we are using PEFT model instead of regular
peft_trainer = Trainer(model=peft_model,
                      args=peft_training_args,
                      train_dataset=tokenize_datasets['train'],
                      eval_dataset=tokenize_datasets['test'],
                 )

peft_trainer.train()

peft_model_path = './peft-dialogue-summary-checkpoint-local'

peft_trainer.model.save_pretrained(peft_model_path)
tokenizer.save_pretrained(peft_model_path)

# !ls -alh ./peft-dialogue-summary-checkpoint-local/adapter_model.bin

"""This is just a 14MB model

Inferencing from PEFT fine tuned model
"""

from peft import PeftModel

peft_model_base = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(model_name)
# peft_model = AutoModelForSeq2SeqLM.from_pretrained(peft_model_path, torch_dtype=torch.bfloat16)
peft_model = PeftModel.from_pretrained(peft_model_base,
                                      './peft-dialogue-summary-checkpoint-local',
                                      torch_dtype=torch.bfloat16,
                                      is_trainable=False) ## is_trainable mean just a forward pass jsut to get a sumamry

index = 200 ## randomly pick index
dialogue = dataset['test'][index]['dialogue']
human_baseline_summary = dataset['test'][index]['summary']

prompt = f"""
Summarize the following conversation.

{dialogue}

Summary:
"""

input_ids = tokenizer(prompt, return_tensors='pt').input_ids
peft_model = peft_model.to('cuda')
input_ids = input_ids.to('cuda')
original_model = original_model.to('cuda')
original_model_outputs = original_model.generate(input_ids=input_ids, generation_config=GenerationConfig(max_new_tokens=200, num_beams=1))
original_model_text_output = tokenizer.decode(original_model_outputs[0], skip_special_tokens=True)


peft_model_outputs = peft_model.generate(input_ids=input_ids, generation_config=GenerationConfig(max_new_tokens=200, num_beams=1))
peft_model_text_output = tokenizer.decode(peft_model_outputs[0], skip_special_tokens=True)

print(f'Human Baseline summary: \n{human_baseline_summary}\n')
print(f'Original Model Output \n{original_model_text_output}\n')
print(f'Peft Model Output \n{peft_model_text_output}\n')

"""### 2.3. Evaluate model quantitavely"""

dialogue = dataset['test'][200]['dialogue']
human_baseline_summaries = dataset['test'][200]['summary']

original_model_summaries = []
peft_model_summaries = []

for _, dialogue in enumerate(dialogue):
    prompt = f"""
    Summarize the following conversations.

    {dialogue}

    Summary: """

    input_ids = tokenizer(prompt, return_tensors='pt').input_ids
    input_ids = input_ids.to('cuda')
    original_model_outputs = original_model.generate(input_ids=input_ids, generation_config=GenerationConfig(max_new_tokens=200, num_beams=1))
    original_model_text_output = tokenizer.decode(original_model_outputs[0], skip_special_tokens=True)
    original_model_summaries.append(original_model_text_output)


    peft_model_outputs = peft_model.generate(input_ids=input_ids, generation_config=GenerationConfig(max_new_tokens=200, num_beams=1))
    peft_model_text_output = tokenizer.decode(peft_model_outputs[0], skip_special_tokens=True)
    peft_model_summaries.append(peft_model_text_output)


zipped_summaries = list(zip(human_baseline_summaries, original_model_summaries,
                           peft_model_summaries))

df = pd.DataFrame(zipped_summaries, columns=['human_baseline_summaries', 'original_model_summaries', 'peft_model_summaries'])
df



"""Calculate rouge Metrics score"""

rouge = evaluate.load('rouge')

original_model_results = rouge.compute(predictions=original_model_summaries,
                                       references=human_baseline_summaries[0: len(original_model_summaries)],
                                      use_aggregator=True,
                                      use_stemmer=True)

peft_model_results = rouge.compute(predictions=peft_model_summaries,
                                    references=human_baseline_summaries[0: len(peft_model_summaries)],
                                    use_aggregator=True,
                                    use_stemmer=True)

print(f'Original Model: \n{original_model_results}\n')
print(f'PEFT Model: \n{peft_model_results}\n')

"""PEFT rouge score is better than the original model. It uses the less resources and improves in comparison to original model. <br>
These are just a few examples but you can  imagine its impact at scale and how much it saves in terms of resources, and times.
"""

