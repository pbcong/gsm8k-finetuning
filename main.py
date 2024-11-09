import pandas as pd
import datasets
from datasets import load_dataset
import torch
from utils import print_number_of_trainable_model_parameters, get_time
from dataset import get_tokenize_function
import time
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
import evaluate
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    GenerationConfig,
    TrainingArguments,
    Trainer,
)

dataset = "pbcong/gsm8k_step_by_step"
data = load_dataset(dataset)

model = "google/flan-t5-base"
original_model = AutoModelForSeq2SeqLM.from_pretrained(
    model, torch_dtype=torch.bfloat16
)
tokenizer = AutoTokenizer.from_pretrained(model)

print_number_of_trainable_model_parameters(original_model)

tokenize_datasets = data.map(get_tokenize_function(tokenizer), batched=True)
tokenize_datasets = tokenize_datasets.remove_columns(["question", "answer"])

# print(f'Training: {tokenize_datasets["train"].shape}')
# print(f'Test: {tokenize_datasets["test"].shape}')
# print(tokenize_datasets)

output_dir = f"./dialogue-summary-training-{str(get_time())}"

lora_config = LoraConfig(
    r=32,  # rank 32,
    lora_alpha=32,  ## LoRA Scaling factor
    target_modules=[
        "q",
        "v",
    ],  ## The modules(for example, attention blocks) to apply the LoRA update matrices.
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.SEQ_2_SEQ_LM,  ## flan-t5
)

peft_model = get_peft_model(original_model, lora_config)

# print(print_number_of_trainable_model_parameters(peft_model))

output_dir = f"./dialogue-summary-training-{str(get_time())}"
## this is we are again back to the hugging face trainer module
peft_training_args = TrainingArguments(
    output_dir=output_dir,
    auto_find_batch_size=True,
    learning_rate=1e-3,
    num_train_epochs=1,
    logging_steps=1,
    max_steps=1,
    report_to="none",  ## can be wandb, but we are reporint to noe
)

## this is same except we are using PEFT model instead of regular
peft_trainer = Trainer(
    model=peft_model, args=peft_training_args, train_dataset=tokenize_datasets["train"]
)

peft_trainer.train()

peft_model_path = './peft-dialogue-summary-checkpoint-local'

peft_trainer.model.save_pretrained(peft_model_path)
tokenizer.save_pretrained(peft_model_path)

peft_model_base = AutoModelForSeq2SeqLM.from_pretrained('google/flan-t5-base', torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained('google/flan-t5-base')

peft_model = PeftModel.from_pretrained(peft_model_base, 
                                      './peft-dialogue-summary-checkpoint-local',
                                      torch_dtype=torch.bfloat16,
                                      is_trainable=False)

index = 200 ## randomly pick index
dialogue = data['test'][index]['question']
human_baseline_summary = data['test'][index]['answer']

prompt = f"""
Summarize the following conversation.

{dialogue}

Summary:
"""

input_ids = tokenizer(prompt, return_tensors='pt').input_ids
peft_model = peft_model.to('cuda')
input_ids = input_ids.to('cuda')
original_model_outputs = original_model.generate(input_ids=input_ids, generation_config=GenerationConfig(max_new_tokens=200, num_beams=1))
original_model_text_output = tokenizer.decode(original_model_outputs[0], skip_special_tokens=True)


peft_model_outputs = peft_model.generate(input_ids=input_ids, generation_config=GenerationConfig(max_new_tokens=200, num_beams=1))
peft_model_text_output = tokenizer.decode(peft_model_outputs[0], skip_special_tokens=True)

# print(f'Human Baseline summary: \n{human_baseline_summary}\n')
# print(f'Original Model Output \n{original_model_text_output}\n')
# print(f'Peft Model Output \n{peft_model_text_output}\n')

dialogue = data['test'][0:10]['question']
human_baseline_summaries = data['test'][0:10]['answer']

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