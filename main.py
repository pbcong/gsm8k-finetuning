import pandas as pd
from datasets import load_dataset, DatasetDict
import torch
from utils import print_number_of_trainable_model_parameters, get_time, TaskPrefixDataCollator, TaskPrefixTrainer
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
from dataset import GSM8KDatasetLoader
from metrics import compute_metrics_equation, compute_equation_acc

dataset_loader = GSM8KDatasetLoader()
datasets = dataset_loader.load_from_json()
train_llm_rationales, train_llm_labels = dataset_loader.load_llm_preds(split='train')
test_llm_rationales, test_llm_labels = dataset_loader.load_llm_preds(split='test')
print(len(datasets), len(train_llm_labels))
datasets['train'] = datasets['train'].add_column('llm_label', train_llm_labels)
datasets['test'] = datasets['test'].add_column('llm_label', test_llm_labels)
datasets['train'] = datasets['train'].add_column('rationale', train_llm_rationales)
datasets['test'] = datasets['test'].add_column('rationale', test_llm_rationales)
train_valid_datasets = datasets['train'].train_test_split(test_size=0.1, seed=0)
datasets = DatasetDict({
            'train': train_valid_datasets['train'],
            'valid': train_valid_datasets['test'],
            'test': datasets['test'],
        })

def tokenize_function(examples):
    model_inputs = tokenizer(['predict: ' + text for text in examples['input']], max_length=1000, truncation=True)
    expl_model_inputs = tokenizer(['explain: ' + text for text in examples['input']], max_length=1000, truncation=True)
    model_inputs['expl_input_ids'] = expl_model_inputs['input_ids']
    model_inputs['expl_attention_mask'] = expl_model_inputs['attention_mask']

    with tokenizer.as_target_tokenizer():
        label_output_encodings = tokenizer(examples['label'], max_length=256, truncation=True)
        rationale_output_encodings = tokenizer(examples['rationale'], max_length=256, truncation=True)

    model_inputs['labels'] = label_output_encodings['input_ids']
    model_inputs['aux_labels'] = rationale_output_encodings['input_ids']

    return model_inputs

model = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model)

datasets['train'] = datasets['train'].remove_columns('label')
datasets['train'] = datasets['train'].add_column('label', datasets['train']['llm_label'])

print(datasets['train'].column_names)
tokenized_datasets = datasets.map(
            tokenize_function,
            remove_columns=['input', 'rationale', 'label', 'llm_label'],
            batched=True
        )
compute_metrics = compute_metrics_equation(tokenizer)


original_model = AutoModelForSeq2SeqLM.from_pretrained(
    model, torch_dtype=torch.bfloat16
)

# print_number_of_trainable_model_parameters(original_model)


# print(f'Training: {tokenize_datasets["train"].shape}')
# print(f'Test: {tokenize_datasets["test"].shape}')
# print(tokenize_datasets)

output_dir = f"./ckpt/dialogue-summary-training-{str(get_time())}"

lora_config = LoraConfig(
    r=32,  # rank 32
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
peft_model = peft_model.to("cuda")
# print(print_number_of_trainable_model_parameters(peft_model))

output_dir = f"./ckpt/dialogue-summary-training-{str(get_time())}"
## this is we are again back to the hugging face trainer module
peft_training_args = TrainingArguments(
    output_dir=output_dir,
    auto_find_batch_size=True,
    learning_rate=1e-3,
    logging_steps=1,
    max_steps=1,
    report_to="wandb",  ## can be wandb, but we are reporint to noe
)

compute_metrics = compute_metrics_equation(tokenizer)

trainer_kwargs = {
        'alpha': 0.5,
        'output_rationale': True,
        'model': model,
        'args': peft_training_args,
        'train_dataset': tokenize_datasets["train"],
        'eval_dataset': {'test': tokenize_datasets["test"],},
        'data_collator': data_collator,
        'tokenizer': tokenizer,
        'compute_metrics': compute_metrics,
    }

## this is same except we are using PEFT model instead of regular
peft_trainer = TaskPrefixTrainer(
    model=peft_model, args=peft_training_args, train_dataset=tokenize_datasets["train"]
)

peft_trainer.train()

peft_model_path = './ckpt/peft-dialogue-summary-checkpoint-local'

peft_trainer.model.save_pretrained(peft_model_path)
tokenizer.save_pretrained(peft_model_path)

peft_model_base = AutoModelForSeq2SeqLM.from_pretrained(model, torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(model)

peft_model = PeftModel.from_pretrained(peft_model_base, 
                                      './ckpt/peft-dialogue-summary-checkpoint-local',
                                      torch_dtype=torch.bfloat16,
                                      is_trainable=False)

index = 200 ## randomly pick index
dialogue = data['test'][index]['question']
human_baseline_summary = data['test'][index]['answer']

prompt = f"""{dialogue}"""

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
    prompt = f"""{dialogue}"""

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