# Copyright 2023 The Distilling-step-by-step authors

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import argparse
import re
import json
import numpy as np
import datasets
from datasets import Dataset, DatasetDict, load_dataset


DATASET_ROOT = 'datasets'


class DatasetLoader(object):
    def __init__(self, dataset_name, source_dataset_name, dataset_version, has_valid, split_map,
                 batch_size, train_batch_idxs, test_batch_idxs, valid_batch_idxs=None):
        self.data_root = DATASET_ROOT
        self.dataset_name = dataset_name
        self.source_dataset_name = source_dataset_name
        self.dataset_version = dataset_version
        self.has_valid = has_valid
        self.split_map = split_map

        self.batch_size = batch_size
        self.train_batch_idxs = train_batch_idxs
        self.test_batch_idxs = test_batch_idxs
        self.valid_batch_idxs = valid_batch_idxs
        
        assert self.split_map is not None    


    def load_from_source(self):
        if self.source_dataset_name is None:
            self.source_dataset_name = self.dataset_name
        if self.dataset_version is None:
            datasets = load_dataset(self.source_dataset_name)
        else:
            datasets = load_dataset(self.source_dataset_name, self.dataset_version)
        return datasets


    def to_json(self, datasets):
        for k, v in self.split_map.items():
            datasets[v].to_json(f'{self.data_root}/{self.dataset_name}/{self.dataset_name}_{k}.json')


    def load_from_json(self):
        data_files = {
            'train': f'{self.data_root}/{self.dataset_name}/{self.dataset_name}_train.json',
            'test': f'{self.data_root}/{self.dataset_name}/{self.dataset_name}_test.json',
        }

        if self.has_valid:
            data_files.update({'valid': f'{self.data_root}/{self.dataset_name}/{self.dataset_name}_valid.json',})

        datasets = load_dataset('json', data_files=data_files)
        datasets = self._post_process(datasets) 

        # subsample training dataset if needed
        num_train = len(datasets['train'])
        idxs = list()
        for idx in self.train_batch_idxs:
            idxs += range(idx*self.batch_size, (idx+1)*self.batch_size)        
        datasets['train'] = Dataset.from_dict(datasets['train'][[idx for idx in idxs if idx < num_train]])

        return datasets


    def load_llm_preds(self, split):
        labels = list()
        rationales = list()
        for idx in getattr(self, f'{split}_batch_idxs'):
            if self.dataset_name == 'gsm8k':
                with open(f'{self.data_root}/{self.dataset_name}/llm/{split}_CoT_{idx}.json') as f:
                    outputs = f.readlines()
                for output in outputs:
                    data_dict = json.loads(output)
                    rationale, label = self._parse_llm_output(data_dict['answer'])

                    rationales.append(rationale)
                    labels.append(label)
            else:
                with open(f'{self.data_root}/{self.dataset_name}/llm/{split}_CoT_{idx}.json') as f:
                    outputs = json.load(f)

                for output in outputs:
                    rationale, label = self._parse_llm_output(output)

                    rationales.append(rationale)
                    labels.append(label)

        return rationales, labels


    def load_gpt_preds(self, split):
        labels = list()
        rationales = list()
        
        with open(f'{self.data_root}/gpt-neox/{self.dataset_name}/{split}.json') as f:
            outputs = json.load(f)

        for output in outputs:
            rationale, label = self._parse_gpt_output(output)

            rationales.append(rationale)
            labels.append(label)

        return rationales, labels


    def _post_process(self, datasets):
        raise NotImplementedError


    def _parse_llm_output(self, output):
        raise NotImplementedError


    def _parse_gpt_output(self, output):
        raise NotImplementedError

class GSM8KDatasetLoader(DatasetLoader):
    def __init__(self):
        dataset_name = 'gsm8k'
        dataset_version = None
        source_dataset_name = 'gsm8k'
        has_valid = False
        split_map = {
            'train': 'train',
            'test': 'test',
        }
        batch_size = 1000
        train_batch_idxs = range(8)
        test_batch_idxs = range(2)

        super().__init__(dataset_name, source_dataset_name, dataset_version, has_valid, split_map,
                 batch_size, train_batch_idxs, test_batch_idxs, valid_batch_idxs=None)


    def _post_process(self, datasets):
        
        def prepare_input(example):
            question = example['question']

            input = f'{question}\n'

            example['input'] = input
            example['label'] = example['answer'].split()[-1].strip()

            return example

        datasets = datasets.map(prepare_input)
        datasets = datasets.remove_columns(['question', 'answer'])

        return datasets


    def _parse_llm_output(self, output):
        rationale_label = output.split('<solution>')[-1]
        rationale_label = rationale_label.rstrip()
        rationale, label = rationale_label.split('</solution>')
        rationale = rationale.rstrip()

        try:
            label = label.split()
            label = label[-1]
        except:
            label = ' '
        
        return rationale, label
    

def get_tokenize_function(tokenizer, input:str="question", gt:str="answer", pre_prompt:str="", post_prompt:str=""):
    def tokenize(example):
        prompt = [pre_prompt + dialogue + post_prompt for dialogue in example[input]]
        example['input_ids'] = tokenizer(prompt, padding='max_length', truncation=True, 
                                        return_tensors='pt').input_ids
        example['labels'] = tokenizer(example['answer'], padding='max_length', truncation=True, 
                                    return_tensors='pt').input_ids
        
        return example
    
    return tokenize
    

