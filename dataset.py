import datasets

def get_tokenize_function(tokenizer, input:str="question", gt:str="answer", pre_prompt:str="", post_prompt:str=""):
    def tokenize(example):
        prompt = [pre_prompt + dialogue + post_prompt for dialogue in example[input]]
        example['input_ids'] = tokenizer(prompt, padding='max_length', truncation=True, 
                                        return_tensors='pt').input_ids
        example['labels'] = tokenizer(example['answer'], padding='max_length', truncation=True, 
                                    return_tensors='pt').input_ids
        
        return example
    
    return tokenize
    

