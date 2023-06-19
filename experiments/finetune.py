## https://huggingface.co/openlm-research/open_llama_13b

from pathlib import Path
import os
import numpy as np
import torch
from transformers import Trainer,  TrainingArguments, LlamaForSequenceClassification, LlamaTokenizer, LlamaForCausalLM
from datasets import load_dataset
import evaluate

model_path = 'openlm-research/open_llama_13b'

dataset = load_dataset("yelp_review_full", cache_dir=os.environ.get('DATADIR'))

tokenizer = LlamaTokenizer.from_pretrained(model_path,
                                           cache_dir=os.environ.get('MODELDIR'))
tokenizer.pad_token = tokenizer.eos_token

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)
tokenized_datasets = dataset.map(tokenize_function, batched=True)

# model = LlamaForCausalLM.from_pretrained(
#     model_path, torch_dtype=torch.float16, device_map='auto',
#     cache_dir=os.environ.get('MODELDIR'),
# )
# prompt = 'Q: What is the largest animal?\nA:'
# generation_output = model.generate(
#     input_ids=tokenizer(prompt, return_tensors="pt").input_ids.to('cuda'),
#     max_new_tokens=100
# )
# print(tokenizer.decode(generation_output[0]))

model = LlamaForSequenceClassification.from_pretrained(
    model_path, cache_dir=os.environ.get('MODELDIR'), num_labels=5)

training_args = TrainingArguments(
    output_dir=Path(os.environ.get('LOGDIR')) / 'open_llama',
    evaluation_strategy="epoch")

small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))

metric = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()
