from collections import Counter, defaultdict
from typing import Dict, cast, Optional, Any
import numpy as np
import torch
from datasets import Dataset
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                             roc_auc_score)
from pytorch_multilabel_balanced_sampler import ClassCycleSampler
from sklearn.model_selection import train_test_split, ParameterGrid
from torch.utils.data import DataLoader
from transformers import BertTokenizer, TrainingArguments, Trainer, EvalPrediction
from MyBert import MyBert
from preprocessing.domain_preprocessing import get_domain_before_tokenization, DATASETS, TASK, get_domain_content_column
import json
import argparse
from helpers import preprocess_delete_per_sample, preprocess_delete_per_label, preprocess_samples

parser = argparse.ArgumentParser(description='Fine-tuning target domain with BCE loss')
parser.add_argument('--removal_strategy', type=str, default='sample', help='Removal per sample or label')
parser.add_argument('--seed', type=int, default=12345, help='Seed value')


args = parser.parse_args()

removal_strategy = args.removal_strategy
seed = args.seed

print(removal_strategy)
print(seed)

torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
np.random.seed(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
max_length = 128

params_reg = []
results_reg = []


param_grid = {
    'learning_rate': [5e-5],
    'batch_size': [64],
    'num_epochs': [12],
    'interpolation_samples': [1],
    #'ratio_labels': [0.5, 0.7, 0.9],
    'dataset': [DATASETS.PUB_MED, DATASETS.ARVIX, DATASETS.MOVIE_PLOT],
}

for params in ParameterGrid(param_grid):
    dataset_type = params['dataset']
    num_labels, categories, contents, dataset = get_domain_before_tokenization(dataset_type, TASK.TARGET)
    dataset_train, dataset_test = train_test_split(dataset, random_state=42, test_size=0.20, shuffle=True)

    dataset_train = Dataset.from_pandas(dataset_train)
    dataset_test = Dataset.from_pandas(dataset_test)


    tokenizer = BertTokenizer.from_pretrained(
        "pu_bert/finetuned_models_trainer/BERT_Multi_label_{dataset}_tokenizer".format(dataset=dataset_type.name)
        #"bert-base-uncased"
        )
    model = MyBert.from_pretrained(
        "pu_bert/finetuned_models_trainer/BERT_Multi_label_{dataset}".format(dataset=dataset_type.name),
                                #"bert-base-uncased",
                                num_labels=num_labels,
                                problem_type="multi_label_classification",
                                )
    
    def preprocess_samples_delete(samples):
        if removal_strategy == "sample":
            return preprocess_delete_per_sample(samples, tokenizer, dataset_type, categories, params['ratio_labels'], seed)
        elif removal_strategy == "label":
            return preprocess_delete_per_label(samples, tokenizer, dataset_type, categories, params['ratio_labels'])    

        return []    

    def preprocess_samples_reg(samples):
        return preprocess_samples(samples, tokenizer, dataset_type, categories)

    #train_dataset = dataset_train.map(preprocess_samples_delete, batched=True, remove_columns=dataset_train.column_names)
    test_dataset = dataset_test.map(preprocess_samples_reg, batched=True, remove_columns=dataset_test.column_names)


    best_mAP = 0.0
    best_params = {}

    #train_dataset.set_format("torch")
    test_dataset.set_format("torch")


    def compute_metrics(p: EvalPrediction) -> Dict[str, Any]:
        y_true = p.label_ids
        logits = p.predictions

        y_score = np.round(torch.sigmoid(torch.tensor(logits)))

        return {
            "mAP": average_precision_score(y_true, y_score, average="macro"),
            "micro mAP": average_precision_score(y_true, y_score, average="micro"),
            "F1": f1_score(y_true, y_score, average="macro"),
            #"F1-class": f1_score(y_true, y_score, average=None),
            "ROC AUC": roc_auc_score(y_true, y_score, average="macro"),
            "ACCURACY": accuracy_score(y_true, y_score)
        }


    training_args = TrainingArguments(
        output_dir='pu_bert/checkpoints',
        evaluation_strategy="epoch",
        per_device_train_batch_size=params['batch_size'],
        per_device_eval_batch_size=params['batch_size'],
        num_train_epochs=params['num_epochs'],
        learning_rate=params['learning_rate'],
        save_strategy="no",
        logging_dir='./logs',
        seed=seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics
    )

    results = trainer.evaluate()

    print(params)
    print(results)

