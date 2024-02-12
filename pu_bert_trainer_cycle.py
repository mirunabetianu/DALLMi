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
import time


import argparse
from helpers import preprocess_delete_per_sample, preprocess_delete_per_label, preprocess_samples

parser = argparse.ArgumentParser(description='Fine-tuning target domain with BCE loss')
parser.add_argument('--removal_strategy', type=str, default='sample', help='Removal per sample or label')
parser.add_argument('--seed', type=int, default=42, help='Seed value')
parser.add_argument('--mixup_strategy', type=str, default="embedding")

args = parser.parse_args()

removal_strategy = args.removal_strategy
seed = args.seed
mixup_strategy = args.mixup_strategy

torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
np.random.seed(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
max_length = 128
params_pu = []
results_pu = []

param_grid = {
    'learning_rate': [5e-5],
    'batch_size': [64],
    'num_epochs': [12],
    'interpolation_samples': [1],
    'ratio_labels': [0.5],
    'dataset': [DATASETS.PUB_MED, DATASETS.ARVIX, DATASETS.MOVIE_PLOT],
    #'dataset': [DATASETS.ARVIX],
    'mixup': [True],
    #'mixup':[True]
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
    mixup_fn = model.forward_mix_embed
    if mixup_strategy == "embedding":
        mixup_fn = model.forward_mix_embed
    elif mixup_strategy == "encoding":
        mixup_fn = model.forward_mix_encoder
    elif mixup_strategy == "sentence":
        mixup_fn = model.forward_mix_sent    

    def preprocess_samples_delete(samples):
        if removal_strategy == "sample":
            return preprocess_delete_per_sample(samples, tokenizer, dataset_type, categories, params['ratio_labels'], seed)
        elif removal_strategy == "label":
            return preprocess_delete_per_label(samples, tokenizer, dataset_type, categories, params['ratio_labels'])    

        return []    

    def preprocess_samples_reg(samples):
        return preprocess_samples(samples, tokenizer, dataset_type, categories)

    train_dataset = dataset_train.map(preprocess_samples_delete, batched=True, remove_columns=dataset_train.column_names)
    test_dataset = dataset_test.map(preprocess_samples_reg, batched=True, remove_columns=dataset_test.column_names)


    best_mAP = 0.0
    best_params = {}

    train_dataset.set_format("torch")
    test_dataset.set_format("torch")


    def compute_metrics(p: EvalPrediction) -> Dict[str, Any]:
        y_true = p.label_ids
        logits = p.predictions

        y_score = np.round(torch.sigmoid(torch.tensor(logits)))

        return {
            "MAP": average_precision_score(y_true, y_score, average="macro"),
            "AVG PREC MICRO": average_precision_score(y_true, y_score, average="micro"),
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
        # load_best_model_at_end=True,
        # metric_for_best_model="eval_mAP",
        # greater_is_better=True
    )


    class PUTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False):
            inputs_labels = inputs.pop("labels")
            outputs = model(**inputs)
            input_ids = inputs['input_ids']
            attention_mask = inputs['attention_mask']
            outputs_logits = outputs.logits
            loss = torch.tensor(0.0, device=model.device)

            for label_id in range(num_labels):
                indices_p: list[int] = []
                indices_u: list[int] = []

                for i, labels in enumerate(inputs_labels):
                    if labels[label_id]:
                        indices_p.append(i)
                    else:
                        indices_u.append(i)

                logits_p = outputs_logits[indices_p, label_id]
                logits_u = outputs_logits[indices_u, label_id]

                var_loss_u = torch.norm(torch.mean(torch.sigmoid(logits_u)))
                if torch.isnan(var_loss_u).any():
                    var_loss_u = 0
                var_loss_p = torch.mean(torch.norm(torch.sigmoid(logits_p)))
               
                var_loss_label = var_loss_u - var_loss_p

                positive_input_ids = input_ids[indices_p]
                unlabelled_input_ids = input_ids[indices_u]

                positive_attention_mask = attention_mask[indices_p]
                unlabelled_attention_mask = attention_mask[indices_u]

                if len(unlabelled_input_ids) > 0 and len(positive_input_ids) > 0 and params['mixup']:
                    index_batch = [np.random.randint(len(unlabelled_input_ids)) for _ in range(params['interpolation_samples'])]
                    index_batch_per_label = [np.random.randint(len(positive_input_ids)) for _ in range(params['interpolation_samples'])]

                    chosen_unlabelled_input_ids, chosen_unlabelled_attention_mask = (
                        unlabelled_input_ids[index_batch], unlabelled_attention_mask[index_batch])

                    chosen_positive_input_ids, chosen_positive_attention_mask = (
                        positive_input_ids[index_batch_per_label], positive_attention_mask[index_batch_per_label])

                    mu = np.random.beta(0.3, 0.3)
                    x_tilda_pred = mixup_fn(chosen_unlabelled_input_ids.view(params['interpolation_samples'], -1),
                                                            chosen_unlabelled_attention_mask.view(params['interpolation_samples'], -1),
                                                            chosen_positive_input_ids.view(params['interpolation_samples'], -1),
                                                            chosen_positive_attention_mask.view(params['interpolation_samples'], -1), mu)

                    output_chosen_unlabelled = model(input_ids=chosen_unlabelled_input_ids.view(params['interpolation_samples'], -1),
                                                        attention_mask=chosen_unlabelled_attention_mask.view(params['interpolation_samples'], -1))
                    phi_tilda = mu * 1.0 + (1.0 - mu) * torch.sigmoid(output_chosen_unlabelled.logits[:, label_id])

                    mixup_term = torch.pow((torch.norm(phi_tilda) - torch.norm(torch.sigmoid(x_tilda_pred[:, label_id]))),
                                            2).item()

                    var_loss_label += mixup_term
                loss += var_loss_label

            return (loss, outputs) if return_outputs else loss

    start_time = time.time()

    trainer = PUTrainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics
    )

    trainer.train()

    end_time = time.time()

    # Calculate training time
    training_time = end_time - start_time
    print(f"Total training time: {training_time} seconds")

    results = trainer.evaluate()

    print(params)
    print(results)

    params_pu.append(params)
    results_pu.append(results)


json.dump(results_pu, open( "log-results-pu_removal_per_{removal_strategy}_mixup_{mixup}_seed={seed}.json".format(removal_strategy=removal_strategy, seed=seed,mixup=mixup_strategy), 'w' ) )
json.dump(params_pu, open( "log-params-pu_removal_per_{removal_strategy}_mixup_{mixup}_seed={seed}.json".format(removal_strategy=removal_strategy, seed=seed, mixup=mixup_strategy), 'w' ) )





