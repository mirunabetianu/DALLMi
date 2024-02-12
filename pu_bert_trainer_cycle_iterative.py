from collections import Counter, defaultdict
from typing import Dict, cast, Optional
import numpy as np
import pandas as pd
import torch
from datasets import Dataset, DatasetDict
from sklearn.metrics import average_precision_score
from pytorch_multilabel_balanced_sampler import ClassCycleSampler
from sklearn.model_selection import train_test_split, ParameterGrid
from torch.utils.data import DataLoader
from transformers import BertTokenizer, TrainingArguments, Trainer, EvalPrediction
from MyBert import MyBert
from preprocessing.domain_preprocessing import get_domain_before_tokenization, DATASETS, TASK, get_domain_content_column
import json
from skmultilearn.model_selection import iterative_train_test_split
import argparse
from helpers import preprocess_delete_per_sample, preprocess_delete_per_label, preprocess_samples

parser = argparse.ArgumentParser(description='Fine-tuning target domain with BCE loss')
parser.add_argument('--removal_strategy', type=str, default='sample', help='Removal per sample or label')
parser.add_argument('--seed', type=int, default=42, help='Seed value')

args = parser.parse_args()

removal_strategy = args.removal_strategy
seed = args.seed

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
    'ratio_labels': [0.5, 0.7, 0.9],
    'dataset': [DATASETS.PUB_MED, DATASETS.ARVIX, DATASETS.MOVIE_PLOT],
    'mixup': [True]
}

for params in ParameterGrid(param_grid):
    dataset_type = params['dataset']
    num_labels, categories, contents, dataset = get_domain_before_tokenization(dataset_type, TASK.TARGET)
    
    df = dataset
    column_names = dataset.columns
    X = df.drop(categories, axis=1, inplace=False)
    y = df.drop([col for col in column_names if col not in categories], axis=1, inplace=False)
    xtrain, ytrain, xtest, ytest = iterative_train_test_split(X.to_numpy(), y.to_numpy(), 0.2)
    xtraindf = pd.DataFrame(xtrain, columns=[col for col in column_names if col not in categories])
    ytraindf = pd.DataFrame(ytrain, columns=[col for col in column_names if col in categories])
    traindf = pd.concat([xtraindf, ytraindf], axis=1)
    xtestdf = pd.DataFrame(xtest, columns=[col for col in column_names if col not in categories])
    ytestdf = pd.DataFrame(ytest, columns=[col for col in column_names if col in categories])
    testdf = pd.concat([xtestdf, ytestdf], axis=1)
    dataset = DatasetDict({"train": Dataset.from_pandas(traindf), "test": Dataset.from_pandas(testdf)})


    dataset_train = dataset["train"]
    dataset_test = dataset["test"]


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


    train_dataset = dataset_train.map(preprocess_samples_delete, batched=True, remove_columns=dataset_train.column_names)
    test_dataset = dataset_test.map(preprocess_samples_reg, batched=True, remove_columns=dataset_test.column_names)


    best_mAP = 0.0
    best_params = {}

    train_dataset.set_format("torch")
    test_dataset.set_format("torch")


    def compute_metrics(p: EvalPrediction) -> Dict[str, float]:
        true_labels = p.label_ids
        predicted_labels = np.round(torch.sigmoid(torch.tensor(p.predictions)))

        return {"mAP": average_precision_score(true_labels, predicted_labels, average="macro")}


    training_args = TrainingArguments(
        output_dir='pu_bert/checkpoints',
        evaluation_strategy="epoch",
        per_device_train_batch_size=params['batch_size'],
        per_device_eval_batch_size=params['batch_size'],
        num_train_epochs=params['num_epochs'],
        learning_rate=params['learning_rate'],
        save_strategy="no",
        logging_dir='./logs',
        seed=42,
        # load_best_model_at_end=True,
        # metric_for_best_model="eval_mAP",
        # greater_is_better=True
    )

    class PUTrainer(Trainer):
        def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
            labels_tensor = torch.Tensor(cast(list, cast(Dataset, self.train_dataset)["labels"]))
            return ClassCycleSampler(labels_tensor.int())

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

                var_loss_u = torch.mean(torch.sigmoid(logits_u))
                if torch.isnan(var_loss_u).any():
                    var_loss_u = 0
                var_loss_p = torch.mean(torch.norm(torch.sigmoid(logits_p), p=2))
               
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
                    x_tilda_pred = model.forward_mix_embed(chosen_unlabelled_input_ids.view(params['interpolation_samples'], -1),
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


    trainer = PUTrainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics
    )

    trainer.train()

    results = trainer.evaluate()

    print(params)
    print(results)

    params_pu.append(params)
    results_pu.append(results)


json.dump(results_pu, open( "results-iterative-pu_removal_per_{removal_strategy}_seed={seed}.json".format(removal_strategy=removal_strategy, seed=seed), 'w' ) )
json.dump(params_pu, open( "params-iterative-pu_removal_per_{removal_strategy}_seed={seed}.json".format(removal_strategy=removal_strategy, seed=seed), 'w' ) )


