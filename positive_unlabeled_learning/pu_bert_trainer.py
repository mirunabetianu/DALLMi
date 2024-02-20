from collections import Counter
from typing import Dict, cast, Optional, Any
import numpy as np
import torch
from datasets import Dataset
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                             roc_auc_score)
from sklearn.model_selection import train_test_split, ParameterGrid
from torch.utils.data import DataLoader
from transformers import BertTokenizer, TrainingArguments, Trainer, EvalPrediction
from helpers.MyBert import MyBert
from preprocessing.domain_preprocessing import get_domain_before_tokenization, DATASETS, TASK, get_domain_content_column
import argparse
from helpers import preprocess_delete_per_sample, preprocess_delete_per_label, preprocess_samples
import json
import time

parser = argparse.ArgumentParser(description='Fine-tuning target domain with BCE loss')
parser.add_argument('--removal_strategy', type=str, default='label', help='Removal per sample or label')
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
    'batch_size_small': [4],
    'num_epochs': [12],
    'interpolation_samples': [1],
    'ratio_labels': [0.5],
    'dataset': [DATASETS.PUB_MED, DATASETS.ARVIX, DATASETS.MOVIE_PLOT],
}

for params in ParameterGrid(param_grid):
    dataset_type = params['dataset']
    num_labels, categories, contents, dataset = get_domain_before_tokenization(dataset_type, TASK.TARGET)
    dataset_train, dataset_test = train_test_split(dataset, random_state=42, test_size=0.20, shuffle=True)

    dataset_train = Dataset.from_pandas(dataset_train)
    dataset_test = Dataset.from_pandas(dataset_test)


    tokenizer = BertTokenizer.from_pretrained("pu_bert/finetuned_models_trainer/BERT_Multi_label_{dataset}_tokenizer".format(dataset=dataset_type.name))
    model = MyBert.from_pretrained("pu_bert/finetuned_models_trainer/BERT_Multi_label_{dataset}".format(dataset=dataset_type.name),
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
        save_strategy="epoch",
        logging_dir='./logs',
        seed=seed,
    )

    train_data_per_label = [train_dataset.filter(lambda x: x["labels"][ind])
                            for ind in range(num_labels)]


    train_data_per_label = [data.shuffle(seed=42).select(range(int(len(data))))
                            for data in train_data_per_label]

    min_samples = min(len(data) for data in train_data_per_label)

    batch_size_small = min(params['batch_size_small'], min_samples)

    train_dataloader_per_label = [DataLoader(data, shuffle=True, drop_last=True, batch_size=batch_size_small)
                                    for data in train_data_per_label]

    train_dataloader_per_label_it = [iter(x) for x in train_dataloader_per_label]


    class PUTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False):
            inputs['labels'] = None
            outputs = model(**inputs)
            unlabelled_input_ids = inputs['input_ids']
            unlabelled_attention_mask = inputs['attention_mask']

            loss = torch.tensor(0.0, device=device)

            for label_id in range(num_labels):
                batch_per_label = None
                try:
                    batch_per_label = next(train_dataloader_per_label_it[label_id])
                except StopIteration:
                    train_dataloader_per_label_it[label_id] = iter(train_dataloader_per_label[label_id])
                    batch_per_label = next(train_dataloader_per_label_it[label_id])

                if batch_per_label is not None:
                    batch_per_label = {k: v.to(device) for k, v in batch_per_label.items()}
                    batch_per_label['labels'] = None
                    positive_input_ids = batch_per_label['input_ids']
                    positive_attention_mask = batch_per_label['attention_mask']
                    outputs_per_label = model(**batch_per_label)

                    outputs_logits = torch.sigmoid(outputs.logits[:, label_id])
                    outputs_per_label_logits = torch.sigmoid(outputs_per_label.logits[:, label_id])

                    var_loss_u = torch.mean(outputs_logits)
                    var_loss_p = torch.mean(torch.norm(outputs_per_label_logits, p=2))

                    var_loss_label = var_loss_u - var_loss_p

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

    start_time = time.time()

    trainer = PUTrainer(
        model=model,
        args=training_args,
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

    print(params_pu)
    print(results_pu)

    params_pu.append(params)
    results_pu.append(results)


json.dump(results_pu, open( "time-results-pu-batch-small_removal_per_{removal_strategy}_seed={seed}.json".format(removal_strategy=removal_strategy, seed=seed), 'w' ) )




