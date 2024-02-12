from collections import Counter, defaultdict
from copy import deepcopy
from typing import Dict, cast, Optional
import numpy as np
import pandas as pd
import torch
from datasets import Dataset, DatasetDict, load_dataset
from sklearn.metrics import average_precision_score
from pytorch_multilabel_balanced_sampler import ClassCycleSampler
from sklearn.model_selection import train_test_split, ParameterGrid
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertTokenizer, TrainingArguments, Trainer, EvalPrediction
from MyBert import MyBert
from preprocessing.domain_preprocessing import get_domain_before_tokenization, DATASETS, TASK, get_domain_content_column
import json
from skmultilearn.model_selection import iterative_train_test_split
import argparse
from helpers import preprocess_delete_per_sample, preprocess_delete_per_label, preprocess_samples

parser = argparse.ArgumentParser(
    description='Fine-tuning target domain with BCE loss')
parser.add_argument('--removal_strategy', type=str,
                    default='sample', help='Removal per sample or label')
parser.add_argument('--seed', type=int, default=12345, help='Seed value')

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


def adapt(src: torch.nn.Module,
          src_dl: torch.utils.data.DataLoader,
          tgt_train_dl: torch.utils.data.DataLoader) -> torch.nn.Module:
    class Discriminator(torch.nn.Module):
        """Discriminator model for source domain."""

        hidden_size = 768
        intermediate_size = 3072

        def __init__(self):
            """Init discriminator."""
            super(Discriminator, self).__init__()
            self.layer = torch.nn.Sequential(
                torch.nn.Linear(self.hidden_size, self.intermediate_size),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(self.intermediate_size,
                                self.intermediate_size),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(self.intermediate_size, 1),
                torch.nn.Sigmoid()
            )

        def forward(self, x):
            """Forward the discriminator."""
            out = self.layer(x)
            return out

    d_learning_rate = 1e-5
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    args = {
        "num_epochs": 12,
        "clip_value": 0.01,
        "temperature": 20,
        "alpha": 1.0,
        "beta": 1.0,
        "max_grad_norm": 1.0,
        "log_step": 1
    }
    discriminator = Discriminator()

    tgt_enc, *classifier_modules = src.children()
    tgt_enc = deepcopy(tgt_enc)
    classifier = torch.nn.Sequential(
        *(deepcopy(module) for module in classifier_modules))

    src.eval()
    tgt_enc.train()
    classifier.eval()
    discriminator.train()

    src.to(dev)
    tgt_enc.to(dev)
    classifier.to(dev)
    discriminator.to(dev)

    BCELoss = torch.nn.BCELoss()
    KLDivLoss = torch.nn.KLDivLoss(reduction="batchmean")
    optimizer_G = torch.optim.Adam(tgt_enc.parameters(), lr=d_learning_rate)
    optimizer_D = torch.optim.Adam(
        discriminator.parameters(), lr=d_learning_rate)
    num_training_steps = args["num_epochs"] * \
        min(len(src_dl), len(tgt_train_dl))
    progress_bar_train = tqdm(range(num_training_steps))

    for _epoch in range(args["num_epochs"]):
        for src_batch, tgt_batch in zip(src_dl, tgt_train_dl):
            src_batch = {k: v.to(dev) for k, v in src_batch.items()}
            tgt_batch = {k: v.to(dev) for k, v in tgt_batch.items()}

            optimizer_D.zero_grad()

            feat_src_tgt = tgt_enc(**src_batch).pooler_output
            feat_tgt = tgt_enc(**tgt_batch).pooler_output
            feat_concat = torch.cat((feat_src_tgt, feat_tgt), 0)

            pred_concat = discriminator(feat_concat.detach())

            label_src = torch.ones(feat_src_tgt.size(0),
                                   device=dev).unsqueeze(1)
            label_tgt = torch.ones(feat_tgt.size(0), device=dev).unsqueeze(1)
            label_concat = torch.cat((label_src, label_tgt), 0)

            dis_loss = BCELoss(pred_concat, label_concat)
            dis_loss.backward()

            for p in discriminator.parameters():
                p.data.clamp_(-args["clip_value"], args["clip_value"])
            optimizer_D.step()

            optimizer_G.zero_grad()

            pred_tgt = discriminator(feat_tgt)

            with torch.no_grad():
                src_out = src(**src_batch).logits
                src_prob = torch.nn.functional.softmax(
                    src_out / args["temperature"], dim=-1)
            tgt_prob = torch.nn.functional.log_softmax(
                classifier(feat_src_tgt) / args["temperature"], dim=-1)
            kd_loss = KLDivLoss(tgt_prob, src_prob.detach()
                                ) * args["temperature"] ** 2

            gen_loss = BCELoss(pred_tgt, label_src)
            loss_tgt = args["alpha"] * gen_loss + args["beta"] * kd_loss
            loss_tgt.backward()
            torch.nn.utils.clip_grad_norm_(
                tgt_enc.parameters(), args["max_grad_norm"])
            optimizer_G.step()

            progress_bar_train.update(1)

    tgt = deepcopy(src)
    (enc_name, _enc_module), *_name_module_tuples = tgt.named_children()
    setattr(tgt, enc_name, tgt_enc)

    return tgt



param_grid = {
    'learning_rate': [5e-5],
    'batch_size': [64],
    'num_epochs': [12],
    'interpolation_samples': [1],
    'ratio_labels': [0.5],#, 0.7, 0.9],
    'dataset': [DATASETS.PUB_MED, DATASETS.ARVIX, DATASETS.MOVIE_PLOT],
    #'mixup': [False, True]
}

for params in ParameterGrid(param_grid):
    dataset_type = params['dataset']
    num_labels, categories, contents, dataset = get_domain_before_tokenization(
        dataset_type, TASK.TARGET)

    dataset_train, dataset_test = train_test_split(
        dataset, random_state=42, test_size=0.20, shuffle=True)

    dataset_train = Dataset.from_pandas(dataset_train)
    dataset_test = Dataset.from_pandas(dataset_test)

    tokenizer = BertTokenizer.from_pretrained(
        "pu_bert/finetuned_models_trainer/BERT_Multi_label_{dataset}_tokenizer".format(
            dataset=dataset_type.name)
        # "bert-base-uncased"
    )
    model = MyBert.from_pretrained(
        "pu_bert/finetuned_models_trainer/BERT_Multi_label_{dataset}".format(
            dataset=dataset_type.name),
        # "bert-base-uncased",
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

    train_dataset = dataset_train.map(
        preprocess_samples_reg, batched=True, remove_columns=dataset_train.column_names)
    test_dataset = dataset_test.map(
        preprocess_samples_reg, batched=True, remove_columns=dataset_test.column_names)

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
    _, _, _, _, _, _, dataset_train, _ = get_domain_before_tokenization(
        dataset_type=dataset_type, task=TASK.SOURCE)
    src_ds = Dataset.from_pandas(dataset_train)
    #src_ds = src_ds["train"]
    column_names = src_ds.column_names
    src_ds = src_ds.map(preprocess_samples_reg, batched=True,
                        remove_columns=column_names)
    src_ds.set_format("torch")
    src_ds = src_ds.remove_columns("labels")
    src_dl = torch.utils.data.DataLoader(
        src_ds, shuffle=True, drop_last=True, batch_size=params['batch_size'])
    train_dataset.set_format("torch")
    train_dataset = train_dataset.remove_columns("labels")
    tgt_train_dl = torch.utils.data.DataLoader(
        train_dataset, shuffle=True, drop_last=True, batch_size=params['batch_size'])
    trainer = Trainer(
        model=adapt(model, src_dl, tgt_train_dl),
        eval_dataset=cast(torch.utils.data.Dataset, test_dataset),
        compute_metrics=compute_metrics,
    )

    results = trainer.evaluate()

    print(params)
    print(results)


