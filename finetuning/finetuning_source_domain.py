from typing import Dict, Any

import numpy as np
import torch
from datasets import Dataset
from sklearn.metrics import average_precision_score
from sklearn.model_selection import ParameterGrid
from transformers import BertForSequenceClassification, BertTokenizer, Trainer, TrainingArguments, \
    EvalPrediction

from preprocessing.domain_preprocessing import get_domain_before_tokenization, TASK, DATASETS, get_domain_content_column

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
max_length = 128
RATIO = 1
RATIO_LABELS = 1

dataset_type = DATASETS.MOVIE_PLOT

num_labels, labels_train, content_train, labels_test, content_test, categories, dataset_train, dataset_test = get_domain_before_tokenization(
        dataset_type=dataset_type, task=TASK.SOURCE)


tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased",
                                                      num_labels=num_labels,
                                                      problem_type="multi_label_classification")

dataset_train = Dataset.from_pandas(dataset_train)
dataset_test = Dataset.from_pandas(dataset_test)

def process_data(samples: Dict[str, Any]) -> Dict[str, Any]:
    text = samples[get_domain_content_column(dataset_type)]
    encoding = tokenizer.batch_encode_plus(
        text,
        padding=True,
        truncation=True,
        max_length=max_length
    )

    labels_batch = {k: samples[k] for k in categories}
    labels_matrix = np.zeros((len(text), num_labels))

    for idx, label in enumerate(categories):
        labels_matrix[:, idx] = labels_batch[label]

    encoding["labels"] = labels_matrix.tolist()

    return encoding

train_dataset = dataset_train.map(process_data, batched=True, remove_columns=dataset_train.column_names)
test_dataset = dataset_test.map(process_data, batched=True, remove_columns=dataset_test.column_names)

train_dataset.set_format("torch")
test_dataset.set_format("torch")

seed = 42
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
np.random.seed(seed)

print("Initializing model")
param_grid = {
    'learning_rate': [5e-5],
    'batch_size': [64],
    'num_epochs': [12]
}

best_mAP = 0.0
best_params = {}

def compute_metrics(p: EvalPrediction) -> Dict[str, float]:
    true_labels = p.label_ids
    predicted_labels = np.round(torch.sigmoid(torch.tensor(p.predictions)))

    return {"mAP": average_precision_score(true_labels, predicted_labels, average="macro")}


for params in ParameterGrid(param_grid):
    training_args = TrainingArguments(
        output_dir='./results_trainer',
        evaluation_strategy="epoch",
        per_device_train_batch_size=params['batch_size'],
        per_device_eval_batch_size=params['batch_size'],
        num_train_epochs=params['num_epochs'],
        learning_rate=params['learning_rate'],
        save_strategy="epoch",
        logging_dir='./logs',
        seed=42,
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics
    )

    trainer.train()

    results = trainer.evaluate()

    trainer.save_model(
        "pu_bert/finetuned_models_trainer/BERT_Multi_label_{dataset}".format(dataset=(dataset_type).name))
    tokenizer.save_pretrained(
        "pu_bert/finetuned_models_trainer/BERT_Multi_label_{dataset}_tokenizer".format(dataset=(dataset_type).name))

    print(results)
