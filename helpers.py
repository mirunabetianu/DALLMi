from collections import defaultdict, Counter
import numpy as np
from preprocessing.domain_preprocessing import get_domain_before_tokenization, DATASETS, TASK, get_domain_content_column

max_length = 128



def preprocess_delete_per_label(samples, tokenizer, dataset_type, categories, ratio):
        text = samples[get_domain_content_column(dataset_type)]
        text_try = [str(t) for t in text]

        encoding = tokenizer.batch_encode_plus(
            text_try,
            padding='max_length',
            truncation=True,
            max_length=max_length
        )

        labels_batch = {k: samples[k] for k in categories}
        labels_matrix = np.zeros((len(text), len(categories)))

        for idx, label in enumerate(categories):
            labels_matrix[:, idx] = labels_batch[label]

        label_weights_simple = np.sum(np.array(labels_matrix), axis=0)
        print(label_weights_simple)

        for label_idx, label in enumerate(categories):
            sample_indices = [idx for idx, val in enumerate(labels_matrix[:, label_idx]) if val == 1]
            if sample_indices:
                num_labels_to_remove = round(len(sample_indices) * ratio)

                if num_labels_to_remove == len(sample_indices):
                    num_labels_to_remove = len(sample_indices) - 1

                samples_to_remove = np.random.choice(sample_indices, size=num_labels_to_remove, replace=False)
                labels_matrix[samples_to_remove, label_idx] = 0

        encoding["labels"] = labels_matrix.tolist()
        return encoding

def preprocess_delete_per_sample(samples, tokenizer, dataset_type, categories, ratio, seed):
    np.random.seed(seed)

    text = samples[get_domain_content_column(dataset_type)]
    text_try = [str(t) for t in text]

    encoding = tokenizer.batch_encode_plus(
        text_try,
        padding='max_length',
        truncation=True,
        max_length=max_length
    )
    
    labels_batch = {k: samples[k] for k in categories}
    labels_matrix = np.zeros((len(text), len(categories)))

    for idx, label in enumerate(categories):
        labels_matrix[:, idx] = labels_batch[label]

    label_popularity = Counter()
    for label in categories:
        label_popularity[label] = sum(samples[label])
    
    preserved_labels = defaultdict(bool)

    label_weights_simple = np.sum(np.array(labels_matrix), axis=0)

    for i in range(len(text)):
        label_indices = [idx for idx, val in enumerate(labels_matrix[i]) if val == 1]
        if label_indices:
            num_labels_to_remove = round(len(label_indices) * ratio)
            label_weights = label_weights_simple[label_indices]
            label_weights /= sum(label_weights)
            label_weights = 1.0 - label_weights
            if sum(label_weights) > 0:
                label_weights /= sum(label_weights)
            else:
                label_weights = np.random.uniform(0, 1, len(label_indices))
                label_weights /= sum(label_weights)

            labels_to_remove = np.random.choice(label_indices, size=num_labels_to_remove, p=label_weights, replace=False)
            labels_matrix[i][labels_to_remove] = 0

            label_indices = [idx for idx, val in enumerate(labels_matrix[i]) if val == 1]

            for label_idx in label_indices:
                preserved_labels[label_idx] = True

    for label_idx in range(len(categories)):
        if not preserved_labels[label_idx]:
            for i in range(len(text)):
                if labels_batch[categories[label_idx]][i] == 1:
                    labels_matrix[i][label_idx] = 1
                    preserved_labels[label_idx] = True
                    break        
    
    encoding["labels"] = labels_matrix.tolist()
    return encoding

def preprocess_samples(samples, tokenizer, dataset_type, categories):
    text = samples[get_domain_content_column(dataset_type)]
    text_try = []
    for t in text:
        text_try.append(str(t))
    encoding = tokenizer.batch_encode_plus(
        text_try,
        padding='max_length',
        truncation=True,
        max_length=max_length
    )

    labels_batch = {k: samples[k] for k in categories}
    labels_matrix = np.zeros((len(text), len(categories)))

    for idx, label in enumerate(categories):
        labels_matrix[:, idx] = labels_batch[label]

    encoding["labels"] = labels_matrix.tolist()

    return encoding       