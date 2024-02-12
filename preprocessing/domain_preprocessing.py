from enum import IntEnum

import pandas as pd
from sklearn.model_selection import train_test_split


class DATASETS(IntEnum):
    PUB_MED = 0
    MOVIE_PLOT = 1
    ARVIX = 2


class TASK(IntEnum):
    SOURCE = 0
    TARGET = 1


def get_source_domain_path(dataset_type: DATASETS):
    main_path = "pu_bert/data/"
    match dataset_type:
        case DATASETS.PUB_MED:
            return main_path + "pubmed_male.csv"
        case DATASETS.MOVIE_PLOT:
            return main_path + "wiki.csv"
        case DATASETS.ARVIX:
            return main_path + "arvix_older_papers.csv"
        case default:
            return ""


def get_target_domain_path(dataset_type: DATASETS):
    main_path = "/home/mbetianu/pu_bert/data/"
    match dataset_type:
        case DATASETS.PUB_MED:
            return main_path + "pubmed_female.csv"
        case DATASETS.MOVIE_PLOT:
            return main_path + "imdb.csv"
        case DATASETS.ARVIX:
            return main_path + "arvix_newer_papers.csv"
        case default:
            return ""


def get_domain_first_label_index(dataset_type: DATASETS):
    match dataset_type:
        case DATASETS.PUB_MED:
            return 6
        case DATASETS.MOVIE_PLOT:
            return 13
        case DATASETS.ARVIX:
            return 3
        case default:
            return -1


def get_domain_content_column(dataset_type: DATASETS):
    match dataset_type:
        case DATASETS.PUB_MED:
            return 'abstractText'
        case DATASETS.MOVIE_PLOT:
            return 'overview'
        case DATASETS.ARVIX:
            return 'summaries'
        case default:
            return -1



def get_domain_before_tokenization(dataset_type: DATASETS, task: TASK):
    if task == TASK.SOURCE:
        if dataset_type == DATASETS.ARVIX:
            dataset = pd.read_csv(get_target_domain_path(dataset_type), header=0, index_col=0, nrows=6000)
        elif dataset_type == DATASETS.PUB_MED:
            dataset = pd.read_csv(get_source_domain_path(dataset_type), header=0, index_col=0, nrows=3000)
        else:
            dataset = pd.read_csv(get_source_domain_path(
                dataset_type), header=0, index_col=0, nrows=2500)
    else:
        if dataset_type == DATASETS.ARVIX:
            dataset = pd.read_csv(get_target_domain_path(dataset_type), header=0, index_col=0, nrows=6000)
        else:
            dataset = pd.read_csv(get_target_domain_path(dataset_type), header=0, index_col=0)
    categories = dataset.columns[get_domain_first_label_index(dataset_type):]
    num_labels = len(categories)

    dataset["one_hot_labels"] = list(dataset[categories].values)

    if task == TASK.SOURCE:
        dataset_train, dataset_test = train_test_split(dataset, random_state=42, test_size=0.20, shuffle=True)

        labels_train = dataset_train['one_hot_labels'].to_list()
        content_train = list(dataset_train[get_domain_content_column(dataset_type)].values)

        labels_test = dataset_test['one_hot_labels'].to_list()
        content_test = list(dataset_test[get_domain_content_column(dataset_type)].values)

        return num_labels, labels_train, content_train, labels_test, content_test, categories, dataset_train, dataset_test
    else:
        return num_labels, categories, list(dataset[get_domain_content_column(dataset_type)].values), dataset
