import datasets


def filter_dataset(dataset: datasets.Dataset, question_type: str) -> datasets.Dataset:
    return dataset.filter(lambda example: example["question_type"] == question_type)


def filter_association(dataset: datasets.Dataset) -> datasets.Dataset:
    return filter_dataset(dataset, "Association")

def filter_country(dataset: datasets.Dataset) -> datasets.Dataset:
    return filter_dataset(dataset, "Country Prediction")

def filter_gk(dataset: datasets.Dataset) -> datasets.Dataset:
    return filter_dataset(dataset, "General Awareness")

def filter_states(dataset: datasets.Dataset) -> datasets.Dataset:
    return filter_dataset(dataset, "State Prediction")

