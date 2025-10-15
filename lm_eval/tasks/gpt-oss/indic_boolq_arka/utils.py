import datasets


def filter_dataset(dataset: datasets.Dataset, language: str) -> datasets.Dataset:
    return dataset.filter(lambda example: example["language"] == language)


def filter_bn(dataset: datasets.Dataset) -> datasets.Dataset:
    return filter_dataset(dataset, "bn")

def filter_gu(dataset: datasets.Dataset) -> datasets.Dataset:
    return filter_dataset(dataset, "gu")

def filter_hi(dataset: datasets.Dataset) -> datasets.Dataset:
    return filter_dataset(dataset, "hi")

def filter_kn(dataset: datasets.Dataset) -> datasets.Dataset:
    return filter_dataset(dataset, "kn")

def filter_mr(dataset: datasets.Dataset) -> datasets.Dataset:
    return filter_dataset(dataset, "mr")

def filter_ml(dataset: datasets.Dataset) -> datasets.Dataset:
    return filter_dataset(dataset, "ml")

def filter_or(dataset: datasets.Dataset) -> datasets.Dataset:
    return filter_dataset(dataset, "or")

def filter_pa(dataset: datasets.Dataset) -> datasets.Dataset:
    return filter_dataset(dataset, "pa")

def filter_ta(dataset: datasets.Dataset) -> datasets.Dataset:
    return filter_dataset(dataset, "ta")

def filter_te(dataset: datasets.Dataset) -> datasets.Dataset:
    return filter_dataset(dataset, "te")