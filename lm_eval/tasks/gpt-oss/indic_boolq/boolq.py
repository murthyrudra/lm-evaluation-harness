from lm_eval.api.task import ConfigurableTask


class IndicBoolq(ConfigurableTask):
    VERSION = 1
    DATASET_PATH = "sarvamai/boolq-indic"

    def __init__(self, config=None):
        super().__init__(
            config={
                "metadata": {"version": self.VERSION},
                "task": "indic_boolq",
                "dataset_path": self.DATASET_PATH,
                "fewshot_config": {"sampler": "first_n"},
                "output_type": "multiple_choice",
                "metric_list": [
                    {
                        "metric": "acc",
                        "aggregation": "mean",
                        "higher_is_better": True,
                    }
                ],
            }
        )

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return True

    def training_docs(self):
        return None

    def should_decontaminate(self):
        return False

    def doc_to_target(self, doc):
        return doc["answer"]

    def doc_to_choice(self, doc):
        options = ["no", "yes"]
        return options

    def doc_to_text(self, doc):
        prompt = f"{doc['passage']}\nQuestion: {doc['question']}?\nAnswer:"

        return prompt


class Bengali_Boolq(IndicBoolq):
    VERSION = 1

    def __init__(self, config=None):
        super().__init__(
            config={
                "metadata": {"version": self.VERSION},
                "task": "bengali_boolq",
                "tag": "indic_boolq",
                "dataset_path": self.DATASET_PATH,
                "output_type": "multiple_choice",
                "metric_list": [
                    {
                        "metric": "acc",
                        "aggregation": "mean",
                        "higher_is_better": True,
                    }
                ],
            }
        )

    def validation_docs(self):
        dataset = self.dataset["train"].filter(
            lambda example: example["language"].startswith("bn"),
            num_proc=8,
            desc="Dropping validation instances whose language is not Bengali",
        )

        return dataset

    def test_docs(self):
        dataset = self.dataset["validation"].filter(
            lambda example: example["language"].startswith("bn"),
            num_proc=8,
            desc="Dropping test instances whose language is not Bengali",
        )

        return dataset


class Odia_Boolq(IndicBoolq):
    VERSION = 1

    def __init__(self, config=None):
        super().__init__(
            config={
                "metadata": {"version": self.VERSION},
                "task": "odia_boolq",
                "tag": "indic_boolq",
                "dataset_path": self.DATASET_PATH,
                "output_type": "multiple_choice",
                "metric_list": [
                    {
                        "metric": "acc",
                        "aggregation": "mean",
                        "higher_is_better": True,
                    }
                ],
            }
        )

    def validation_docs(self):
        dataset = self.dataset["train"].filter(
            lambda example: example["language"].startswith("or"),
            num_proc=8,
            desc="Dropping validation instances whose language is not Odiya",
        )

        return dataset

    def test_docs(self):
        dataset = self.dataset["validation"].filter(
            lambda example: example["language"].startswith("or"),
            num_proc=8,
            desc="Dropping test instances whose language is not Odiya",
        )

        return dataset


class Kannada_Boolq(IndicBoolq):
    VERSION = 1

    def __init__(self, config=None):
        super().__init__(
            config={
                "metadata": {"version": self.VERSION},
                "task": "kannada_boolq",
                "tag": "indic_boolq",
                "dataset_path": self.DATASET_PATH,
                "output_type": "multiple_choice",
                "metric_list": [
                    {
                        "metric": "acc",
                        "aggregation": "mean",
                        "higher_is_better": True,
                    }
                ],
            }
        )

    def validation_docs(self):
        dataset = self.dataset["train"].filter(
            lambda example: example["language"].startswith("kn"),
            num_proc=8,
            desc="Dropping validation instances whose language is not Kannada",
        )

        return dataset

    def test_docs(self):
        dataset = self.dataset["validation"].filter(
            lambda example: example["language"].startswith("kn"),
            num_proc=8,
            desc="Dropping test instances whose language is not Kannada",
        )

        return dataset


class Punjabi_Boolq(IndicBoolq):
    VERSION = 1

    def __init__(self, config=None):
        super().__init__(
            config={
                "metadata": {"version": self.VERSION},
                "task": "punjabi_boolq",
                "tag": "indic_boolq",
                "dataset_path": self.DATASET_PATH,
                "output_type": "multiple_choice",
                "metric_list": [
                    {
                        "metric": "acc",
                        "aggregation": "mean",
                        "higher_is_better": True,
                    }
                ],
            }
        )

    def validation_docs(self):
        dataset = self.dataset["train"].filter(
            lambda example: example["language"].startswith("pa"),
            num_proc=8,
            desc="Dropping validation instances whose language is not Punjabi",
        )

        return dataset

    def test_docs(self):
        dataset = self.dataset["validation"].filter(
            lambda example: example["language"].startswith("pa"),
            num_proc=8,
            desc="Dropping test instances whose language is not Punjabi",
        )

        return dataset


class Gujarati_Boolq(IndicBoolq):
    VERSION = 1

    def __init__(self, config=None):
        super().__init__(
            config={
                "metadata": {"version": self.VERSION},
                "task": "gujarati_boolq",
                "tag": "indic_boolq",
                "dataset_path": self.DATASET_PATH,
                "output_type": "multiple_choice",
                "metric_list": [
                    {
                        "metric": "acc",
                        "aggregation": "mean",
                        "higher_is_better": True,
                    }
                ],
            }
        )

    def validation_docs(self):
        dataset = self.dataset["train"].filter(
            lambda example: example["language"].startswith("gu"),
            num_proc=8,
            desc="Dropping validation instances whose language is not Gujarati",
        )

        return dataset

    def test_docs(self):
        dataset = self.dataset["validation"].filter(
            lambda example: example["language"].startswith("gu"),
            num_proc=8,
            desc="Dropping test instances whose language is not Gujarati",
        )

        return dataset


class Tamil_Boolq(IndicBoolq):
    VERSION = 1

    def __init__(self, config=None):
        super().__init__(
            config={
                "metadata": {"version": self.VERSION},
                "task": "tamil_boolq",
                "tag": "indic_boolq",
                "dataset_path": self.DATASET_PATH,
                "output_type": "multiple_choice",
                "metric_list": [
                    {
                        "metric": "acc",
                        "aggregation": "mean",
                        "higher_is_better": True,
                    }
                ],
            }
        )

    def validation_docs(self):
        dataset = self.dataset["train"].filter(
            lambda example: example["language"].startswith("ta"),
            num_proc=8,
            desc="Dropping validation instances whose language is not Tamil",
        )

        return dataset

    def test_docs(self):
        dataset = self.dataset["validation"].filter(
            lambda example: example["language"].startswith("ta"),
            num_proc=8,
            desc="Dropping test instances whose language is not Tamil",
        )

        return dataset


class Malayalam_Boolq(IndicBoolq):
    VERSION = 1

    def __init__(self, config=None):
        super().__init__(
            config={
                "metadata": {"version": self.VERSION},
                "task": "malayalam_boolq",
                "tag": "indic_boolq",
                "dataset_path": self.DATASET_PATH,
                "output_type": "multiple_choice",
                "metric_list": [
                    {
                        "metric": "acc",
                        "aggregation": "mean",
                        "higher_is_better": True,
                    }
                ],
            }
        )

    def validation_docs(self):
        dataset = self.dataset["train"].filter(
            lambda example: example["language"].startswith("ml"),
            num_proc=8,
            desc="Dropping validation instances whose language is not Malayalam",
        )

        return dataset

    def test_docs(self):
        dataset = self.dataset["validation"].filter(
            lambda example: example["language"].startswith("ml"),
            num_proc=8,
            desc="Dropping test instances whose language is not Malayalam",
        )

        return dataset


class Hindi_Boolq(IndicBoolq):
    VERSION = 1

    def __init__(self, config=None):
        super().__init__(
            config={
                "metadata": {"version": self.VERSION},
                "task": "hindi_boolq",
                "tag": "indic_boolq",
                "dataset_path": self.DATASET_PATH,
                "output_type": "multiple_choice",
                "metric_list": [
                    {
                        "metric": "acc",
                        "aggregation": "mean",
                        "higher_is_better": True,
                    }
                ],
            }
        )

    def validation_docs(self):
        dataset = self.dataset["train"].filter(
            lambda example: example["language"].startswith("hi"),
            num_proc=8,
            desc="Dropping validation instances whose language is not Hindi",
        )

        return dataset

    def test_docs(self):
        dataset = self.dataset["validation"].filter(
            lambda example: example["language"].startswith("hi"),
            num_proc=8,
            desc="Dropping test instances whose language is not Hindi",
        )

        return dataset


class Marathi_Boolq(IndicBoolq):
    VERSION = 1

    def __init__(self, config=None):
        super().__init__(
            config={
                "metadata": {"version": self.VERSION},
                "task": "marathi_boolq",
                "tag": "indic_boolq",
                "dataset_path": self.DATASET_PATH,
                "output_type": "multiple_choice",
                "metric_list": [
                    {
                        "metric": "acc",
                        "aggregation": "mean",
                        "higher_is_better": True,
                    }
                ],
            }
        )

    def validation_docs(self):
        dataset = self.dataset["train"].filter(
            lambda example: example["language"].startswith("mr"),
            num_proc=8,
            desc="Dropping validation instances whose language is not Marathi",
        )

        return dataset

    def test_docs(self):
        dataset = self.dataset["validation"].filter(
            lambda example: example["language"].startswith("mr"),
            num_proc=8,
            desc="Dropping test instances whose language is not Marathi",
        )

        return dataset


class Telugu_Boolq(IndicBoolq):
    VERSION = 1

    def __init__(self, config=None):
        super().__init__(
            config={
                "metadata": {"version": self.VERSION},
                "task": "telugu_boolq",
                "tag": "indic_boolq",
                "dataset_path": self.DATASET_PATH,
                "output_type": "multiple_choice",
                "metric_list": [
                    {
                        "metric": "acc",
                        "aggregation": "mean",
                        "higher_is_better": True,
                    }
                ],
            }
        )

    def validation_docs(self):
        dataset = self.dataset["train"].filter(
            lambda example: example["language"].startswith("te"),
            num_proc=8,
            desc="Dropping validation instances whose language is not Telugu",
        )

        return dataset

    def test_docs(self):
        dataset = self.dataset["validation"].filter(
            lambda example: example["language"].startswith("te"),
            num_proc=8,
            desc="Dropping test instances whose language is not Telugu",
        )

        return dataset
