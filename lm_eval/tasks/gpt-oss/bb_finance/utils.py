import logging
import datasets

eval_logger = logging.getLogger(__name__)

def load_dataset(**kwargs):

    # Get specific qa split
    range = kwargs.get("range")
    split=kwargs.get("split")

    eval_logger.info(
        f"Loading bb_finance dataset: range={range}"
    )
    split_range=str(split+'['+range+']')
    print(split_range)
    dataset = datasets.load_dataset(
        'bharatgenai/BhashaBench-Finance','English', split=str(split+'['+range+']')
    )
    return {split: dataset}