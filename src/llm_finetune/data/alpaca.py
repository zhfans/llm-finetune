from datasets import DatasetDict
from datasets import load_dataset as load_hf_dataset


def load_dataset() -> DatasetDict:
    return load_hf_dataset("tatsu-lab/alpaca")
