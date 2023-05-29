from datasets import load_dataset, Dataset
import typing as t
from transformers import (
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
    BatchEncoding,
    DataCollatorWithPadding
)
import pandas as pd
from pathlib import Path
import logging


def get_news_dataset(
        data_file: Path,
        tokenizer: t.Union[
            PreTrainedTokenizer,
            PreTrainedTokenizerFast]) -> Dataset:

    logging.info(f"Loading news dataset from {data_file}")
    if not data_file.exists():
        raise FileNotFoundError(f"File {data_file} not found")
    data = pd.read_csv(data_file)
    data = data[["title", "label"]]

    # rename title to text
    data = data.rename(columns={"title": "text"})

    dataset = Dataset.from_pandas(data)
    logging.info("Dataset loaded")

    return dataset
