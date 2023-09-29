from typing import List

import pandas as pd
import pytorch_lightning as pl
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, T5Tokenizer, get_scheduler

from sklearn.model_selection import train_test_split


class SummaryDataset(Dataset):
    def __init__(
        self,
        data: pd.DataFrame,
        model_name="t5-base",
        text_max_token_len: int = 512,
        summary_max_token_len: int = 128,
    ):
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.data = data
        self.text_max_token_len = text_max_token_len
        self.summary_max_token_len = summary_max_token_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        data_row: pd.Series = self.data.iloc[index]

        text: str = data_row["text"]

        # Encode the full, original text
        text_encoding = self.tokenizer(
            text,
            max_length=self.text_max_token_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt",
        )

        # Encode the "true" summary
        summary_encoding = self.tokenizer(
            data_row["summary"],
            max_length=self.summary_max_token_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt",
        )

        # The "label" is the encoded input_ids of the true summary
        labels = summary_encoding["input_ids"]
        labels_attention_mask = summary_encoding["attention_mask"]
        labels[labels == 0] = -100

        return dict(
            text=text,
            summary=data_row["summary"],
            text_input_ids=text_encoding["input_ids"].flatten(),
            text_attention_mask=text_encoding["attention_mask"].flatten(),
            labels=labels.flatten(),
            labels_attention_mask=labels_attention_mask.flatten(),
        )


class SummaryDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        tokenizer: T5Tokenizer,
        batch_size: int = 8,
        text_max_token_len: int = 512,
        summary_max_token_len: int = 128,
    ):
        super().__init__()

        self.train_df = train_df
        self.test_df = test_df

        self.batch_size = batch_size

        self.tokenizer = tokenizer
        self.text_max_token_len = text_max_token_len
        self.summary_max_token_len = summary_max_token_len

    def setup(self, stage=None):
        """
        Create training/test SummaryDatasets from pandas dataframes.
        """
        self.train_dataset = SummaryDataset(
            self.train_df,
            self.tokenizer,
            self.text_max_token_len,
            self.summary_max_token_len,
        )

        self.test_dataset = SummaryDataset(
            self.test_df,
            self.tokenizer,
            self.text_max_token_len,
            self.summary_max_token_len,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=2
        )

    def val_dataloader(self):
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=2
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0
        )


def csv_to_dataset(path_to_train_data, path_to_test_data):
    dataset = load_dataset(
        "csv", data_files={"train": path_to_train_data, "test": path_to_test_data}
    )
    train_dataset = dataset["train"]
    test_dataset = dataset["test"]
    return train_dataset, test_dataset


def tokenize_function(examples):
    tokenizer = T5Tokenizer.from_pretrained("t5-base")
    return tokenizer(examples["text"], padding="max_length", truncation=True)


def tokenize_dataset(train_dataset):
    tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
    tokenized_train_dataset = tokenized_train_dataset.remove_columns(["text"])
    tokenized_train_dataset.set_format("torch")
    return tokenized_train_dataset


def model_to_device(model):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    return device


def sample_batch(dataloader):
    return next(iter(dataloader)).copy()


def create_lr_scheduler(n_epochs, train_dataloader, optimizer):
    n_training_steps = n_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=n_training_steps,
    )
    return lr_scheduler


def predict_on_sequences(sequences: List[str], device, model, tokenizer):
    tokens = tokenizer(sequences, padding=True, truncation=True, return_tensors="pt")
    batch = {k: v.to(device) for k, v in tokens.items()}
    output = model(**batch)
    preds = torch.argmax(output.logits, dim=-1)
    return preds


def tokenize_datasets(sample):
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    return tokenizer(sample["text"], padding="max_length", truncation=True)


def generate_tokenized_splits(df):
    test_size = 0.1
    val_size = test_size
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=42, shuffle=True
    )
    train_df, val_df = train_test_split(
        train_df, test_size=val_size, random_state=42, shuffle=True
    )
    splits = {"train": train_df, "val": val_df, "test": test_df}
    dataset_splits = [
        Dataset.from_pandas(s).remove_columns(["__index_level_0__"])
        for s in splits.values()
    ]

    for split in dataset_splits.keys():
        dataset_splits[split] = dataset_splits[split].map(
            tokenize_datasets, batched=True
        )

        dataset_splits[split] = dataset_splits[split].remove_columns(["text"])
        dataset_splits[split].set_format(type="torch")

    return dataset_splits
