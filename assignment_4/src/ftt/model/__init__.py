import logging
import typing as t
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast
)
import torch


def get_model_and_tokenizer() -> t.Tuple[
        AutoModelForSequenceClassification,
        t.Union[
            PreTrainedTokenizer,
            PreTrainedTokenizerFast]]:
    model_name = "j-hartmann/emotion-english-distilroberta-base"
    id2label = {
        0: "anger",
        1: "disgust",
        2: "fear",
        3: "joy",
        4: "neutral",
        5: "sadness",
        6: "surprise"
    }
    label2id = {
        "anger": 0,
        "disgust": 1,
        "fear": 2,
        "joy": 3,
        "neutral": 4,
        "sadness": 5,
        "surprise": 6
    }

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    logging.info("Tokenizer loaded")
    logging.info(tokenizer)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=7, id2label=id2label, label2id=label2id
    )
    model = model.to_bettertransformer()

    logging.info("Model loaded")
    return model, tokenizer


def inference(model, data, tokenizer):
    text = data["text"]
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        output = model(**inputs)
        logits = output.logits
        probs = torch.softmax(logits, dim=1)
    return probs
