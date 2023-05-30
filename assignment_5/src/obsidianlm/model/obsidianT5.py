from transformers import (
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    BatchEncoding,
    T5Model,
    T5TokenizerFast
)

from transformers.trainer import EvalPrediction
import typing as t
import numpy as np
import evaluate
from nltk.tokenize import sent_tokenize
from pathlib import Path


def get_model(
        max_len_in: int = 512,
        max_len_out: int = 512) -> t.Tuple[
            T5Model,
            T5TokenizerFast,
            t.Callable[[t.Any], t.Dict]]:
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
    tokenizer = T5TokenizerFast.from_pretrained(
        "google/flan-t5-base",
        use_fast=True)
    preprocess_fn = get_preprocess_fn(
        tokenizer,
        max_len_in,
        max_len_out,
        padding="max_length"
    )
    return model, tokenizer, preprocess_fn


def get_preprocess_fn(
        tokenizer: T5TokenizerFast,
        max_len_in: int,
        max_len_out: int,
        padding: str = "max_length",
        lbl_pad_id: int = -100) -> t.Callable[[t.Any], t.Dict]:
    max_len_in = min(max_len_in, 512)
    max_len_out = min(max_len_out, 512)

    def preprocess_text(text):
        inputs = "summarize: " + text["text"]
        encoding: BatchEncoding = tokenizer(
            inputs,
            max_length=max_len_in,
            padding=padding,
            truncation=True,
            return_tensors="pt",
        )
        input_ids, attention_mask = encoding.input_ids, encoding.attention_mask

        target_encoding: BatchEncoding = tokenizer(
            text_target=text["summary"],
            max_length=max_len_out,
            padding=padding,
            truncation=True,
            return_tensors="pt",
        )
        labels = target_encoding.input_ids
        labels[labels == tokenizer.pad_token_id] = lbl_pad_id
        return {
            "input_ids": input_ids.squeeze(0),
            "attention_mask": attention_mask.squeeze(0),
            "labels": labels.squeeze(0),
        }

    return preprocess_text


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # rougeLSum expects newline after each sentence
    preds = ["\n".join(sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(sent_tokenize(label)) for label in labels]

    return preds, labels


def compute_metrics(eval_pred: EvalPrediction) -> t.Dict:
    metric = evaluate.load("accuracy")
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)  # type: ignore
    metrics = metric.compute(predictions=predictions, references=labels)
    return metrics if metrics else {}


def get_data_collator(model, tokenizer, lbl_pad_id):
    return DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=lbl_pad_id,
        pad_to_multiple_of=8,
        return_tensors="pt",
    )


def train_model(
        model,
        tokenizer,
        train_dataset,
        eval_dataset: t.Optional[t.Any] = None,
        batch_size=8,
        rouge=True,
        lbl_pad_id=-100,
        ds_length: t.Optional[int] = None,
        out_path: t.Optional[Path] = None):

    metric_fn = compute_metrics
    data_collator = get_data_collator(model, tokenizer, lbl_pad_id)

    if rouge:
        metric = evaluate.load("rouge")

        def rouge_compute_metrics(eval_pred: EvalPrediction) -> t.Dict:
            preds, labels = eval_pred
            if isinstance(preds, tuple):
                preds = preds[0]
            decoded_preds = tokenizer.batch_decode(
                preds,
                skip_special_tokens=True
            )
            labels = np.where(
                labels != lbl_pad_id,
                labels,
                tokenizer.pad_token_id
            )
            decoded_labels = tokenizer.batch_decode(
                labels,
                skip_special_tokens=True
            )
            decoded_preds, decoded_labels = postprocess_text(
                decoded_preds,
                decoded_labels
            )
            result = metric.compute(
                predictions=decoded_preds,
                references=decoded_labels,
                use_stemmer=True
            )
            if result is None:
                return {}
            result = {k: round(v * 100, 4) for k, v in result.items()}
            prediction_lens = [
                np.count_nonzero(
                    pred != tokenizer.pad_token_id
                ) for pred in preds]
            result["gen_len"] = np.mean(prediction_lens)
            return result

        metric_fn = rouge_compute_metrics

    default_args = {
        "evaluation_strategy": "no",
        "num_train_epochs": 1,
        "log_level": "error",
        "fp16": True,
        "tf32": False,
        "learning_rate": 3e-4,
        "gradient_accumulation_steps": 16,
        "report_to": "tensorboard",
        "gradient_checkpointing": True,  # memory savings, but slower
        "optim": "adamw_apex_fused",
        "save_steps": 1,
        "save_total_limit": 1,
        "dataloader_pin_memory": True,
        "logging_steps": 1,
        "do_eval": False,
    }
    if ds_length:
        default_args["max_steps"] = ds_length // batch_size
    if out_path:
        default_args["output_dir"] = str(out_path)
        default_args["overwrite_output_dir"] = True
        default_args["logging_dir"] = str(out_path / "runs")
        default_args["logging_strategy"] = "steps"
    if eval_dataset:
        default_args["evaluation_strategy"] = "steps"
        default_args["eval_steps"] = 1
        default_args["do_eval"] = True
    training_args = Seq2SeqTrainingArguments(
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        predict_with_generate=rouge,
        **default_args)
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=metric_fn,
        data_collator=data_collator,
    )

    trainer.train()
