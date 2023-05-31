from transformers import (
    Seq2SeqTrainer,
    Trainer,
    Seq2SeqTrainingArguments,
    TrainingArguments,
    DataCollatorForSeq2Seq,
    DefaultDataCollator,
    BatchEncoding,
    T5ForConditionalGeneration,
    T5TokenizerFast
)
import torch
from transformers.trainer import EvalPrediction
import typing as t
import numpy as np
import evaluate
from nltk.tokenize import sent_tokenize
from pathlib import Path


def get_model(
        max_len_in: int = 512,
        max_len_out: int = 512) -> t.Tuple[
            T5ForConditionalGeneration,
            T5TokenizerFast,
            t.Callable[[t.Any], t.Dict]]:
    model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")
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
        lbl_pad_id: int = -100,
        mask_lm: bool = True,
        mask_prob: float = 0.15) -> t.Callable[[t.Any], t.Dict]:
    max_len_in = min(max_len_in, 512)
    max_len_out = min(max_len_out, 512)

    if not mask_lm:
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

    def mask_tokens(text):
        inputs = text["text"]
        encoding: BatchEncoding = tokenizer(
            inputs,
            max_length=max_len_in,
            padding=padding,
            truncation=True,
            return_special_tokens_mask=True,
            return_tensors="pt",
        )
        input_ids, attention_mask, special_tokens_mask = (
            encoding.input_ids,
            encoding.attention_mask,
            encoding.special_tokens_mask,
        )
        device = input_ids.device

        labels = input_ids.detach().clone()

        probability_matrix = torch.full(
            input_ids.shape,
            mask_prob,
            dtype=torch.float32,
            device=device
        )
        special_tokens_mask = special_tokens_mask.bool()

        probability_matrix[special_tokens_mask] = 0.0

        masked_indices = torch.binomial(
            torch.ones(input_ids.shape, dtype=torch.float32, device=device),
            probability_matrix).bool()

        labels[~masked_indices] = lbl_pad_id

        extra_token_counter = 0

        for idx in range(input_ids.shape[1]):
            if masked_indices[0, idx] and extra_token_counter < 100:  # Ensure we don't exceed the maximum of 100 sentinel tokens
                # Replace with sentinel token in input
                input_ids[0, idx] = tokenizer.additional_special_tokens_ids[extra_token_counter]
                # Replace sentinel token followed by original token in labels
                labels[0, idx] = tokenizer.additional_special_tokens_ids[extra_token_counter]
                if idx < input_ids.shape[1] - 1:  # Make sure not to go out of bounds
                    labels = torch.cat((labels[:, :idx+1], input_ids[0, idx+1].unsqueeze(0).unsqueeze(0), labels[:, idx+2:]), dim=1)
                extra_token_counter += 1

        # Convert tensors back to half precision if necessary
        if device.type == 'cuda':
            input_ids = input_ids.half()
            attention_mask = attention_mask.half()

        return {
            "input_ids": input_ids.squeeze(0),
            "attention_mask": attention_mask.squeeze(0),
            "labels": labels.squeeze(0),
        }

    return mask_tokens


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # rougeLSum expects newline after each sentence
    preds = ["\n".join(sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(sent_tokenize(label)) for label in labels]

    return preds, labels


def train_model(
        model,
        tokenizer,
        train_dataset,
        eval_dataset: t.Optional[t.Any] = None,
        batch_size=8,
        rouge=False,
        lbl_pad_id=-100,
        ds_length: t.Optional[int] = None,
        out_path: t.Optional[Path] = None,
        mask_lm: bool = True):

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
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        def compute_metrics(eval_pred: EvalPrediction) -> t.Dict:
            logits, labels = eval_pred.predictions, eval_pred.label_ids
            # logits, labels = eval_pred

            # Ensure these are torch tensors
            logits = torch.from_numpy(logits).to(device)
            labels = torch.from_numpy(labels).to(device)

            # Convert to boolean tensor and reshape to match `labels`
            mask = (labels != -100).reshape_as(labels)

            logits = logits[mask]
            labels = labels[mask]

            # convert back to numpy if needed
            logits = logits.numpy()
            labels = labels.numpy()

            # compute the softmax
            logits = np.exp(logits) / np.exp(logits).sum(-1, keepdims=True)

            # compute the loss
            loss = -np.log(logits[np.arange(len(logits)), labels]).mean()

            num_labels = np.count_nonzero(mask.numpy())

            return {"loss": loss, "num_labels": num_labels}

        metric_fn = compute_metrics

    default_args = {
        "evaluation_strategy": "no",
        "num_train_epochs": 5,
        "log_level": "error",
        "fp16": True,
        "tf32": False,
        "learning_rate": 3e-4,
        "gradient_accumulation_steps": 16,
        "report_to": "tensorboard",
        "gradient_checkpointing": True,  # memory savings, but slower
        "optim": "adamw_apex_fused",
        "save_steps": 1,
        "save_total_limit": 5,
        "dataloader_pin_memory": True,
        "logging_steps": 1,
        "metric_for_best_model": "loss",
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

    if not mask_lm:
        data_collator = DataCollatorForSeq2Seq(
            tokenizer,
            model=model,
            label_pad_token_id=lbl_pad_id,
            pad_to_multiple_of=8,
            return_tensors="pt",
        )
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
    else:
        data_collator = DefaultDataCollator(
            # tokenizer=tokenizer,
            # pad_to_multiple_of=8,
            return_tensors="pt",
        )

        default_args["optim"] = "adafactor"
        training_args = TrainingArguments(
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            **default_args)
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=metric_fn,
            data_collator=data_collator,
        )
    trainer.train(resume_from_checkpoint=False)
