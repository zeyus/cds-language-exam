from transformers import (
    Seq2SeqTrainer,
    Trainer,
    Seq2SeqTrainingArguments,
    TrainingArguments,
    DataCollatorForSeq2Seq,
    DefaultDataCollator,
    BatchEncoding,
    T5ForConditionalGeneration,
    T5ForQuestionAnswering,
    T5TokenizerFast,
    GenerationConfig,
)
import torch
from transformers.trainer import EvalPrediction, get_last_checkpoint
import typing as t
import numpy as np
import evaluate
from nltk.tokenize import sent_tokenize
from pathlib import Path
import gc


def get_model(
        max_len_in: int = 512,
        max_len_out: int = 512,
        qa_head: bool = False) -> t.Tuple[
            t.Union[T5ForConditionalGeneration, T5ForQuestionAnswering],
            T5TokenizerFast,
            t.Callable[[t.Any], t.Dict]]:
    model_class = T5ForQuestionAnswering if qa_head else T5ForConditionalGeneration
    model = model_class.from_pretrained("google/flan-t5-base", max_length=max_len_out, min_length=None)
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


def get_model_from_checkpoint(
        checkpoint_path: Path,
        max_len_in: int = 512,
        max_len_out: int = 512,
        qa_head: bool = False) -> t.Tuple[
            t.Union[T5ForConditionalGeneration, T5ForQuestionAnswering],
            T5TokenizerFast,
            t.Callable[[t.Any], t.Dict]]:
    model_class = T5ForQuestionAnswering if qa_head else T5ForConditionalGeneration
    model = model_class.from_pretrained(checkpoint_path)
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

    # end of output
    eos_token_id: int = tokenizer.eos_token_id  # type: ignore
    sentinal_token_ids: t.List[int] = tokenizer.get_sentinel_token_ids()  # type: ignore
    max_mask_tokens = len(sentinal_token_ids)

    if not mask_lm:
        def preprocess_text(text) -> t.Dict[str, torch.Tensor]:
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

    def mask_tokens(text) -> t.Dict[str, torch.Tensor]:
        inputs = tokenizer.decode(sentinal_token_ids[0]) + text["text"]
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
        # print("\n\n input 0 \n", input_ids[0], "\n\nattn 0\n", attention_mask[0], "\n\nspecial 0 \n", special_tokens_mask[0])
        # print("\n\ndecoded 0", tokenizer.decode(input_ids[0]))
        # labels = input_ids.detach().clone()
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
        # do not allow more than 100 masked indices
        masked_indices = masked_indices & (
            torch.cumsum(masked_indices, dim=1) <= max_mask_tokens)
        # create labels of size (batch_size, max_mask_tokens * 2)
        # create tensor filled with lbl_pad_id
        # lbl_pad_id is ignored by loss function
        labels = torch.full(
            (input_ids.shape[0], input_ids.shape[1]),
            lbl_pad_id,
            dtype=torch.long,
            device=device
        )
        # loop through batches
        for batch_idx in range(input_ids.shape[0]):
            token = 0
            # we only need to iterate through the masked indices
            masked_idxs = torch.argwhere(masked_indices[batch_idx]).flatten()
            if len(masked_idxs) == 0:
                # no masked tokens
                labels[batch_idx, 0] = eos_token_id
                continue
            prev_masked_idx = -100
            for idx in masked_idxs:
                if idx == prev_masked_idx + 1:
                    # if idx is consecutive, skip
                    continue
                # save original token in labels
                labels[batch_idx, token * 2 + 1] = input_ids[batch_idx, idx]
                # Replace original token with sentinel token
                input_ids[batch_idx, idx] = sentinal_token_ids[token]
                # update attention mask
                # attention_mask[batch_idx, idx] = 1
                # Add sentinel token to labels
                labels[batch_idx, token * 2] = sentinal_token_ids[token]
                token += 1
                prev_masked_idx = idx
            # Add eos token to labels
            labels[batch_idx, (token - 1) * 2 + 2] = eos_token_id

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
        model: t.Union[T5ForConditionalGeneration, T5ForQuestionAnswering],
        tokenizer: T5TokenizerFast,
        train_dataset: t.Any,
        eval_dataset: t.Optional[t.Any] = None,
        batch_size=8,
        rouge=False,
        lbl_pad_id=-100,
        ds_length: t.Optional[int] = None,
        out_path: t.Optional[Path] = None,
        mask_lm: bool = True,
        max_len_gen: int = 512,
        eval_batch_size = 1) -> None:

    def decode_for_metrics(eval_pred: EvalPrediction) -> t.Tuple[np.ndarray, np.ndarray]:
        preds = eval_pred.predictions
        labels = eval_pred.label_ids
        if isinstance(preds, tuple):
            preds = preds[0]
        if isinstance(labels, tuple):
            labels = labels[0]
        # preds are numpy arrays of shape (batch_size, seq_len, vocab_size)
        # get argmax of prediction scores
        preds = np.argmax(preds, axis=-1)
        preds = tokenizer.batch_decode(
            preds,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )
        labels = np.where(
            labels != lbl_pad_id,
            labels,
            tokenizer.eos_token_id  # type: ignore
        )
        labels = tokenizer.batch_decode(
            labels,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )
        # may have to skip special tokens, or somehow fix UNK / SEP so words don't collapse together

        return preds, labels  # type: ignore

    if rouge:
        metric = evaluate.load("rouge")

        def rouge_compute_metrics(eval_pred: EvalPrediction) -> t.Dict:
            decoded_preds, decoded_labels = decode_for_metrics(eval_pred)
            r = metric.compute(
                predictions=decoded_preds,
                references=decoded_labels,
                use_stemmer=True
            )
            return r if r is not None else {}

        metric_fn = rouge_compute_metrics
    else:
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        metric = evaluate.load("ter")

        def compute_metrics(eval_pred: EvalPrediction) -> t.Dict:
            decoded_preds, decoded_labels = decode_for_metrics(eval_pred)
            r = metric.compute(
                predictions=decoded_preds,
                references=decoded_labels,
                normalized=True,
                ignore_punct=True,
            )
            return r if r is not None else {}
        metric_fn = compute_metrics

    def wrap_metric_fn(eval_pred: EvalPrediction) -> t.Dict:
        """Wrap metric_fn to free up memory after each (batched) evaluation step"""
        metrics = metric_fn(eval_pred)
        gc.collect()
        torch.cuda.empty_cache()
        return metrics
    wrapped_metric_fn = wrap_metric_fn

    metric_str = "no_eval"
    if out_path:
        if eval_dataset:
            metric_str = "rouge_metric" if rouge else "ter_metric"

    # Define some defaults for training
    default_args = {
        "evaluation_strategy": "no",
        "num_train_epochs": 5,
        "log_level": "error",
        "fp16": False,
        # "tf32": False,
        "learning_rate": 3e-4,
        "gradient_accumulation_steps": 16,
        "report_to": "tensorboard",
        "gradient_checkpointing": True,  # memory savings, but slower
        "optim": "adamw_apex_fused",
        # "optim": "adafactor",
        "save_steps": 5,
        "save_total_limit": 5,
        "dataloader_pin_memory": True,
        "logging_steps": 1,
        "metric_for_best_model": "eval_score" if not rouge else "eval_rouge1",
        "do_eval": False,
        "load_best_model_at_end": False,
        "greater_is_better": True if rouge else False,
    }
    if ds_length:
        default_args["max_steps"] = ds_length // batch_size
    if out_path:
        default_args["overwrite_output_dir"] = True
        default_args["logging_dir"] = str(out_path / "runs" / metric_str)
        default_args["logging_strategy"] = "steps"
    if eval_dataset:
        # if the evaluation dataset has been provided
        # run evaluation every 5 steps
        default_args["evaluation_strategy"] = "steps"
        default_args["eval_steps"] = 5
        default_args["do_eval"] = True
        default_args["load_best_model_at_end"] = True

    generation_config = GenerationConfig(**{
        "min_length": 0,  # for some reason if this is not 0, people have reported issues with the model generation
        "decoder_start_token_id": tokenizer.pad_token_id,  # the decoder_input_ids start with the pad token
        "num_beams": 5,  # number of beams for beam search
        "early_stopping": True,
        "max_length": max_len_gen,
    })
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=lbl_pad_id,
        pad_to_multiple_of=8,
        return_tensors="pt",
    )
    training_args = Seq2SeqTrainingArguments(
        output_dir="" if not out_path or not metric_str else str(out_path / metric_str),
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=eval_batch_size,
        predict_with_generate=rouge,
        **default_args,
        generation_max_length=max_len_gen,
        generation_config=generation_config)
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=wrapped_metric_fn,
        data_collator=data_collator,
    )
    # else:
    #     data_collator = DefaultDataCollator(
    #         # tokenizer=tokenizer,
    #         # pad_to_multiple_of=8,
    #         return_tensors="pt",
    #     )
    #     # default_args["optim"] = "adafactor"
    #     training_args = TrainingArguments(
    #         output_dir=str(out_path),
    #         per_device_train_batch_size=batch_size,
    #         per_device_eval_batch_size=batch_size,
    #         **default_args)
    #     trainer = Trainer(
    #         model=model,
    #         args=training_args,
    #         train_dataset=train_dataset,
    #         eval_dataset=eval_dataset,
    #         compute_metrics=wrapped_metric_fn,
    #         data_collator=data_collator,
    #     )
    last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint:
        print(f"Resuming from checkpoint: {last_checkpoint}")
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
