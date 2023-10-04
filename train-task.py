import argparse
import logging
import math
import os
import random
import torch.distributed as dist
import datasets
import nltk
import numpy as np
import torch
import valohai
from random import Random
from datasets import load_dataset, load_metric
from torch.utils.data.dataloader import DataLoader
from tqdm.auto import tqdm
import torch.multiprocessing as mp
import transformers

from filelock import FileLock
from torch.optim import AdamW
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    SchedulerType,
    get_scheduler,
    set_seed,
)

summarization_name_mapping = {
    "amazon_reviews_multi": ("review_body", "review_title"),
    "big_patent": ("description", "abstract"),
    "cnn_dailymail": ("article", "highlights"),
    "orange_sum": ("text", "summary"),
    "pn_summary": ("article", "summary"),
    "psc": ("extract_text", "summary_text"),
    "samsum": ("dialogue", "summary"),
    "thaisum": ("body", "summary"),
    "xglue": ("news_body", "news_title"),
    "xsum": ("document", "summary"),
    "wiki_summary": ("article", "highlights"),
}


class Partition:

    def __init__(self, data, index):
        self.data = data
        self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        data_idx = self.index[index]
        return self.data[data_idx]


class DataPartitioner:

    def __init__(self, dataset, sizes=None, seed=1234):
        if sizes is None:
            sizes = [0.7, 0.2, 0.1]
        self.data = dataset
        self.partitions = []
        rng = Random()
        rng.seed(seed)
        data_len = len(dataset)
        indexes = [x for x in range(0, data_len)]
        rng.shuffle(indexes)
        for frac in sizes:
            part_len = int(frac * data_len)
            self.partitions.append(indexes[0:part_len])
            indexes = indexes[part_len:]

    def use(self, partition):
        return Partition(self.data, self.partitions[partition])

def average_gradients(model):
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
        param.grad.data /= size


def train(my_rank):
    torch.manual_seed(1234)

    logger = logging.getLogger(__name__)
    device = torch.device("cuda:{}".format(my_rank))

    num_train_epochs = 5
    datasets.utils.logging.set_verbosity_error()
    transformers.utils.logging.set_verbosity_error()
    raw_datasets = load_dataset('samsum')
    model_ckpt = "facebook/bart-large-cnn"
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

    model = AutoModelForSeq2SeqLM.from_pretrained(model_ckpt).to(device)

    model.resize_token_embeddings(len(tokenizer))
    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

    prefix = ""

    column_names = raw_datasets["train"].column_names

    # Get the column names for input/target.
    dataset_columns = summarization_name_mapping.get('samsum', None)
    text_column_name = dataset_columns[0] if dataset_columns is not None else column_names[0]

    padding = "max_length"
    summary_column = 'summary'

    # Temporarily set max_target_length for training.
    max_target_length = 128
    padding = "max_length"

    def preprocess_function(examples):
        inputs = examples[text_column_name]
        targets = examples[summary_column]
        inputs = [prefix + inp for inp in inputs]
        model_inputs = tokenizer(inputs, max_length=1024, padding=padding, truncation=True)

        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, max_length=max_target_length, padding=padding, truncation=True)

        if padding == "max_length":
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def partition_dataset(preprocesed_dataset, collator):
        size = dist.get_world_size()
        bsz = int(2 / float(size))
        partition_sizes = [1.0 / size for _ in range(size)]
        partition = DataPartitioner(preprocesed_dataset, partition_sizes)
        partition = partition.use(dist.get_rank())
        set = DataLoader(
            partition,
            batch_size=bsz,
            shuffle=True,
            collate_fn=collator
        )
        print('batch_size', bsz)
        return set, bsz


    processed_datasets = raw_datasets.map(
        preprocess_function, batched=True, remove_columns=column_names
    )

    train_dataset = processed_datasets["train"]
    eval_dataset = processed_datasets["validation"]

    for index in random.sample(range(len(train_dataset)), 1):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    label_pad_token_id = -100
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
    )

    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]

        # rougeLSum expects newline after each sentence
        preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
        labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

        return preds, labels

    train_dataloader, batch_size = partition_dataset(train_dataset, data_collator)
    # eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=1)

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=5e-5)

    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = len(train_dataloader)
    max_train_steps = num_train_epochs * num_update_steps_per_epoch

    lr_scheduler = get_scheduler(
        name='linear',
        optimizer=optimizer,
        num_training_steps=max_train_steps,
        num_warmup_steps=1
    )

    # Train!
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {num_train_epochs}")

    progress_bar = tqdm(range(max_train_steps))
    completed_steps = 0

    num_batches = math.ceil(len(train_dataloader.dataset) / float(batch_size))

    for epoch in range(num_train_epochs):
        model.train()
        epoch_loss = 0.0

        for step, batch in enumerate(train_dataloader):
            outputs = model(**batch.to(device))
            loss = outputs.loss
            epoch_loss += loss.item()
            loss.backward()
            average_gradients(model)

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)
            completed_steps += 1

            if completed_steps >= max_train_steps:
                break

        print(f'Rank {dist.get_rank()}, epoch {epoch}: {epoch_loss / num_batches}')

        # model.eval()
        #
        # gen_kwargs = {
        #     "max_length": 128,
        #     "num_beams": None,
        # }
        # for step, batch in enumerate(eval_dataloader):
        #     with torch.no_grad():
        #         generated_tokens = accelerator.unwrap_model(model).generate(
        #             batch["input_ids"],
        #             attention_mask=batch["attention_mask"],
        #             **gen_kwargs,
        #         )
        #
        #         generated_tokens = accelerator.pad_across_processes(
        #             generated_tokens, dim=1, pad_index=tokenizer.pad_token_id
        #         )
        #         labels = batch["labels"]
        #         if not args.pad_to_max_length:
        #             # If we did not pad to max length, we need to pad the labels too
        #             labels = accelerator.pad_across_processes(batch["labels"], dim=1, pad_index=tokenizer.pad_token_id)
        #
        #         generated_tokens = accelerator.gather(generated_tokens).cpu().numpy()
        #         labels = accelerator.gather(labels).cpu().numpy()
        #
        #         labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        #         if isinstance(generated_tokens, tuple):
        #             generated_tokens = generated_tokens[0]
        #         decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        #         decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        #
        #         decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
        #
        #         metric.add_batch(predictions=decoded_preds, references=decoded_labels)
        # result = metric.compute(use_stemmer=True)
        # # Extract a few results from ROUGE
        # result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
        #
        # result = {k: round(v, 4) for k, v in result.items()}
        #
        # logger.info(result)

        # if argsoutput_dir is not None:
        #     accelerator.wait_for_everyone()
        #     unwrapped_model = accelerator.unwrap_model(model)
        #     unwrapped_model.save_pretrained(args.output_dir, save_function=accelerator.save)

def init(master_url, my_rank, world_size, fn):
    dist.init_process_group(init_method=master_url, rank=my_rank, world_size=world_size, backend='nccl')
    fn(my_rank)


if __name__ == '__main__':

    master_port = 1234
    master_ip = valohai.distributed.master().primary_local_ip
    url = f"tcp://{master_ip}:{master_port}"

    size = valohai.distributed.required_count
    rank = valohai.distributed.me().rank

    print('rank ', rank)
    print('size ', size)

    mp.set_start_method('spawn')
    p = mp.Process(target=init, args=(url, rank, size, train))
    p.start()
    print('p start')
    p.join()
    print('p join')

