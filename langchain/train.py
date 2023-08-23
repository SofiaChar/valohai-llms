import sys
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import pipeline, set_seed
import matplotlib.pyplot as plt
from datasets import load_dataset, load_metric
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import nltk
from nltk.tokenize import sent_tokenize
from tqdm import tqdm
import torch

nltk.download("punkt")
from transformers import DataCollatorForSeq2Seq, TrainingArguments, Trainer
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import valohai
import os
from transformers import get_scheduler
os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'


class PegasusSamsumTrainer:
    def __init__(self, model_ckpt, batch_size=2, num_epochs=1, warmup_steps=500, evaluation_steps=500):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print('torch.cuda.device_count() ', torch.cuda.device_count())
        print('self.device ', self.device)

        print('__Python VERSION:', sys.version)
        print('__pyTorch VERSION:', torch.__version__)
        print('__CUDA VERSION')
        from subprocess import call
        print('__CUDNN VERSION:', torch.backends.cudnn.version())
        print('__Number CUDA Devices:', torch.cuda.device_count())
        print('__Devices')
        call(["nvidia-smi", "--format=csv",
              "--query-gpu=index,name,driver_version,memory.total,memory.used,memory.free"])
        print('Active CUDA Device: GPU', torch.cuda.current_device())

        print('Available devices ', torch.cuda.device_count())
        print('Current cuda device ', torch.cuda.current_device())
        self.model_ckpt = model_ckpt
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.warmup_steps = warmup_steps
        self.evaluation_steps = evaluation_steps

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_ckpt)
        self.model_pegasus = AutoModelForSeq2SeqLM.from_pretrained(self.model_ckpt).to(self.device)

    def generate_batch_sized_chunks(self, list_of_elements):
        """split the dataset into smaller batches that we can process simultaneously
        Yield successive batch-sized chunks from list_of_elements."""
        for i in range(0, len(list_of_elements), self.batch_size):
            yield list_of_elements[i: i + self.batch_size]

    def calculate_metric_on_test_ds(self, dataset, metric):
        article_batches = list(self.generate_batch_sized_chunks(dataset['article']))
        target_batches = list(self.generate_batch_sized_chunks(dataset['highlights']))

        for article_batch, target_batch in tqdm(zip(article_batches, target_batches), total=len(article_batches)):
            inputs = self.tokenizer(article_batch, max_length=1024, truncation=True,
                                    padding="max_length", return_tensors="pt")

            summaries = self.model_pegasus.generate(input_ids=inputs["input_ids"].to(self.device),
                                                    attention_mask=inputs["attention_mask"].to(self.device),
                                                    length_penalty=0.8, num_beams=8, max_length=128)

            decoded_summaries = [self.tokenizer.decode(s, skip_special_tokens=True,
                                                       clean_up_tokenization_spaces=True) for s in summaries]

            decoded_summaries = [d.replace("", " ") for d in decoded_summaries]

            metric.add_batch(predictions=decoded_summaries, references=target_batch)

        score = metric.compute()
        return score

    def convert_examples_to_features(self, example_batch):
        input_encodings = self.tokenizer(example_batch['dialogue'], padding="max_length", truncation=True)

        with self.tokenizer.as_target_tokenizer():
            target_encodings = self.tokenizer(example_batch['summary'], padding="max_length", truncation=True)
        # print("input_encodings['input_ids']", input_encodings['input_ids'])
        return {
            'input_ids': input_encodings['input_ids'][0],
            'attention_mask': input_encodings['attention_mask'],
            'labels': target_encodings['input_ids']
        }

    def train(self, output_dir, train_dataset, eval_dataset):
        dataset_samsum_pt = train_dataset.map(self.convert_examples_to_features, batched=True)

        seq2seq_data_collator = DataCollatorForSeq2Seq(self.tokenizer, model=self.model_pegasus)

        trainer_args = TrainingArguments(
            output_dir=output_dir, num_train_epochs=self.num_epochs, warmup_steps=self.warmup_steps,
            per_device_train_batch_size=self.batch_size, per_device_eval_batch_size=self.batch_size,
            weight_decay=0.01, logging_steps=10, evaluation_strategy='steps', eval_steps=self.evaluation_steps,
            save_steps=1e6, gradient_accumulation_steps=16
        )

        trainer = Trainer(model=self.model_pegasus, args=trainer_args, tokenizer=self.tokenizer,
                          data_collator=seq2seq_data_collator, train_dataset=dataset_samsum_pt,
                          eval_dataset=eval_dataset)

        trainer.train()

        self.model_pegasus.save_pretrained(output_dir)

    def training_loop(self, output_dir, train_dataset, eval_dataset):
        optimizer = AdamW(self.model_pegasus.model.parameters(), lr=5e-5)
        lr_scheduler = get_scheduler(
            name="linear", optimizer=optimizer, num_warmup_steps=self.warmup_steps, num_training_steps=16
        )

        dataset_samsum_pt = train_dataset.map(self.convert_examples_to_features, batched=True)

        seq2seq_data_collator = DataCollatorForSeq2Seq(self.tokenizer, model=self.model_pegasus)

        train_data = DataLoader(dataset_samsum_pt, collate_fn=seq2seq_data_collator, shuffle=True, batch_size=self.batch_size)

        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        for epoch in tqdm(range(self.num_epochs)):
            self.model_pegasus.train()
            for batch in train_data:
                # batch = print('k ', k ) for k, v in batch.items()
                outputs = self.model_pegasus(**batch)
                loss = outputs.loss
                loss.backward()

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()


def run():
    model_ckpt = "facebook/bart-large-cnn"
    trainer = PegasusSamsumTrainer(model_ckpt=model_ckpt)
    dataset_samsum = load_dataset('samsum')

    print(f"Train dataset size: {len(dataset_samsum['train'])}")
    print(f"Test dataset size: {len(dataset_samsum['test'])}")

    train_dataset = dataset_samsum["train"]
    eval_dataset = dataset_samsum["validation"]

    output_dir = "bart-samsum-model"
    # trainer.train(output_dir=output_dir, train_dataset=train_dataset, eval_dataset=eval_dataset)
    trainer.training_loop(output_dir=output_dir, train_dataset=train_dataset, eval_dataset=eval_dataset)


# def init(master_url, my_rank, world_size, fn):
#     dist.init_process_group(init_method=master_url, rank=my_rank, world_size=world_size, backend='gloo')
#     fn(my_rank, world_size)


if __name__ == '__main__':
    # master_port = 1234
    # master_ip = valohai.distributed.master().primary_local_ip
    # url = f"tcp://{master_ip}:{master_port}"
    #
    # size = valohai.distributed.required_count
    # rank = valohai.distributed.me().rank
    #
    # mp.set_start_method('spawn')
    # p = mp.Process(target=init, args=(url, rank, size, run))
    # p.start()
    # p.join()
    run()
