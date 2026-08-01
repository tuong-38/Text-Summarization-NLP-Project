import os
import torch
from transformers import TrainingArguments, Trainer
from transformers import DataCollatorForSeq2Seq
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_from_disk
from textSummarizer.entity.config_entity import ModelTrainerConfig

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def train(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_ckpt)
        model_pegasus = AutoModelForSeq2SeqLM.from_pretrained(
            self.config.model_ckpt, 
            use_safetensors=True
        ).to(device)
        seq2seq_data_collator = DataCollatorForSeq2Seq(tokenizer, model=model_pegasus)
        
        # Load dataset
        dataset_samsum_pt = load_from_disk(self.config.data_path)

        trainer_args = TrainingArguments(
            output_dir=self.config.root_dir,
            num_train_epochs=self.config.num_train_epochs,
            warmup_steps=self.config.warmup_steps,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.per_device_train_batch_size,
            weight_decay=self.config.weight_decay,
            logging_steps=self.config.logging_steps,
            eval_strategy=self.config.eval_strategy,
            eval_steps=self.config.eval_steps,
            save_steps=1e6,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            fp16=True
        )

        trainer = Trainer(
            model=model_pegasus,
            args=trainer_args,
            processing_class=tokenizer,
            data_collator=seq2seq_data_collator,
            train_dataset=dataset_samsum_pt["test"].select(range(10)),      # Lấy 10 mẫu test
            eval_dataset=dataset_samsum_pt["validation"].select(range(10))  # Lấy 10 mẫu val
        )
        
        trainer.train()

        # Save model & tokenizer
        model_pegasus.save_pretrained(os.path.join(self.config.root_dir, "pegasus-samsum-model"))
        tokenizer.save_pretrained(os.path.join(self.config.root_dir, "tokenizer"))