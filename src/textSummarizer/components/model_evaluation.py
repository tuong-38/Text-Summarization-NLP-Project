import torch
import pandas as pd
from tqdm import tqdm
from datasets import load_from_disk
from evaluate import load
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from textSummarizer.entity.config_entity import ModelEvaluationConfig

class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    def generate_batch_sized_chunks(self, list_of_elements, batch_size):
        """Chia dữ liệu thành các batch nhỏ hơn để xử lý trên GPU"""
        for i in range(0, len(list_of_elements), batch_size):
            yield list_of_elements[i : i + batch_size]

    def calculate_metric_on_test_ds(self, dataset, metric, model, tokenizer, 
                                   batch_size=16, device="cuda" if torch.cuda.is_available() else "cpu", 
                                   column_text="dialogue", column_summary="summary"):
        text_batches = list(self.generate_batch_sized_chunks(dataset[column_text], batch_size))
        target_batches = list(self.generate_batch_sized_chunks(dataset[column_summary], batch_size))

        for text_batch, target_batch in tqdm(zip(text_batches, target_batches), total=len(text_batches)):
            inputs = tokenizer(text_batch, max_length=1024, truncation=True, padding="max_length", return_tensors="pt")
            
            # Sinh văn bản tóm tắt bằng GPU
            summaries = model.generate(
                input_ids=inputs["input_ids"].to(device),
                attention_mask=inputs["attention_mask"].to(device),
                length_penalty=0.8,
                num_beams=8,
                max_length=128
            )
            
            # Decode kết quả ra chuỗi văn bản
            decoded_summaries = [tokenizer.decode(s, skip_special_tokens=True, clean_up_tokenization_spaces=True) for s in summaries]
            decoded_summaries = [d.replace("", " ") for d in decoded_summaries]
            
            metric.add_batch(predictions=decoded_summaries, references=target_batch)
            
        # Tính toán điểm ROUGE
        score = metric.compute()
        return score

    def evaluate(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_path)
        model_pegasus = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_path).to(device)
        
        # Load dataset
        dataset_samsum_pt = load_from_disk(self.config.data_path)
        
        # Load chỉ số ROUGE từ thư viện evaluate
        rouge_metric = load('rouge')

        # Đánh giá trên tập test (lấy 10 mẫu để test nhanh pipeline)
        test_dataset = dataset_samsum_pt["test"].select(range(10))

        score = self.calculate_metric_on_test_ds(
            test_dataset, rouge_metric, model_pegasus, tokenizer, batch_size=2, column_text='dialogue', column_summary='summary'
        )

        # Lưu kết quả ra file CSV
        rouge_dict = {rn: score[rn] for rn in ['rouge1', 'rouge2', 'rougeL', 'rougeLsum']}
        df = pd.DataFrame(rouge_dict, index=['pegasus'])
        df.to_csv(self.config.metric_name, index=False)