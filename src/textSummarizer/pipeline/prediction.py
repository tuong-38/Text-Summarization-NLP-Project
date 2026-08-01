from textSummarizer.config.configuration import ConfigurationManager
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


class PredictionPipeline:

  def __init__(self):
    self.config = ConfigurationManager().get_model_evaluation_config()

  def predict(self, text):
    # Dùng model/tokenizer từ Hugging Face Hub thay vì đọc folder local artifacts
    model_name = "google/pegasus-samsum"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # 1. Tokenize văn bản
    inputs = tokenizer(
        text,
        max_length=1024,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )

    # 2. Sinh bản tóm tắt
    summary_ids = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        length_penalty=0.8,
        num_beams=8,
        max_length=128,
    )

    # 3. Decode kết quả
    output = tokenizer.decode(
        summary_ids[0],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )

    return output