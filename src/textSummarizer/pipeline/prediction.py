from textSummarizer.config.configuration import ConfigurationManager
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class PredictionPipeline:
    def __init__(self):
        self.config = ConfigurationManager().get_model_evaluation_config()

    def predict(self, text):
        tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_path)

        # 1. Tokenize văn bản đầu vào
        inputs = tokenizer(text, max_length=1024, truncation=True, padding="max_length", return_tensors="pt")

        # 2. Sinh bản tóm tắt bằng model PEGASUS
        summary_ids = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            length_penalty=0.8,
            num_beams=8,
            max_length=128
        )

        # 3. Decode mã token thành chuỗi văn bản hoàn chỉnh
        output = tokenizer.decode(summary_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
        
        print("\nDialogue:")
        print(text)
        print("\nModel Summary:")
        print(output)

        return output