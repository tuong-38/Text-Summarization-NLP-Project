import requests


class PredictionPipeline:

  def __init__(self):
    # Dùng Hugging Face Inference API cho model PEGASUS
    self.api_url = (
        "https://api-inference.huggingface.co/models/google/pegasus-xsum"
    )

  def predict(self, text):
    payload = {
        "inputs": text,
        "parameters": {
            "max_length": 128,
            "min_length": 30,
            "do_sample": False,
        },
    }

    response = requests.post(self.api_url, json=payload)

    if response.status_code == 200:
      result = response.json()
      if isinstance(result, list) and len(result) > 0:
        return result[0].get("summary_text", "")
      return str(result)
    else:
      # Trường hợp model đang khởi động ở phía server Hugging Face
      return f"API Error ({response.status_code}): Vui lòng thử lại sau vài giây."