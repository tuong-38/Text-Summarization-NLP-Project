import os
import requests


class PredictionPipeline:

  def __init__(self):
    # Đọc token từ biến môi trường HF_TOKEN trên Render
    self.hf_token = os.environ.get("HF_TOKEN", "")

    # Endpoint Router của Hugging Face
    self.api_url = (
        "https://router.huggingface.co/hf-inference/models/google/pegasus-xsum"
    )

  def predict(self, text):
    headers = {
        "Authorization": f"Bearer {self.hf_token}",
        "Content-Type": "application/json",
    }

    payload = {
        "inputs": text,
        "parameters": {
            "max_length": 128,
            "min_length": 30,
            "do_sample": False,
        },
    }

    try:
      response = requests.post(
          self.api_url, headers=headers, json=payload, timeout=30
      )

      if response.status_code == 200:
        result = response.json()
        if isinstance(result, list) and len(result) > 0:
          return result[0].get("summary_text", "")
        return str(result)
      else:
        return (
            f"Lỗi API ({response.status_code}): {response.text}. Vui lòng thử"
            " lại sau giây lát."
        )

    except Exception as e:
      return f"Lỗi kết nối: {str(e)}"