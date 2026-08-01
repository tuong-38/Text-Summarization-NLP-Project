import os
import uvicorn
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from textSummarizer.pipeline.prediction import PredictionPipeline

app = FastAPI()

# Model nhận dữ liệu cho Swagger API
class ClientInput(BaseModel):
    text: str

# 1. Trang Web UI chính (Giao diện người dùng)
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    html_content = """
    <!DOCTYPE html>
    <html lang="vi">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Text Summarizer - NLP Project</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
        <style>
            body { background-color: #f8f9fa; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
            .header { background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%); color: white; padding: 2rem 0; margin-bottom: 2rem; border-radius: 0 0 15px 15px; }
            .card { border: none; border-radius: 12px; box-shadow: 0 4px 15px rgba(0,0,0,0.05); }
            textarea { border-radius: 8px; resize: vertical; }
            .btn-summarize { background-color: #2a5298; color: white; border: none; padding: 10px 25px; border-radius: 8px; font-weight: 600; transition: all 0.3s; }
            .btn-summarize:hover { background-color: #1e3c72; color: white; transform: translateY(-2px); }
            .output-box { background-color: #eef2f5; min-height: 200px; border-radius: 8px; padding: 15px; white-space: pre-wrap; font-size: 1rem; color: #2d3748; }
        </style>
    </head>
    <body>
        <div class="header text-center">
            <h1>📝 AI Text Summarizer</h1>
            <p class="mb-0">Tóm tắt văn bản & cuộc hội thoại tự động với PEGASUS Transformer Model</p>
        </div>

        <div class="container mb-5">
            <div class="row g-4">
                <!-- Cột trái: Đầu vào -->
                <div class="col-md-6">
                    <div class="card h-100 p-4">
                        <h5 class="card-title text-primary mb-3">📥 Văn bản đầu vào (Dialogue / Text)</h5>
                        <form action="/predict_web" method="post">
                            <div class="mb-3">
                                <textarea class="form-control" name="text" rows="10" placeholder="Nhập hoặc dán đoạn văn bản/hội thoại tiếng Anh vào đây..." required></textarea>
                            </div>
                            <button type="submit" class="btn btn-summarize w-100">🚀 Tóm Tắt Ngay</button>
                        </form>
                    </div>
                </div>

                <!-- Cột phải: Đầu ra -->
                <div class="col-md-6">
                    <div class="card h-100 p-4">
                        <h5 class="card-title text-success mb-3">✨ Kết quả Tóm tắt (Summary)</h5>
                        <div class="output-box">
                            <em>Kết quả tóm tắt sẽ xuất hiện tại đây sau khi bạn nhấn "Tóm Tắt Ngay"...</em>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


# 2. Xử lý yêu cầu tóm tắt từ giao diện Web UI
@app.post("/predict_web", response_class=HTMLResponse)
async def predict_web(text: str = Form(...)):
    try:
        obj = PredictionPipeline()
        summary = obj.predict(text)
        
        # Trả về lại giao diện kèm theo cả Input và Output
        html_content = f"""
        <!DOCTYPE html>
        <html lang="vi">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Text Summarizer - NLP Project</title>
            <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
            <style>
                body {{ background-color: #f8f9fa; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }}
                .header {{ background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%); color: white; padding: 2rem 0; margin-bottom: 2rem; border-radius: 0 0 15px 15px; }}
                .card {{ border: none; border-radius: 12px; box-shadow: 0 4px 15px rgba(0,0,0,0.05); }}
                textarea {{ border-radius: 8px; resize: vertical; }}
                .btn-summarize {{ background-color: #2a5298; color: white; border: none; padding: 10px 25px; border-radius: 8px; font-weight: 600; transition: all 0.3s; }}
                .btn-summarize:hover {{ background-color: #1e3c72; color: white; transform: translateY(-2px); }}
                .output-box {{ background-color: #eef2f5; min-height: 200px; border-radius: 8px; padding: 15px; white-space: pre-wrap; font-size: 1rem; color: #2d3748; border-left: 4px solid #198754; }}
            </style>
        </head>
        <body>
            <div class="header text-center">
                <h1>📝 AI Text Summarizer</h1>
                <p class="mb-0">Tóm tắt văn bản & cuộc hội thoại tự động với PEGASUS Transformer Model</p>
            </div>

            <div class="container mb-5">
                <div class="row g-4">
                    <!-- Cột trái: Giữ nguyên văn bản đã nhập -->
                    <div class="col-md-6">
                        <div class="card h-100 p-4">
                            <h5 class="card-title text-primary mb-3">📥 Văn bản đầu vào (Dialogue / Text)</h5>
                            <form action="/predict_web" method="post">
                                <div class="mb-3">
                                    <textarea class="form-control" name="text" rows="10" required>{text}</textarea>
                                </div>
                                <button type="submit" class="btn btn-summarize w-100">🚀 Tóm Tắt Ngay</button>
                            </form>
                        </div>
                    </div>

                    <!-- Cột phải: Hiển thị kết quả tóm tắt -->
                    <div class="col-md-6">
                        <div class="card h-100 p-4">
                            <h5 class="card-title text-success mb-3">✨ Kết quả Tóm tắt (Summary)</h5>
                            <div class="output-box">{summary}</div>
                        </div>
                    </div>
                </div>
            </div>
        </body>
        </html>
        """
        return HTMLResponse(content=html_content)
    except Exception as e:
        return HTMLResponse(content=f"<h3>Đã xảy ra lỗi: {e}</h3>", status_code=500)


# 3. API POST cho Swagger / Postman
@app.post("/predict")
async def predict_route(data: ClientInput):
    try:
        obj = PredictionPipeline()
        summary = obj.predict(data.text)
        return {"summary": summary}
    except Exception as e:
        raise e

if __name__ == "__main__":
  import os

  # Render sẽ tự động cấp cổng thông qua biến môi trường PORT, nếu không có sẽ dùng 8080
  port = int(os.environ.get("PORT", 8080))
  # Bắt buộc host phải là "0.0.0.0" để Render kết nối được vào Container
  uvicorn.run(app, host="0.0.0.0", port=port)