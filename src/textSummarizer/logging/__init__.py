import os
import sys
import logging

# Định dạng hiển thị nhật ký log
logging_str = "[%(asctime)s: %(levelname)s: %(module)s: %(message)s]"

# Thư mục và file lưu log
log_dir = "logs"
log_filepath = os.path.join(log_dir, "running_logs.log")
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format=logging_str,
    handlers=[
        logging.FileHandler(log_filepath), # Ghi log vào file logs/running_logs.log
        logging.StreamHandler(sys.stdout)  # In log ra màn hình Terminal
    ]
)

# Khởi tạo đối tượng logger
logger = logging.getLogger("textSummarizerLogger")