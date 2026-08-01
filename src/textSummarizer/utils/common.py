import os
from box.exceptions import BoxValueError
import yaml
from textSummarizer.logging import logger
from ensure import ensure_annotations
from box import ConfigBox
from pathlib import Path
from typing import Any

@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """Đọc file YAML và trả về ConfigBox (cho phép truy cập key dạng dot notation: config.key)

    Args:
        path_to_yaml (Path): Đường dẫn tới file yaml

    Raises:
        ValueError: Nếu file yaml rỗng
        e: Lỗi phát sinh khác

    Returns:
        ConfigBox: Dữ liệu từ file yaml dưới dạng ConfigBox
    """
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"yaml file: {path_to_yaml} loaded successfully")
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError("yaml file is empty")
    except Exception as e:
        raise e


@ensure_annotations
def create_directories(path_to_directories: list, verbose=True):
    """Tạo danh sách các thư mục

    Args:
        path_to_directories (list): Danh sách đường dẫn các thư mục cần tạo
        ignore_log (bool, optional): Bỏ qua log thông báo tạo thư mục. Mặc định là False.
    """
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"created directory at: {path}")


@ensure_annotations
def get_size(path: Path) -> str:
    """Lấy dung lượng file tính theo KB

    Args:
        path (Path): Đường dẫn tới file

    Returns:
        str: Dung lượng file dạng chuỗi (KB)
    """
    size_in_kb = round(os.path.getsize(path) / 1024)
    return f"~ {size_in_kb} KB"