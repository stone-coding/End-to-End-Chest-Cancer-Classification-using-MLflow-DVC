import shutil
import os
from pathlib import Path
from cnnClassifier import logger   # 用你项目里统一的logger

def copy_models():
    src = Path("artifacts/training")
    dest = Path("model")
    os.makedirs(dest, exist_ok=True)

    files_to_copy = [
        "best_model.keras",
        "model.h5",
        "class_indices.json",   #  类别映射文件一起带走
    ]

    for fname in files_to_copy:
        src_file = src / fname
        if src_file.exists():
            shutil.copy(src_file, dest / fname)
            logger.info(f"Copied {src_file} → {dest / fname}")
        else:
            logger.warning(f"File not found, skip: {src_file}")

    logger.info("All models copied to /model directory for deployment")
