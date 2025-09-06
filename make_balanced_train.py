# make_balanced_train.py
import random, shutil
from pathlib import Path

random.seed(42)

# 原始数据根目录（你当前训练用的目录）
SRC = Path("artifacts/data_ingestion/COVID-19_Lung_CT_Scans")
CLS0 = "COVID-19"
CLS1 = "Non-COVID-19"

# 目标目录：仅用于“训练集”的平衡版本（验证集继续用原目录+validation_split）
DST = Path("artifacts/balanced_train")

# 目标比例：把 Non-COVID 过采样（复制+增强可以先只复制）到接近 1:1
# 你可以把目标数量设置为两类里较大的那个，或者略小一点
TARGET_PER_CLASS = 3000   # 举例：两类都凑到 3000（如果原图不够就用能有的最大值）

def clean_dir(p: Path):
    if p.exists():
        shutil.rmtree(p)
    p.mkdir(parents=True, exist_ok=True)

def copy_some(src_dir: Path, dst_dir: Path, n: int):
    files = [f for f in src_dir.glob("*") if f.suffix.lower() in {".png", ".jpg", ".jpeg"}]
    if not files:
        return 0
    if n <= len(files):
        pick = random.sample(files, n)
    else:
        # 若不够，先全复制一遍，再随机补到 n（允许重复文件名时加前缀）
        pick = files.copy()
        extra = random.choices(files, k=n - len(files))
        pick.extend(extra)
    # 拷贝，若重复则加序号前缀
    used = set()
    cnt = 0
    for i, f in enumerate(pick):
        name = f.name
        if name in used:
            name = f"{i}_{name}"
        used.add(name)
        shutil.copy2(f, dst_dir / name)
        cnt += 1
    return cnt

def main():
    # 清空/新建目标目录
    clean_dir(DST / CLS0)
    clean_dir(DST / CLS1)

    n0 = copy_some(SRC / CLS0, DST / CLS0, TARGET_PER_CLASS)
    n1 = copy_some(SRC / CLS1, DST / CLS1, TARGET_PER_CLASS)

    print(f"[balanced_train] {CLS0}: {n0}, {CLS1}: {n1}")
    print("Balanced train prepared at:", DST.resolve())

if __name__ == "__main__":
    main()
