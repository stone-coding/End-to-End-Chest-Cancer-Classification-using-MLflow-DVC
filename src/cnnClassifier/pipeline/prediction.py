import json
from pathlib import Path
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

class PredictionPipeline:
    def __init__(self, filename):
        self.filename = filename
        
    def _load_model(self):
        # 优先用 model/ 下的权重；找不到再回退到 artifacts/training/
        candidates = [
            Path("model") / "best_model.keras",
            Path("model") / "model.h5",
            Path("artifacts") / "training" / "best_model.keras",
            Path("artifacts") / "training" / "model.h5",
        ]
        for p in candidates:
            if p.exists():
                return load_model(p.as_posix()), p
        raise FileNotFoundError(
            "No model file found. Checked: " + ", ".join([c.as_posix() for c in candidates])
        )

    def _load_mapping(self):
        # 同样优先读 model/ 下的 class_indices.json，找不到再回退
        candidates = [
            Path("model") / "class_indices.json",
            Path("artifacts") / "training" / "class_indices.json",
        ]
        for p in candidates:
            if p.exists():
                ci = json.loads(p.read_text())
                return {v: k for k, v in ci.items()}
        # 兜底
        return {0: "COVID-19", 1: "Non-COVID-19"}

    def _preprocess(self, fname):
        img = image.load_img(fname, target_size=(224, 224))  # 与训练一致
        x = image.img_to_array(img) / 255.0                  # 与训练 rescale=1./255 对齐
        return np.expand_dims(x, axis=0)

    # def predict(self):
    #     model, model_path = self._load_model()
    #     idx_to_label = self._load_mapping()
    #     x = self._preprocess(self.filename)

    #     y = model.predict(x, verbose=0)  # (1,2) softmax 或 (1,1) sigmoid

    #     # 先得到一个默认的预测
    #     if y.shape[-1] == 1:
    #         pred_idx = int(float(y[0][0]) >= 0.5)
    #     else:
    #         pred_idx = int(np.argmax(y, axis=1)[0])
    #         # 如果存在阈值文件，则用阈值对 Non-COVID-19 进行二次判定
    #         thr_path = Path("artifacts/training/threshold.json")
    #         if thr_path.exists():
    #             thr = json.loads(thr_path.read_text()).get("non_covid_threshold", 0.5)
    #             # 找到 Non-COVID-19 的类别索引
    #             non_idxs = [i for i, lbl in idx_to_label.items() if lbl.lower().startswith("non-covid")]
    #             if non_idxs and len(idx_to_label) == 2:
    #                 idx_non = non_idxs[0]
    #                 # 另一类索引（两类时安全）
    #                 idx_other = [i for i in idx_to_label.keys() if i != idx_non][0]
    #                 p_non = float(y[0, idx_non])
    #                 pred_idx = idx_non if p_non >= thr else idx_other

    #     # 现在再根据最终 pred_idx 取标签并格式化展示
    #     label = idx_to_label.get(pred_idx, "Unknown")
    #     display = "COVID-19 infected" if label.lower().startswith("covid") else "Non-COVID-19"

    #     # 调试：看输入分布 & 原始概率
    #     try:
    #         print(f"[infer] model={model_path} output_shape={model.output_shape} "
    #               f"x.mean={x.mean():.4f} x.std={x.std():.4f} "
    #               f"y={y.tolist()} -> idx={pred_idx} label={label}")
    #     except Exception:
    #         pass

    #     # 也把概率一并返回，便于前端/排查（可选）
    #     payload = {"image": display}
    #     if y.shape[-1] == 2:
    #         payload["probs"] = {
    #             "COVID-19": float(y[0, 0]),
    #             "Non-COVID-19": float(y[0, 1]),
    #         }
    #     return [payload]



    def predict(self):
        model, model_path = self._load_model()
        idx_to_label = self._load_mapping()
        x = self._preprocess(self.filename)

        y = model.predict(x, verbose=0)  # (1,2) softmax 或 (1,1) sigmoid

        # —— 只用 argmax / 二分类阈值0.5 —— #
        if y.shape[-1] == 1:
            pred_idx = int(float(y[0, 0]) >= 0.5)
        else:
            pred_idx = int(np.argmax(y, axis=1)[0])

        label = idx_to_label.get(pred_idx, "Unknown")
        display = "COVID-19 infected" if label.lower().startswith("covid") else "Non-COVID-19"

        # 调试信息（可留可去）
        try:
            print(f"[infer] model={model_path} output_shape={model.output_shape} "
                f"x.mean={x.mean():.4f} x.std={x.std():.4f} "
                f"y={y.tolist()} -> idx={pred_idx} label={label}")
        except Exception:
            pass

        payload = {"image": display}
        # 可选：把概率也返回到前端，有助于你观察边界样本
        if y.shape[-1] == 2:
            payload["probs"] = {
                "COVID-19": float(y[0, 0]),
                "Non-COVID-19": float(y[0, 1]),
            }

        return [payload]

