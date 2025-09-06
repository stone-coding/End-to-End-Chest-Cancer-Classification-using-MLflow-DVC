import os
import urllib.request as request
from zipfile import ZipFile
from sklearn.utils import compute_class_weight
import tensorflow as tf
import time
import json
import numpy as np
from collections import Counter
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau


from cnnClassifier.entity.config_entity import TrainingConfig
from pathlib import Path
import tensorflow as tf

import pandas as pd

class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config

    
    def get_base_model(self):
        self.model = tf.keras.models.load_model(
            self.config.updated_base_model_path
        )

    def train_valid_generator(self):
        from pathlib import Path
        import tensorflow as tf

        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            interpolation="bilinear"
        )

        # 预先切好的“平衡训练目录”
        balanced_train = Path("artifacts/balanced_train")
        use_balanced = balanced_train.exists() and any(balanced_train.iterdir())

        if use_balanced:
            # 训练：用平衡目录（不使用 validation_split）
            train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
                rescale=1./255,
                rotation_range=40 if self.config.params_is_augmentation else 0,
                horizontal_flip=bool(self.config.params_is_augmentation),
                width_shift_range=0.2 if self.config.params_is_augmentation else 0.0,
                height_shift_range=0.2 if self.config.params_is_augmentation else 0.0,
                shear_range=0.2 if self.config.params_is_augmentation else 0.0,
                zoom_range=0.2 if self.config.params_is_augmentation else 0.0,
            )
            self.train_generator = train_datagen.flow_from_directory(
                directory=balanced_train.as_posix(),
                shuffle=True,
                **dataflow_kwargs
            )

            # 验证：继续用原目录 + validation_split（保证验证分布保持原始真实分布）
            datagenerator_kwargs = dict(rescale=1./255, validation_split=0.20)
            valid_datagen = tf.keras.preprocessing.image.ImageDataGenerator(**datagenerator_kwargs)
            self.valid_generator = valid_datagen.flow_from_directory(
                directory=self.config.training_data,
                subset="validation",
                shuffle=False,
                **dataflow_kwargs
            )
        else:
            # 回退到原有方案：同一目录 + validation_split
            datagenerator_kwargs = dict(rescale=1./255, validation_split=0.20)

            valid_datagen = tf.keras.preprocessing.image.ImageDataGenerator(**datagenerator_kwargs)
            self.valid_generator = valid_datagen.flow_from_directory(
                directory=self.config.training_data,
                subset="validation",
                shuffle=False,
                **dataflow_kwargs
            )

            if self.config.params_is_augmentation:
                train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
                    rotation_range=40,
                    horizontal_flip=True,
                    width_shift_range=0.2,
                    height_shift_range=0.2,
                    shear_range=0.2,
                    zoom_range=0.2,
                    **datagenerator_kwargs
                )
            else:
                train_datagen = valid_datagen

            self.train_generator = train_datagen.flow_from_directory(
                directory=self.config.training_data,
                subset="training",
                shuffle=True,
                **dataflow_kwargs
            )



    # def train_valid_generator(self):

    #     datagenerator_kwargs = dict(
    #         rescale = 1./255,
    #         validation_split=0.20
    #     )

    #     dataflow_kwargs = dict(
    #         target_size=self.config.params_image_size[:-1],
    #         batch_size=self.config.params_batch_size,
    #         interpolation="bilinear"
    #     )

    #     valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
    #         **datagenerator_kwargs
    #     )

    #     self.valid_generator = valid_datagenerator.flow_from_directory(
    #         directory=self.config.training_data,
    #         subset="validation",
    #         shuffle=False,
    #         **dataflow_kwargs
    #     )

    #     if self.config.params_is_augmentation:
    #         train_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
    #             rotation_range=40,
    #             horizontal_flip=True,
    #             width_shift_range=0.2,
    #             height_shift_range=0.2,
    #             shear_range=0.2,
    #             zoom_range=0.2,
    #             **datagenerator_kwargs
    #         )
    #     else:
    #         train_datagenerator = valid_datagenerator

    #     self.train_generator = train_datagenerator.flow_from_directory(
    #         directory=self.config.training_data,
    #         subset="training",
    #         shuffle=True,
    #         **dataflow_kwargs
    #     )

    
    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)

    

    def train(self):
        # 1) steps
        self.steps_per_epoch   = self.train_generator.samples // self.train_generator.batch_size
        self.validation_steps  = self.valid_generator.samples  // self.valid_generator.batch_size

        # 2) 路径
        out_dir   = Path(self.config.root_dir)                  # artifacts/training
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_path  = out_dir / "training_log.csv"
        best_path = out_dir / "best_model.keras"

        # 3) 回调：监控小类召回（recall_non）
        csv_logger = CSVLogger(csv_path.as_posix(), append=True)
        checkpoint = ModelCheckpoint(
            best_path.as_posix(),
            monitor="val_recall_non",     # 小类 Non-COVID 的召回
            mode="max",
            save_best_only=True,
            verbose=1
        )
        earlystop  = EarlyStopping(
            monitor="val_recall_non",
            mode="max",
            patience=5,
            restore_best_weights=True
        )

        # 4) 类权重（解决 8:1 失衡）
        classes = np.unique(self.train_generator.classes)
        weights = compute_class_weight(class_weight="balanced", classes=classes, y=self.train_generator.classes)
        class_weight = {int(c): float(w) for c, w in zip(classes, weights)}
        print(">> class_weight:", class_weight)
        reduce = ReduceLROnPlateau(monitor="val_recall_non", mode="max",
                           factor=0.5, patience=2, min_lr=1e-6, verbose=1)

        # 5) 训练
        history = self.model.fit(
            self.train_generator,
            epochs=self.config.params_epochs,
            steps_per_epoch=self.steps_per_epoch,
            validation_steps=self.validation_steps,
            validation_data=self.valid_generator,
            callbacks=[csv_logger, checkpoint, earlystop, reduce],  
            class_weight=class_weight
        )

        # 6) 统计样本数
        print(">> train class_indices:", self.train_generator.class_indices)
        print(">> valid class_indices:", self.valid_generator.class_indices)
        from collections import Counter
        train_counts = Counter(self.train_generator.classes)
        valid_counts = Counter(self.valid_generator.classes)
        print(">> train counts:", dict(train_counts))
        print(">> valid counts:", dict(valid_counts))

        # 7) 用“最优权重”评估验证集（而不是内存中的最后一轮）
        if best_path.exists():
            eval_model = tf.keras.models.load_model(best_path.as_posix())
            print(">> evaluating with BEST model:", best_path.resolve())
        else:
            eval_model = self.model
            print(">> best model not found, evaluating with current in-memory model")

        y_prob = eval_model.predict(self.valid_generator, verbose=0)   # (N, 2) softmax
        y_pred = y_prob.argmax(axis=1)
        y_true = self.valid_generator.classes

        # 8) 混淆矩阵 + 报告
        from sklearn.metrics import confusion_matrix, classification_report, f1_score
        print(">> confusion_matrix:\n", confusion_matrix(y_true, y_pred))
        target_names = [k for k, _ in sorted(self.valid_generator.class_indices.items(), key=lambda x: x[1])]
        print(">> report:\n", classification_report(y_true, y_pred, target_names=target_names))

        # 9) 在验证集上为 Non-COVID-19 自动调阈值（推理可用）
        # class_indices: {'COVID-19':0, 'Non-COVID-19':1}
        idx_non   = self.valid_generator.class_indices.get('Non-COVID-19', 1)
        probs_non = y_prob[:, idx_non]
        ths = np.linspace(0.10, 0.90, 81)
        f1s = [f1_score((y_true == idx_non).astype(int), (probs_non >= t).astype(int), pos_label=1) for t in ths]
        best_t = float(ths[int(np.argmax(f1s))])
        thr_file = out_dir / "threshold.json"
        thr_file.write_text(json.dumps({"non_covid_threshold": best_t}, indent=2))
        print(">> tuned threshold for Non-COVID-19:", best_t)

        # 10) 保存类别映射（推理端读取）
        ci_path = out_dir / "class_indices.json"
        ci_path.write_text(json.dumps(self.train_generator.class_indices, indent=2))

        # 11) 打印模型信息与保存最终模型
        print(">> output_shape:", self.model.output_shape)
        print(">> saved final to:", Path(self.config.trained_model_path).resolve())
        print(">> best checkpoint should be at:", best_path.resolve())

        final_path = Path(self.config.trained_model_path)       # artifacts/training/model.h5
        final_path.parent.mkdir(parents=True, exist_ok=True)
        self.save_model(path=final_path, model=self.model)

        # 12) 保存训练曲线
        pd.DataFrame(history.history).to_csv(out_dir / "history.csv", index=False)

        # 13) 路径回显
        print("Saved final model to:", final_path.resolve())
        print("Saved best model to :", best_path.resolve())
        print("Saved log to        :", csv_path.resolve())

        return history




    


    # def train(self):
    #     # Calculate steps per epoch and validation steps
    #     self.steps_per_epoch = self.train_generator.samples // self.train_generator.batch_size
    #     self.validation_steps = self.valid_generator.samples // self.valid_generator.batch_size

    #     # === Ensure output directory and file paths ===
    #     out_dir = Path(self.config.root_dir)                # e.g., artifacts/training
    #     out_dir.mkdir(parents=True, exist_ok=True)

    #     csv_path  = out_dir / "training_log.csv"            # log file inside artifacts/training
    #     best_path = out_dir / "best_model.keras"            # best model checkpoint (can also be .h5)

    #     # Setup callbacks: CSV logger, checkpoint saver, early stopping
    #     csv_logger = CSVLogger(csv_path.as_posix(), append=True)
    #     # checkpoint = ModelCheckpoint(
    #     #     best_path.as_posix(),
    #     #     monitor="val_accuracy",
    #     #     save_best_only=True,
    #     #     verbose=1
    #     # )
    #     # earlystop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

    #     checkpoint = ModelCheckpoint(
    #         best_path.as_posix(), 
    #         monitor="val_recall_non", 
    #         mode="max", 
    #         save_best_only=True, 
    #         verbose=1)
    #     earlystop  = EarlyStopping(monitor="val_recall_non", mode="max", patience=5, restore_best_weights=True)


    #     # Train the model
    #     classes = np.unique(self.train_generator.classes)
    #     weights = compute_class_weight(
    #         class_weight='balanced',
    #         classes=classes,
    #         y=self.train_generator.classes
    #     )
    #     class_weight = {int(c): float(w) for c, w in zip(classes, weights)}
    #     print(">> class_weight:", class_weight)

    #     history = self.model.fit(
    #         self.train_generator,
    #         epochs=self.config.params_epochs,
    #         steps_per_epoch=self.steps_per_epoch,
    #         validation_steps=self.validation_steps,
    #         validation_data=self.valid_generator,
    #         callbacks=[csv_logger, checkpoint, earlystop],
    #         class_weight=class_weight      # 加上这个
    #     )
    #     # history = self.model.fit(
    #     #     self.train_generator,
    #     #     epochs=self.config.params_epochs,
    #     #     steps_per_epoch=self.steps_per_epoch,
    #     #     validation_steps=self.validation_steps,
    #     #     validation_data=self.valid_generator,
    #     #     callbacks=[csv_logger, checkpoint, earlystop],
    #     # )

    #     # ====== 统计各类样本数量 ======
    #     print(">> train class_indices:", self.train_generator.class_indices)
    #     print(">> valid class_indices:", self.valid_generator.class_indices)

    #     train_counts = Counter(self.train_generator.classes)
    #     valid_counts = Counter(self.valid_generator.classes)
    #     print(">> train counts:", dict(train_counts))   # e.g. {0: 6000, 1: 700}
    #     print(">> valid counts:", dict(valid_counts))

    #     # ====== 验证集混淆矩阵 ======
    #     from sklearn.metrics import confusion_matrix, classification_report
    #     # 预测验证集
    #     y_prob = self.model.predict(self.valid_generator, verbose=0)
    #     y_pred = y_prob.argmax(axis=1)
    #     y_true = self.valid_generator.classes

    #     print(">> confusion_matrix:\n", confusion_matrix(y_true, y_pred))
    #     print(">> report:\n", classification_report(y_true, y_pred, target_names=[k for k,_ in sorted(self.valid_generator.class_indices.items(), key=lambda x:x[1])]))




    #     print(">> output_shape:", self.model.output_shape)
    #     print(">> train class_indices:", self.train_generator.class_indices)

    #     ci_path = Path("artifacts/training/class_indices.json")
    #     ci_path.parent.mkdir(parents=True, exist_ok=True)
    #     ci_path.write_text(json.dumps(self.train_generator.class_indices, indent=2))

    #     # 保存一份最优模型路径也打印下
    #     print(">> saved final to:", Path(self.config.trained_model_path).resolve())
    #     print(">> best checkpoint should be at:", (Path(self.config.root_dir) / "best_model.keras").resolve())



    #     # === Save the final model to the path specified in config.yaml ===
    #     final_path = Path(self.config.trained_model_path)   # e.g., artifacts/training/model.h5
    #     final_path.parent.mkdir(parents=True, exist_ok=True)
    #     self.save_model(path=final_path, model=self.model)

    #     # Optionally save training history to a CSV for plotting
    #     pd.DataFrame(history.history).to_csv(out_dir / "history.csv", index=False)

    #     # Debug: print out actual save paths
    #     print("Saved final model to:", final_path.resolve())
    #     print("Saved best model to :", best_path.resolve())
    #     print("Saved log to        :", csv_path.resolve())

    #     return history




