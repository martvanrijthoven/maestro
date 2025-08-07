import os
import shutil
from typing import Callable, Optional

import lightning
import mlflow
from lightning.pytorch.callbacks import Callback

from maestro.trainer.common.training import MaestroTrainer, TModel, TProcessor


class SaveCheckpoint(Callback):
    def __init__(
        self, 
        result_path: str, 
        save_model_callback: Callable[[str, TProcessor, TModel], None],
        enable_mlflow: bool = False
    ):
        self.result_path = result_path
        self.save_model_callback = save_model_callback
        self.enable_mlflow = enable_mlflow

    def on_train_epoch_end(self, trainer: lightning.Trainer, pl_module: MaestroTrainer):
        checkpoint_path = f"{self.result_path}/latest"
        if os.path.exists(checkpoint_path):
            shutil.rmtree(checkpoint_path)
        self.save_model_callback(checkpoint_path, pl_module.processor, pl_module.model)
        print(f"Saved latest checkpoint to {checkpoint_path}")
        
        # Log checkpoint to MLflow if enabled
        if self.enable_mlflow and mlflow.active_run():
            try:
                # Log with epoch-specific path but also tag as latest
                artifact_path = f"checkpoints/epoch_{pl_module.current_epoch}"
                mlflow.log_artifacts(checkpoint_path, artifact_path)
                
                # Log metadata to identify the latest checkpoint
                mlflow.log_param("latest_checkpoint_epoch", pl_module.current_epoch)
                mlflow.log_param("latest_checkpoint_path", artifact_path)
                
                print(f"Logged checkpoint to MLflow: {artifact_path} (marked as latest)")
            except Exception as e:
                print(f"Warning: Failed to log checkpoint to MLflow: {e}")

        # TODO: Get current metric value from trainer
        # TODO: Compare with best value and save if better
        # TODO: Save best model to {self.result_path}/best if metric improved

    def on_train_end(self, trainer: lightning.Trainer, pl_module: MaestroTrainer):
        pass
