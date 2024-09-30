from detectron2.engine.hooks import HookBase
import logging

class EarlyStoppingHook(HookBase):
    def __init__(self, patience, threshold=0.001):
        self.patience = patience
        self.threshold = threshold
        self.best_validation_loss = None
        self.counter = 0
        self.stop_training = False
        self.best_model_saved = False  # Track whether the best model is already saved
        self.logger = logging.getLogger("detectron2")
        
    def after_step(self):
        storage = self.trainer.storage

        if "validation_loss" not in storage.histories():
            return

        validation_loss = storage.history("validation_loss").latest()

        if self.best_validation_loss is None or validation_loss < self.best_validation_loss - self.threshold:
            self.best_validation_loss = validation_loss
            self.trainer.checkpointer.save("best_val_loss_model")
            self.counter = 0  # Reset patience counter if validation loss improves
        else:
            self.counter += 1  # Increment patience counter if no improvement

        if self.counter >= self.patience:
            self.logger.info(f"Stopping early at iteration {self.trainer.iter} due to no improvement in validation loss.")
            if not self.best_model_saved:
                self.trainer.checkpointer.save(f"model_iteration_{self.trainer.iter}_early_stopped")
            self.stop_training = True

        if self.stop_training:
            self.logger.info(f"Early stopping triggered at epoch {self.trainer.epoch}.")
            exit(0)
