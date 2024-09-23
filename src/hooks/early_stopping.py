from detectron2.engine.hooks import HookBase
import logging

class EarlyStoppingHook(HookBase):
    def __init__(self, patience, threshold=0.001):
        self.patience = patience
        self.threshold = threshold
        self.best_validation_loss = None
        self.counter = 0

    def after_step(self):
        # Access the trainer's storage
        storage = self.trainer.storage

        # Safely access total_loss (always available)
        total_loss = storage.history("total_loss").latest()

        # Check if validation_loss is available
        if "validation_loss" not in storage.histories():
            logging.info("Validation loss not available yet. Skipping early stopping check for this iteration.")
            return

        # Safely retrieve validation_loss
        validation_loss = storage.history("validation_loss").latest()

        # Log the losses
        logging.info(f"Training Loss (total_loss): {total_loss}, Validation Loss: {validation_loss}")

        # Early stopping logic based on validation loss
        if self.best_validation_loss is None or validation_loss < self.best_validation_loss - self.threshold:
            self.best_validation_loss = validation_loss
            self.counter = 0  # Reset patience counter if validation loss improves
        else:
            self.counter += 1  # Increment patience counter if no improvement

        # Early stopping condition: stop if no improvement for `patience` iterations
        if self.counter >= self.patience:
            logging.info(f"Stopping early at iteration {self.trainer.iter} due to no improvement in validation loss.")
            raise StopIteration
