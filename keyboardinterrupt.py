import tensorflow as tf
import threading

class KeyboardInterruptCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.interrupted = False
        threading.Thread(target=self._listen_to_keypress, daemon=True).start()

    def _listen_to_keypress(self):
        while not self.interrupted:
            key = input("Press 'Enter' to stop training...\n")
            if key == "":
                self.interrupted = True

    def on_epoch_end(self, epoch, logs=None):
        if self.interrupted:
            self.model.stop_training = True
            print(f"Training interrupted at epoch {epoch + 1}")