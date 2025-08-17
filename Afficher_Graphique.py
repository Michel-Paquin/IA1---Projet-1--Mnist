import matplotlib.pyplot as plt
import tensorflow as tf

def afficher_graphique(historique: tf.keras.callbacks.History):
    h = historique.history
    epochs = range(1, len(h.get("loss", [])) + 1)

    # Accuracy
    if "accuracy" in h:
        plt.figure(figsize=(7,4))
        plt.plot(epochs, h["accuracy"], label="train")
        if "val_accuracy" in h:
            plt.plot(epochs, h["val_accuracy"], label="val")
        plt.xlabel("Époque"); plt.ylabel("Accuracy")
        plt.title("Accuracy par époque")
        plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()

    # Loss
    if "loss" in h:
        plt.figure(figsize=(7,4))
        plt.plot(epochs, h["loss"], label="train")
        if "val_loss" in h:
            plt.plot(epochs, h["val_loss"], label="val")
        plt.xlabel("Époque"); plt.ylabel("Loss")
        plt.title("Loss par époque")
        plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()
