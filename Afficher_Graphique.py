
# MNIST Programme d'entraînement et de prédictions pour le Projet no 1 du cours AI-1
# Par Samira Lehlou et Michel Paquin cours 420-004-XX Groupe 12504
'''

Ce programme affiche les graphiques d'accuracy et de loss
à partir des données entraînées MNIST.
'''

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
