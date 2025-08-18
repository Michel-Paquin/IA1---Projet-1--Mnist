
# MNIST Programme d'entraînement et de prédictions pour le Projet no 1 du cours AI-1
# Par Samira Lehlou et Michel Paquin cours 420-004-XX Groupe 12504
'''
Ce programme extrait une représentation pixélisée d'un chiffre manuscrit dans la base MNIST
et prédit quel est le chiffre réellement.
Le chiffre peut venir du dataset d'entraînement, de test ou de validation  au choix.
Il affiche le graphique du chiffre en question et donne la valeur réelle.
Ensuite il affiche la probabiltié que ce soit chaqun des chiffres de 0 à 9.
'''

# predict_simple.py — autonome
import os
from functools import lru_cache
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split  # pour créer le set de validation

# --- Caches pour éviter les rechargements à toutes les fois que la fonction est appelée ---
@lru_cache(maxsize=1)
def _load_mnist_splits(random_state: int = 42):  #On charge MNIST et on divise en 3 parts train/test/val
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train / 255.0
    x_test  = x_test  / 255.0
    # même split que durant l’entraînement (20% de validation, stratifié)
    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, test_size=0.2, stratify=y_train, random_state=random_state
    )
    return (x_train, y_train), (x_val, y_val), (x_test, y_test)

@lru_cache(maxsize=2)
def _load_model(model_path: str) -> tf.keras.Model:  # On charge le modèle qui a été sauvegardé
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modèle introuvable: {model_path}")
    return tf.keras.models.load_model(model_path)

def _predict_and_plot(x, y, idx: int, model: tf.keras.Model, dataset_name: str, show: bool):
    if idx < 0 or idx >= len(x):
        raise IndexError(f"Index {idx} hors limites (0..{len(x)-1})")

    img = x[idx]
    true_label = int(y[idx])

    # (28,28) -> (1,28,28)
    probs = model.predict(img[np.newaxis, ...], verbose=0)[0] # vecteur contenant la probabilité des chiffres de 0 à 9
    pred_label = int(np.argmax(probs))      # Chiffre prédit
    pred_prob  = float(np.max(probs))       # Probabilité que ce soit ce chiffre

    if show:
        # On affiche sur deux lignes les probabilités de chacun des chiffres de 0 à 9
        probs_line1 = "|".join([f"{d}:{p:.3f}" for d, p in enumerate(probs[:5])])   # Probabilité de 0 è 4
        probs_line2 = "|".join([f"{d}:{p:.3f}" for d, p in enumerate(probs[5:], start=5)]) # probabilité de 5 à 9

        fig, ax = plt.subplots(figsize=(4.8, 4.8))
        ax.imshow(img, cmap="gray")
        ax.set_axis_off()
        # affichage de l'entête
        ax.set_title(f"{dataset_name} #{idx} | Vérité: {true_label}\n"
                     f"Prédiction: {pred_label} (p={pred_prob:.3f})", fontsize=10)
        # affichage des probabilités sous le graphique
        ax.text(0.5, -0.18, probs_line1, ha="center", va="top",
                transform=ax.transAxes, fontsize=8, family="monospace")
        ax.text(0.5, -0.28, probs_line2, ha="center", va="top",
                transform=ax.transAxes, fontsize=8, family="monospace")
        plt.tight_layout()
        plt.show()

    return {
        "true_label": true_label,
        "pred_label": pred_label,
        "pred_prob": pred_prob,
        "probs": probs.tolist(),
    }

def predict_mnist(dataset: str, idx: int, model_path: str, show: bool = True):
    """
    dataset: 'train' | 'val' | 'test'
    idx: index de l'image à prédire
    model_path: chemin du modèle .keras
    """
    model = _load_model(model_path)
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = _load_mnist_splits()

    if dataset == "train":
        x, y, name = x_train, y_train, "Train"
    elif dataset == "val":
        x, y, name = x_val, y_val, "Val"
    elif dataset == "test":
        x, y, name = x_test, y_test, "Test"
    else:
        raise ValueError("dataset doit être 'train', 'val' ou 'test'")

    return _predict_and_plot(x, y, idx, model, name, show)

if __name__ == "__main__":
    # Exemples (les appels ne rechargent plus le modèle/données à chaque fois)
    predict_mnist("train", 1, "tp1.keras", True)
    predict_mnist("test", 6, "tp1.keras", True)
    predict_mnist("val", 3513, "tp1.keras", True)
