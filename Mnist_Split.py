
# MNIST Programme d'entraînement et de prédictions pour le Projet no 1 du cours AI-1
# Par Samira Lehlou et Michel Paquin cours 420-004-XX Groupe 12504
'''

Cette fonction divise le dataset MNIST en 3 parties , Train, Test et Validation
'''

import tensorflow as tf
from sklearn.model_selection import train_test_split

def split_mnist():
    mnist = tf.keras.datasets.mnist
    # Charger MNIST (une seule fois)
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Normalisation
    x_train = x_train / 255.0
    x_test  = x_test  / 255.0

    # (Optionnel) split train/validation

    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, test_size=0.2, stratify=y_train, random_state=42
    )

    print("Train:", x_train.shape, "| Val:", x_val.shape, "| Test:", x_test.shape)


    # Retourner les datasets
    return (x_train, y_train), (x_val, y_val), (x_test, y_test)

if __name__ == "__main__":
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = split_mnist()
