
# MNIST Programme d'entraînement et de prédictions pour le Projet no 1 du cours AI-1
# Par Samira Lehlou et Michel Paquin cours 420-004-XX Groupe 12504
'''

Compilation du modèle MNIST en ajoutant les couches dans la topologie du modèle
'''

import tensorflow as tf

def creer_modele(pas_apprentissage):

    modele = tf.keras.models.Sequential()  # Les modèles sont séquenciels
    # Les données sont sauvegardées dans un tableau de 28x28 - Il faut les mettre dans un vecteur de 784
    modele.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
    modele.add(tf.keras.layers.Dense(units=32, activation='relu')) # On définit 32 neurones à ce modèle
    # Couche de sortie avec 10 neurones (les chiffres de 0 à 9)
    modele.add(tf.keras.layers.Dense(units=10, activation='softmax'))
    # On désactive 20% des neurones au hasard pour éviter le sur-apprentissage
    modele.add(tf.keras.layers.Dropout(rate=0.2))
    #Compilation du modèle avec les pas d'apprentissage en paramètre sur le modèle d'optimisation 'Adam'
    modele.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=pas_apprentissage),
        loss="sparse_categorical_crossentropy",
        metrics=['accuracy'])
    return modele
