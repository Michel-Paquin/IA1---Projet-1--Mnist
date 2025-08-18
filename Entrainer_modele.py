
# MNIST Programme d'entraînement et de prédictions pour le Projet no 1 du cours AI-1
# Par Samira Lehlou et Michel Paquin cours 420-004-XX Groupe 12504
'''
Entraînement du modèle MNIST .  le nombre d'itération (epoch) est 25 par défaut.
La fonction retourne les poids dans l'historique
'''


import tensorflow as tf
def entraine_modele(modele, x_train, y_train, epochs = 25,
    batch_size=128, validation_split=0.2):

    #On entraîne le modèle ne lui donnant le data

    historique = modele.fit(x=x_train, y=y_train, batch_size=batch_size,
                 epochs=epochs, shuffle=True,
                 validation_split=validation_split, verbose=1)


    return historique