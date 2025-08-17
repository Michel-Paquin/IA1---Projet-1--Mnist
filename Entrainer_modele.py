import tensorflow as tf
def entraine_modele(modele, x_train, y_train, epochs = 25,
    batch_size=128, validation_split=0.2):

    #On entraîne le modèle ne lui donnant le data

    historique = modele.fit(x=x_train, y=y_train, batch_size=batch_size,
                 epochs=epochs, shuffle=True,
                 validation_split=validation_split, verbose=1)


    return historique