import tensorflow as tf

def creer_modele(pas_apprentissage):

    modele = tf.keras.models.Sequential()
    modele.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
    modele.add(tf.keras.layers.Dense(units=32, activation='relu'))
    modele.add(tf.keras.layers.Dropout(rate=0.2))
    modele.add(tf.keras.layers.Dense(units=10, activation='softmax'))
    modele.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=pas_apprentissage),
        loss="sparse_categorical_crossentropy",
        metrics=['accuracy'])
    return modele
