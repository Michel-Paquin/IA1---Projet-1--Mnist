
from Mnist_Split import *
from Créer_modèle import *
from Entrainer_modele import *
from Afficher_Graphique import *

def Generer_modele(nb_iterations=25, pas_apprentissage=.001,modele_compile="tp1.keras"):
    # 1 - Chargement des données de test et division en 3 catégories Train, Validation et Test
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = split_mnist()
    #2 - on crée et compile le modèle
    modèle=creer_modele(pas_apprentissage)

    #3 -on entraîne le modèle
    historique = entraine_modele(modèle, x_train, y_train,nb_iterations,
                batch_size=128, validation_split=0.2)
    #4 - Sauvegarde du modèle
    modèle.save(modele_compile)
    print(f"✔ Modèle sauvegardé: {modele_compile}")

    #5 Afficher les graphiques d'accuracy et loss
    afficher_graphique(historique)


if __name__ == "__main__":
    Generer_modele(25  ,.01 ,"tp1.keras")
