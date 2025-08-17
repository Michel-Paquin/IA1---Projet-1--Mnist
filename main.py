
# MNIST Programme d'entraînement et de prédictions pour le Projet no 1 du cours AI-1
# Par Samira Lehlou et Michel Paquin cours 420-004-XX Groupe 12504
'''
'''
from Génération_du_modèle import *
from Prédiction_Entrainement import *

modele_compile="tp1.keras"
print("Projet Mnist")
print()
print("1- Génération du Modèle")
print("2- Prédictions")
print('Tout autre caractère met fin au programme')
option ='9'
while option == '9':
    option = input("Entrez 1 ou 2 et faites <enter>: ")
    if option == '1':
        Generer_modele(25, .001, modele_compile)
    elif option == '2':
        if os.path.exists(modele_compile):
            predict_mnist("train", 1, modele_compile, True)
            predict_mnist("train", 6, modele_compile, True)
            predict_mnist("train", 3513, modele_compile, True)
            predict_mnist("train", 10123, modele_compile, True)
            predict_mnist("train", 43213, modele_compile, True)
        else:
            print("Le modèle n'a pas été généré - roulez l'option 1 d'abord")
            option = '9'
