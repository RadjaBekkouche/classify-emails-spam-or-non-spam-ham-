# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 07:04:14 2023

@author: Radja
"""
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Charger le dataset depuis le fichier tp.csv
df = pd.read_csv('C:/Users/Radja/Downloads/dataset_tp1.csv')

# Séparation des données en texte et étiquettes
texts = df['text']
labels = df['label']

# Extraction des Caractéristiques avec CountVectorizer
vectorizer = CountVectorizer(ngram_range=(1, 2))
X = vectorizer.fit_transform(texts)

# Division des données
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

#  Création et Entraînement du Modèle - Naive Bayes Multinomial
model = MultinomialNB()  # Choisir le modèle approprié

#Optimisation des hyperparamètres avec Grid Search
param_grid = {'alpha': [0.1, 0.5, 1.0, 2.0, 5.0]}
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

best_alpha = grid_search.best_params_['alpha']
print(f"Meilleur hyperparamètre pour Naive Bayes Multinomial: alpha = {best_alpha}")

# Utiliser le meilleur modèle trouvé
model = grid_search.best_estimator_

# Prédictions et Évaluation
predictions = model.predict(X_test)

# Affichage des Résultats
accuracy = accuracy_score(y_test, predictions)
confusion_mat = confusion_matrix(y_test, predictions)
classification_rep = classification_report(y_test, predictions)

print("\nPerformance du modèle Naive Bayes Multinomial:")
print("Précision du modèle:", accuracy)
print("Matrice de confusion:\n", confusion_mat)
print("Rapport de classification:\n", classification_rep)

# Section 6: Création et Entraînement du Modèle - SVM
svm_model = SVC(kernel='linear')  # Choisir le modèle approprié

# Section 9: Optimisation des hyperparamètres avec Grid Search pour SVM
param_grid_svm = {'C': [0.1, 1, 10, 100]}
grid_search_svm = GridSearchCV(svm_model, param_grid_svm, cv=5, scoring='accuracy')
grid_search_svm.fit(X_train, y_train)

best_C_svm = grid_search_svm.best_params_['C']
print(f"Meilleur hyperparamètre pour SVM: C = {best_C_svm}")

# Utiliser le meilleur modèle SVM trouvé
svm_model = grid_search_svm.best_estimator_

# Section 7: Prédictions et Évaluation pour SVM
predictions_svm = svm_model.predict(X_test)

# Section 8: Affichage des Résultats pour SVM
accuracy_svm = accuracy_score(y_test, predictions_svm)
confusion_mat_svm = confusion_matrix(y_test, predictions_svm)
classification_rep_svm = classification_report(y_test, predictions_svm)

print("\nPerformance du modèle SVM:")
print("Précision du modèle:", accuracy_svm)
print("Matrice de confusion:\n", confusion_mat_svm)
print("Rapport de classification:\n", classification_rep_svm)
