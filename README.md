🎓 ÉTAPE 1 — COMPRÉHENSION DU PROBLÈME

(C’est une étape conceptuelle, pas technique)

À ce stade, on cherche à répondre clairement à ces questions :

1️⃣ Quel est le but exact du problème ?
2️⃣ Quelle est la variable cible ?
3️⃣ Quel est le type de problème ML ?
4️⃣ Que représentent les données (contexte réel) ?
5️⃣ Qu’est-ce qui est autorisé / interdit par la consigne Kaggle ?

---------------------REPONSES---------------------------------
1️⃣ L'objectif est de deviner le prix de biens immobiliers grâce à de nombreuses variables
2️⃣ la variable cible est "SalePrice" qui est donc le prix du bien
3️⃣C'est un problème de regression
            🧠 Les principaux types de problèmes ML
            1️⃣ Régression

            But : prédire une valeur continue.

            Exemple concret : prédire le prix d’une maison (House Prices), la température demain, le salaire d’un employé.

            Variable cible : un nombre réel (float ou int).

            Exemples d’algos : Linear Regression, Random Forest Regressor, Gradient Boosting.

            Astuce prof : si la variable cible peut prendre n’importe quelle valeur numérique → c’est de la régression.

            2️⃣ Classification

            But : prédire une catégorie ou une classe.

            Exemple concret : spam ou non spam (emails), churn client (oui/non), type d’animal (chat, chien, lapin).

            Variable cible : étiquettes/discrètes.

            Exemples d’algos : Logistic Regression, Random Forest Classifier, SVM, KNN.

            Astuce prof : si ton output est une classe ou un label → c’est de la classification.

            3️⃣ Clustering (ou regroupement)

            But : regrouper des objets similaires sans label.

            Exemple concret : segmenter les clients en groupes pour marketing, regrouper des articles similaires.

            Variable cible : aucune (non supervisé).

            Exemples d’algos : K-Means, DBSCAN, Hierarchical Clustering.

            Astuce prof : tu n’as pas de variable cible → c’est du non supervisé.

            4️⃣ Séries temporelles (Time Series Forecasting)

            But : prédire une valeur future basée sur le passé.

            Exemple concret : prix de l’action demain, consommation d’électricité, trafic web.

            Variable cible : continue, mais dépend du temps.

            Exemples d’algos : ARIMA, Prophet, LSTM.

            Astuce prof : si tes données sont indexées par le temps, il faut traiter la saison, tendance et autocorrélation.

            5️⃣ Réduction de dimension / Feature extraction

            But : simplifier les données tout en conservant l’information.

            Exemple concret : PCA pour visualiser des données en 2D, ou extraire des composants principaux.

            Variable cible : aucune (souvent non supervisé).
4️⃣ Toutes les informations concernant les biens
5️⃣Rien n'est interdit en gros






🧭 ÉTAPE 2 — EXPLORATION DES DONNÉES (EDA)

L’objectif ici est de comprendre tes données concrètement, avant tout nettoyage ou modélisation.
Reste attentif : pas de sélection de features ni de modélisation à cette étape.

1️⃣ Charger les fichiers

Tu as à disposition :

train.csv → données d’entraînement + cible

test.csv → données sans la cible

sample_submission.csv → format de soumission

data_description.txt → description des variables

À faire :

Charger train.csv et test.csv dans ton notebook.

Vérifier la taille des datasets : combien de lignes et colonnes.

Vérifier le type de chaque variable : numérique, catégorielle, date, texte, etc.

2️⃣ Comprendre les variables

À partir de data_description.txt :

Lire la signification de chaque colonne

Identifier :

Variables numériques

Variables catégorielles

Variables ordinales (qualité, condition…)

Astuce prof : prends des notes sur papier ou un document, ça t’aidera pour l’étape suivante.

3️⃣ Aperçu rapide des données

Pour chaque colonne du train :

Voir les valeurs manquantes

Voir quelques exemples de valeurs (head, tail)

Visualiser rapidement les distributions (histogrammes ou counts)

Objectif : détecter tout problème évident (NaN, valeurs aberrantes, catégories étranges).

4️⃣ Vérifier la variable cible (SalePrice)

Distribution générale : histogramme, boxplot

Détecter si elle est très asymétrique → pourrait nécessiter transformation plus tard

5️⃣ Comparer train et test

Vérifier que les colonnes sont identiques

Vérifier les types et catégories : certaines catégories peuvent exister dans test mais pas dans train
