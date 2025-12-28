import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns   

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Voir les 5 premières lignes
print("en 1 ", train.head())

# Nombre de lignes et colonnes
print("en 2 ",train.shape)

# Liste des colonnes et types
print("en 3 ",train.dtypes)

# Statistiques descriptives pour les numériques
print("en 4 ",train.describe())

# Nombre de NaN par colonne
print("en 5 ", train.isnull().sum().sort_values(ascending=False))

# Numériques
num_features = train.select_dtypes(include=['int64', 'float64']).columns
print("Numériques :", num_features)

# Catégorielles
cat_features = train.select_dtypes(include=['object']).columns
print("Catégorielles :", cat_features)

# Aperçu statistique
print(train['SalePrice'].describe())

# Histogramme
sns.histplot(train['SalePrice'], kde=True)

plt.show()

# Boxplot pour voir les outliers
sns.boxplot(x=train['SalePrice'])
plt.show()
