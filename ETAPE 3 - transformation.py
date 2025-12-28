from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error, r2_score
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

train = pd.read_csv('train.csv')

num = ["OverallQual", "GrLivArea", "GarageCars", "TotalBsmtSF", "YearBuilt", "FullBath"]
cat = ["Neighborhood", "KitchenQual"]

X = train.drop(columns=["SalePrice"])  # toutes les colonnes sauf la cible
y = train["SalePrice"]                 # la cible

X_train, X_test, y_train, y_test = train_test_split(
    X, y,            # X = variables indépendantes, y = variable cible
    test_size=0.2,   # proportion des données pour le test (ici 20%)
    random_state=42  # pour que la séparation soit reproductible
)

y_train_log = np.log1p(y_train) 
y_test_log = np.log1p(y_test)

num_transformer = SimpleImputer(strategy='median')
# num_transformer.fit
# num_transformer.transform
# num_transformer.fit_transform ==> pris en charge dans le pipeline directement.
cat_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(transformers=[("num", num_transformer, num), ("cat", cat_transformer, cat)])

model = Pipeline(steps=[
    ('preprocessor', preprocessor),  # notre ColumnTransformer
    ('regressor', LinearRegression())  # modèle ML
])


#apprentissage des coefficients
model.fit(X_train, y_train_log) 

# Prédiction sur le train ou validation
y_pred_log_test = model.predict(X_test)
y_pred_log_train = model.predict(X_train)   

y_pred_test = np.expm1(y_pred_log_test)  # revenir à l'échelle initiale
y_pred_train = np.expm1(y_pred_log_train)  # revenir à l'échelle initiale


#Histogramme comparatif  log ou non

plt.figure(figsize=(10, 6))
sns.histplot(y_pred_test - y_test, color='blue', label=' test SalePrice', kde=True, stat="density")
sns.histplot(y_pred_train - y_train, color='orange', label='Train SalePrice', kde=True, stat="density")
plt.legend()
plt.title("Distribution : vrai vs prédit (train)")
plt.xlabel("SalePrice")
plt.ylabel("Densité")
plt.show()

# Méthode 3 — Le graphique le plus parlant (scatter)
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred_test, alpha=0.4, color='blue', label='Test')
plt.scatter(y_train, y_pred_train, alpha=0.4, color='orange', label='Train')
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         color='red')

plt.xlabel("Vrai SalePrice")
plt.ylabel("SalePrice prédit")
plt.title("Vrai vs Prédit (train)")
plt.show()

#Boxplot des features numériques
# plt.figure(figsize=(12, 6))
# sns.boxplot(data=X_train[num])
# plt.title("Boxplot des features numériques (Train)")
# plt.show()



###Évaluation du modèle

rmse = root_mean_squared_error(y_test_log, y_pred_log_test)
r2 = r2_score(y_test_log, y_pred_log_test)
rmse_train = root_mean_squared_error(y_train_log, y_pred_log_train)
r2_train = r2_score(y_train_log, y_pred_log_train)
print(f"RMSE test: {rmse}")
print(f"R² test: {r2}")
print(f"RMSE train: {rmse_train}")
print(f"R² train: {r2_train}")
