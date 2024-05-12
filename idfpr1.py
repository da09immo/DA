# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 23:13:25 2024

@author: Akshit
"""
import os
import pandas as pd 
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from joblib import dump


df_fullMLIdF =pd.read_csv('GouvImmo_full_MLIdF.csv', sep = ',',low_memory=False)



#je distingue les différentes catégories numériques et catégorielles

num_cat = ['surface_reelle_bati', 'nombre_pieces_principales', 'Prix au m²(€)N-1']
cat_cat = ['type_local_x', 'code_postal', 'Dépendance', 'year']



# Attempt to convert numeric columns to floats, handle errors
for column in num_cat:
    try:
        df_fullMLIdF[column] = pd.to_numeric(df_fullMLIdF[column], errors='coerce')
    except Exception as e:
        print(f"Error converting {column} to float: {e}")
        

        

# Préprocesseur qui sera utilisé pour chaque modèle

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_cat),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cat)
    ])

#Modele DEcision Tree Regressor

tree_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', DecisionTreeRegressor(random_state=42))
])

#Modele Lineear Regression

linear_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

#Modele Gradient Boosting REgressor

gradient_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', GradientBoostingRegressor(random_state=42))
])

#Je prépare ma liste de région par effectuer l'analyse

unique_reg = df_fullMLIdF['nom_departement'].unique()

#JE crée une liste pour stocker les résultats

results = []

# Base path for saving models
base_path = "C:/Users/Akshit/DSP/DA/idf"

# J'effectue une boucle sur chaque région

for departement in unique_reg:
    
# JE filtre les données par région
    
    df_departement = df_fullMLIdF[df_fullMLIdF['nom_departement'] == departement]
    
#Machine LEarning, valeur cible: prix au m²

    X = df_departement.drop('Prix au m²(€)', axis=1)
    y = df_departement['Prix au m²(€)']
    
# Train Test Split 
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
# Entraînement et évaluation de chaque modèle
    
    for name, model in [("Decision Tree", tree_pipeline), ("Linear Regression", linear_pipeline), ("Gradient Boosting", gradient_pipeline)]:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred)
        results.append((departement, name, mae, mape))
        model_filename = f"{departement.replace(' ', '_')}_{name.replace(' ', '_')}_model.joblib"
        model_path = os.path.join(base_path, model_filename)
        
        # Save model
        dump(model, model_path)

# Affichage des résultats en DataFrame 

results_df = pd.DataFrame(results, columns=['Departement', 'Model', 'MAE', 'MAPE']).round(2)


# Finalement, on effectue un tri spécificique pour les différents modèles, tri par MAPE croissante:

#Pour le Decision Tree Regressor
results_tree_df_tri = results_df[results_df['Model'] == 'Decision Tree'].sort_values(by='MAPE').reset_index(drop=True)
print("Decision Tree Regressor:")
print(results_tree_df_tri)

# Pour la Linear Regression
results_linear_df_tri = results_df[results_df['Model'] == 'Linear Regression'].sort_values(by='MAPE').reset_index(drop=True)
print("\nLinear Regression:")
print(results_linear_df_tri)

# Pour le Gradient Boosting Regressor
results_gradient_df_tri = results_df[results_df['Model'] == 'Gradient Boosting'].sort_values(by='MAPE').reset_index(drop=True)
print("\nGradient Boosting Regressor:")
print(results_gradient_df_tri)