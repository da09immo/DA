
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import folium_static
import os
from joblib import load
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

st.set_page_config(
        page_title="Étude de l'immobilier en France",
        page_icon="🏠",  # Replace this with the path to your icon file
        layout="wide"
    )

st.title("ETUDE DU MARCHE DE L’IMMOBILIER EN FRANCE")
st.sidebar.title("🏠 Etude Immobilier")
pages=["Le Projet", "Le Jeu de données", "DataVizualization", "Modélisation","Conclusion & Perspective"]
page=st.sidebar.radio("Aller vers", pages)

col = st.columns((4.5, 2), gap='medium')

with col[0]:
 if page == pages[0] : 
  st.write("### Introduction Du Projet")
  st.write("""
 <div style="text-align: justify;">
  <p>  Ce projet consiste à analyser le marché immobilier français afin de comprendre ses tendances, anticiper les évolutions futures et conseiller les parties prenantes. Il inclut la collecte, le nettoyage et l'analyse des données de 2018 à 2023, en tenant compte de facteurs tels que les politiques économiques, les taux d'intérêt, la démographie et les avancées technologiques. Sur le plan scientifique, il applique des méthodes statistiques et d'apprentissage automatique pour analyser les tendances, prédire les développements futurs et fournir des insights pour la prise de décision.

    Dans le prétraitement des données, une normalisation et le traitement des valeurs manquantes et aberrantes sont effectués. Le feature engineering permettra de créer de nouvelles variables pertinentes pour l'analyse, comme le prix au mètre carré et la catégorisation des DPE en fonction de l'évolution des prix des biens par régions. Des outils de visualisation tels que Matplotlib, Seaborn et Power BI seront utilisés pour présenter l'analyse et représenter les tendances et les relations dans les données. Des analyses statistiques aideront à déterminer les facteurs influençant le marché immobilier pour une meilleure prise de décisions.

    </p>
 </div>
    """, unsafe_allow_html=True)
  st.image("immobilier.jpg")
  
 if page == pages[1] : 
    df=pd.read_csv("GouvImmo_full_MLIdF.csv",sep=',',low_memory=False)
    st.write("### Exploration des données")
    
    st.dataframe(df.head())
    
    st.write("Dimensions du dataframe :")
    
    st.write(df.shape)
    
    if st.checkbox("Afficher les valeurs manquantes") : 
        st.dataframe(df.isna().sum())
        
    if st.checkbox("Afficher les doublons") : 
        st.write(df.duplicated().sum())
    
 if page == pages[2] : 
    @st.cache_data
    def charger_donnees():
      df = pd.read_csv('GouvImmo_full_MLIdF.csv')
      df['nombre_pieces_principales'] = pd.to_numeric(df['nombre_pieces_principales'], errors='coerce')
      df['surface_reelle_bati'] = pd.to_numeric(df['surface_reelle_bati'], errors='coerce')  # Ensure it is numeric
      df['valeur_fonciere'] = pd.to_numeric(df['valeur_fonciere'], errors='coerce')  # Ensure it is numeric
      # Create 'Prix au m²' safely
      df['Prix au m²'] = df.apply(lambda row: row['valeur_fonciere'] / row['surface_reelle_bati'] if row['surface_reelle_bati'] > 0 else np.nan, axis=1)
      return df
    
    df = charger_donnees()

    df_filtered = df[(df['nombre_pieces_principales'] != 0) & (df['nombre_pieces_principales'] <= 15)]

    # Display the charts based on user selection
    # Select Chart Type
    chart_type = st.sidebar.radio("Choisir le type de graphique:", (
        'Distribution de "type_local_x"',
        'Distribution des Pieces Principales',
        'Surface Distribution par Type',
        'Distribution du Prix au m²',
        'Évolution de la valeur des biens',
        'Carte des Transactions Immobilières'
                                
     ))

    if chart_type == 'Distribution de "type_local_x"':
      fig, ax = plt.subplots()
      sns.countplot(x='type_local_x', data=df_filtered, palette='deep', ax=ax)
      st.pyplot(fig)

    elif chart_type == 'Distribution des Pieces Principales':
      fig, ax = plt.subplots()
      sns.countplot(x='nombre_pieces_principales', data=df_filtered, palette='deep', ax=ax)
      st.pyplot(fig)

    elif chart_type == 'Surface Distribution par Type':
      fig = plt.figure()
      sns.boxplot(x='type_local_x', y='surface_reelle_bati', data=df_filtered)
      plt.xticks(rotation=45)
      st.pyplot(fig)

    
    elif chart_type == 'Distribution du Prix au m²':
      df_filtered['Prix au m2'] = df_filtered['valeur_fonciere'] / df_filtered['surface_reelle_bati']
      fig, ax = plt.subplots()
      sns.distplot(df_filtered['Prix au m2'].dropna(), bins=50, kde=False)
      st.pyplot(fig)
      
    elif chart_type == 'Évolution de la valeur des biens':
       # Convert 'date_mutation' to datetime format
       df['date_mutation'] = pd.to_datetime(df['date_mutation'])
       
       # Extract year from 'date_mutation' and create a new column
       df['year'] = df['date_mutation'].dt.year
       
       # Group by year and calculate mean and median price per m²
       price_per_sqm_yearly = df.groupby('year')['Prix au m²(€)'].agg(['mean', 'median']).reset_index()
       
       # Create a line plot using Plotly
       fig2 = px.line(price_per_sqm_yearly, x='year', y=['mean', 'median'],
                     labels={'value':'Price per m² (€)', 'variable':'Statistic'},
                     title='Trends in Property Values (Price per m²) Over Time in Ile-de-France')
       
       # Update traces to have different markers
       fig2.update_traces(mode='lines+markers')
       
       # Add more details to the layout
       fig2.update_layout(xaxis_title='Year',
                         yaxis_title='Price per m² (€)',
                         legend_title='Statistic',
                         legend=dict(y=1, x=1),
                         margin=dict(l=20, r=20, t=30, b=20))
       
       # Show the figure
       st.plotly_chart(fig2)
       
    elif chart_type == 'Carte des Transactions Immobilières':
        
        dataV = df[df['code_postal'] == 78000]
        
        @st.cache_data
        def create_map(dataV):
            # Création de l'objet carte centré sur Versailles
            m = folium.Map(location=[48.8047, 2.1204], zoom_start=12)

            # Utilisation de MarkerCluster pour gérer de nombreux marqueurs
            marker_cluster = MarkerCluster().add_to(m)

            # Boucle sur les données pour ajouter des marqueurs
            for idx, row in dataV.iterrows():
                # Création du texte pour la popup
                popup_text = f"""
                <strong>Date:</strong> {row['date_mutation']}<br>
                <strong>Adresse:</strong> {row['Adresse']}<br>
                <strong>Prix/m²:</strong> {row['Prix au m²(€)']}€<br>
                <strong>Surface:</strong> {row['surface_reelle_bati']} m²<br>
                <strong>Prix total:</strong> {row['valeur_fonciere']}€
                """
                folium.Marker(
                    location=[row['latitude'], row['longitude']],
                    popup=popup_text,
                    icon=folium.Icon(icon="info-sign")
                ).add_to(marker_cluster)

            return m

        # Création de la carte
        map_obj = create_map(dataV)
        
        st.title('Carte des Transactions Immobilières')
        st.write('Voici une carte interactive montrant les transactions immobilières en se basant sur la date de mutation, l’adresse, et les détails financiers de chaque transaction.')
        folium_static(map_obj)
        
 if page == pages[3] :
    # Define the base path to the model directory
    BASE_PATH = r'C:\Users\Akshit\DA\idf'  # Ensure this is the correct path where your models are stored
    
    # Possible choices for regions and models
    department = [
        "Essonne", "Val-d'Oise", "Seine-et-Marne", "Hauts-de-Seine",
        "Yvelines", "Val-de-Marne", "Seine-Saint-Denis",
        "Paris"
    ]
    models = ["Decision Tree", "Linear Regression", "Gradient Boosting"]
    
    @st.cache_data
    def load_model(department, model_name):
        """ Load the model based on the region and model name. """
        filename = f"{department.replace(' ', '_')}_{model_name.replace(' ', '_')}_model.joblib"
        model_path = os.path.join(BASE_PATH, filename)
        if not os.path.exists(model_path):
            st.error(f"Model file not found: {model_path}")
            return None
        return load(model_path)
    
    @st.cache_data
    def prepare_data_for_department(department, full_dataset):
        """ Prepare data for the given region. Assumes full_dataset includes 'nom_region' as a column. """
        if 'nom_departement' not in full_dataset or 'Prix au m²(€)' not in full_dataset:
            st.error("Required columns are missing from the dataset.")
            return None, None
        department_data = full_dataset[full_dataset['nom_departement'] == department]
        if department_data.empty:
            st.error("No data found for the selected region.")
            return None, None
        X = department_data.drop('Prix au m²(€)', axis=1, errors='ignore')  # Safely drop target column
        y = department_data['Prix au m²(€)']
        return X, y
    
    try:
        # Load the full dataset
        full_dataset = pd.read_csv('GouvImmo_full_MLIdF.csv', sep=',', low_memory=False)
    except Exception as e:
        st.error(f"Failed to load data: {str(e)}")
        full_dataset = None
    
    st.title('Interface de prévision des prix immobiliers')
    
    if full_dataset is not None:
        # Selection boxes for regions and models
        selected_department = st.selectbox('Select department', department)
        selected_model = st.selectbox('Select Model', models)
    
        if st.button('Predict'):
            model = load_model(selected_department, selected_model)
            if model is not None:
                X_department, y_true_department = prepare_data_for_department(selected_department, full_dataset)
                if X_department is not None and y_true_department is not None:
                    y_pred_department = model.predict(X_department)
                    mae = mean_absolute_error(y_true_department, y_pred_department)
                    mape = mean_absolute_percentage_error(y_true_department, y_pred_department) 
                    st.write(f"Mean Absolute Error (MAE): {mae:.2f}")
                    st.write(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
    else:
        st.write("Please check the data file path and ensure it's correct.")

 if page == pages[4] :
  st.write("""
 <div style="text-align: justify;">
  <p>  Une piste d’amélioration qui pourrait permettre selon nous d’augmenter les performances du modèle serait justement d’obtenir les informations complémentaires pouvant impacter le prix d’un bien. Une partie pourrait provenir des sites d’agences immobilières (se loger, bienici, pap, leboncoin etc), et il faudrait donc effectuer du webscrapping par exemple pour les récupérer. Cependant, là encore il y aurait plusieurs limites à cette piste :
       Tous les biens vendus n’ont pas forcément été mis en vente sur un site d’agence immobilière
       On aurait toujours le problème de la clé commune, comme les informations nécessaires comme l’adresse ne sont pas disponibles sur ces sites.
       Il faudrait être capable de retrouver ces données de façon rétroactive,
       C’est -à -dire retrouver les annonces des mois après que les biens sont vendus.
       Une piste d’amélioration qui pourrait être possible mais que nous n’avons pas intégré par manque de temps serait de lier des données externes comme la présence d’écoles, transports, parcs etc. Cela pourrait être des données supplémentaires pouvant impacter le prix de vente des biens, même si là encore la liaison ne serait que relative (peut-être effectuer une liaison via la rue par exemple ?)
    </p>
 </div>
    """, unsafe_allow_html=True)  
  img_col1, img_col2, img_col3 = st.columns(3)

  with img_col1:
        st.image("eric.jpg", use_column_width=False, caption='Eric')

  with img_col2:
        st.image("dev.jpg", use_column_width=False, caption='Marc')

  with img_col3:
        st.image("momo.jpg", use_column_width=False, caption='Mohamed')
   
 with col[1]:
  with st.expander('About', expanded=True):
        st.write('''
            - Data: [Data.Gouv.fr](https://files.data.gouv.fr/geo-dvf/latest/csv/).
            - :orange[**DataScientest**]: Ce projet a été réalisé dans le cadre de notre formation en data science via l'organisme Datascientest.
            - :green[**Participants**]: Éric LOUGUET Marc DEVADEVAN Mohamed KEITA
                 ''')
            