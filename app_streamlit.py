import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import json
import os
import joblib
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import re 


tokenizer_path = 'Bert_final/tokenizer_directory (8)'  # Remplacez par votre chemin
# Charger le tokenizer depuis le répertoire local
tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
# Spécifiez le chemin vers le modèle enregistré
model_path = "Bert_final/best_model (8).pt"





# Nettoyage de texte simple

# def clean_text(text):
    
#     text = text.lower() # Conversion en minuscules
#     text = re.sub(r'\b(?:https?://|www\.)\S+\b', '', text) # Suppression des URLs
#     text = re.sub(r'\s+', ' ', text).strip() # Suppression des espaces superflus
#     text = re.sub(r'#(\w+)', r'\1', text) # Suppression des hashtags tout en conservant les mots
#     text = re.sub(r'[^\w\s!?\.,]', '', text)# Conserver les signes de ponctuation importants comme ! et ?
    
#     return text


import re
import emoji
import contractions

def clean_text(text):
    text = text.lower()  # Mettre en minuscules
    text = emoji.demojize(text)  # Convertir les emojis en texte
    # Supprimer les emojis
    text = emoji.replace_emoji(text, replace='')   
    text = contractions.fix(text)  # Expander les contractions
    text = re.sub(r'@\w+', '', text)  # Supprimer les mentions Twitter
    text = re.sub(r'\b(?:https?://|www\.)\S+\b', '', text)  # Supprimer les URLs
    text = re.sub(r'#(\w+)', r'\1', text)  # Supprimer le symbole # et garder le mot après
    text = re.sub(r'[^\w\s\.,!?\'\"-]', '', text)  # Supprimer les caractères spéciaux, sauf ponctuation
    text = re.sub(r'\d+', '', text)  # Supprimer les chiffres
    text = re.sub(r'\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b', '', text)  # Supprimer les dates
    text = re.sub(r'[-/]', '', text)  # Supprimer les tirets et barres restantes
    text = re.sub(r'[^\w\s\.,!?\'\"-]', '', text) # Supprimer tous les caractères non alphanumériques
    text = re.sub(r'\s+', ' ', text).strip()  # Supprimer les espaces superflus

    return text




# Custom CSS to style the sidebar and other elements
st.markdown("""
    <style>
        /* Style the entire sidebar container */
    
        .css-18e3th9 {
            #background-color: #FF8C00; /* Dark Orange for sidebar background */
            #padding: 10px;
        }
        
        /* Style the sidebar header text (username) */
        .css-1d391kg {
            color: white;
        }

        
        /* Style the menu buttons */
        .stButton > button {
            background-color: #FF8C00; /* Dark Orange to match the sidebar */
            color: white; /* White text for buttons */
            border: none;
            border-radius: 5px;
            padding: 10px;
            width: 100%;
            margin: 5px 0;
            text-align: left;
            font-size: 16px;
        }
        .stButton > button:hover {
            background-color: #e67300; /* Darker orange for hover effect */
            color: #ffffff;
        }
    </style>
""", unsafe_allow_html=True)

# Sidebar Menu
#st.sidebar.image("https://via.placeholder.com/150x50?text=LOGO", use_column_width=True)
st.sidebar.write("**Choisissez une option de menu 👇**", unsafe_allow_html=True)


#st.sidebar.image("logo.jpg", width=150)  # URL du logo
    

# List of clickable menu options (Titles)
menu_items = [ "A propos du projet 📋", "Détecter le sentiment 🧠💬","Aperçue de dataset 🔍", "nuages des mots 📊", "Analyse Temporelle 🕒", "Répartition Géographique 🗺️","Détéction des sujets 🧐"]





dataset=None









def show_dashboard():
    # Ajouter un logo au dashboard via URL
   
    st.title("🦠💉 Analyse des Sentiments sur la Vaccination contre le COVID-19")

    # Introduction et Contexte
    st.markdown("## 📝 Introduction")
    st.write("""
    Ce dashboard présente une analyse des sentiments des patients concernant la vaccination contre le COVID-19, en utilisant des données de Twitter entre 2020     et 2021.
    """)
    
    st.markdown("---")  # Séparateur visuel

    # Objectifs du Projet
    st.markdown("## 🎯 Objectifs du Projet")
    st.write("""
    1. **Identifier** les sentiments dominants exprimés par les patients.
    2. **Analyser** la répartition des sentiments et leur évolution dans le temps.
    3. **Explorer** les tendances géographiques des sentiments.
    4. **Détecter** les sujets des tweets
    5. **Explorer** les nuages de mots faits
    6. **Fournir** des visualisations claires pour aider à la prise de décision.
    
    
    """)
    
    st.markdown("---")

    # Créer deux colonnes pour la présentation des résultats et du guide de navigation
    
    

    st.markdown("## 🗺️ Guide de Navigation")
    st.write("Utilisez le menu sur la gauche pour explorer les différentes sections du projet d'analyse des sentiments :")
    st.write("- **A propos du projet 📋 :** Informations générales sur le projet et ses objectifs.")
    st.write("- **Détecter le sentiment 🧠💬 :** Outil permettant d'analyser les sentiments en fonction d'un texte ou d'un dataset fourni par l'utilisateur.")
    st.write("- **Aperçu du dataset 🔍 :** Visualisation rapide des données utilisées pour l'analyse.")
    st.write("- **Nuage de mots 📊 :** Affichage des mots les plus fréquemment utilisés dans chaque catégorie de sentiment.")
    st.write("- **Analyse Temporelle 🕒 :** Évolution des sentiments dans le temps.")
    st.write("- **Répartition Géographique 🗺️ :** Analyse des sentiments en fonction des régions géographiques.")
    st.write("- **Détection des sujets 🧐 :** Extraction des thèmes principaux liés aux sentiments exprimés.")

    
    st.markdown("---")

    # Ajouter des éléments supplémentaires pour rendre l'expérience plus interactive
    st.markdown("### 💬 Que disent les patients ?")
    st.write("Explorez les tweets pour voir les tendances sentimentales et les sujets clés.")

    st.markdown("---")





 




# # Variable globale pour stocker le dataset
# dataset = None


# def load_dataset(filepath):
#     global dataset
#     dataset = pd.read_csv(filepath)
#     return dataset

# def show_data():
#     st.title("Aperçu du Dataset")
#     # Vérifiez si le dataset est chargé
#     if dataset is not None:
#         st.write(dataset.head(20))
#     else:
#         st.write("Aucun dataset n'a été chargé. Veuillez charger un fichier.")

# # Charger le dataset (exécuter cette fonction au début ou dans une autre partie du code)
# dataset = load_dataset('C:/Users/Lenovo/Downloads/PFA/code/data_balanced2.csv')

































def show_overview():
    global dataset
    st.header("Vue d'ensemble des Sentiments")
    st.write("Voici ici la répartition globale des sentiments dans notre dataset .")
    if 'Sentiment_vader' in dataset.columns:
        # Compter le nombre de textes dans chaque catégorie
        sentiment_counts = dataset['Sentiment_vader'].value_counts()

        # Créer le diagramme circulaire
        plt.figure(figsize=(8, 8))
        plt.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', colors=['#1f77b4', '#ff7f0e', '#2ca02c'])
        plt.title('Proportion des Sentiments')

        # Afficher le graphique dans Streamlit
        st.pyplot(plt)
        plt.close()  # Fermer la figure pour éviter des problèmes d'affichage
    else:
        st.error("La colonne 'Sentiment' est manquante dans le dataset.")




# Fonction pour afficher la visualisation des sentiments
def show_sentiment_visualization():
    st.title("Voici les nuages de mots")

                
    if 'cleaned_text_vader' in dataset.columns:
        if 'Sentiment_vader' in dataset.columns:
            
            # Première ligne : Nuage de mots Positif et Négatif
            col1, col2 = st.columns(2)
            
            with col1:
           
                        # Afficher le nuage de mots pour tous les textes
                        text_combined = " ".join(dataset['cleaned_text_vader'].dropna().astype(str))
                        wordcloud = WordCloud(
                            max_words=500,
                            width=1600,
                            height=800,
                            background_color="white"
                        ).generate(text_combined)
                        plt.figure(figsize=(10, 5))
                        plt.imshow(wordcloud, interpolation='bilinear')
                        plt.axis("off")
                        plt.title("Nuage de mots pour l'ensemble des textes", fontsize=19)
                        st.pyplot(plt)
                        plt.close()  # Fermer la figure pour éviter des problèmes d'affichage


            with col2:

                neutre_texts = dataset[dataset['Sentiment_vader'] == 'Neutral']['cleaned_text_vader']
                if not neutre_texts.empty:
                    neutre_combined = " ".join(neutre_texts.dropna().astype(str))
                    wordcloud_neutre = WordCloud(
                            background_color="white",
                            width=800,
                            height=400,
                            max_words=200,
                            colormap="viridis"
                        ).generate(neutre_combined)
                    plt.figure(figsize=(10, 5))
                    plt.imshow(wordcloud_neutre, interpolation="bilinear")
                    plt.axis("off")
                    plt.title('Neutral Word Cloud', fontsize=18)
                    st.pyplot(plt)
                    plt.close()
                else:
                    st.write("Aucun texte neutre trouvé dans le dataset.")

            # Deuxième ligne : Nuage de mots Neutre
            col3, col4 = st.columns(2)
            
            with col3:
                
                positive_texts = dataset[dataset['Sentiment_vader'] == 'Positive']['cleaned_text_vader']
                if not positive_texts.empty:
                    positive_combined = " ".join(positive_texts.dropna().astype(str))
                    wordcloud_positive = WordCloud(
                        background_color="white",
                        width=800,
                        height=400,
                        max_words=200,
                        colormap="viridis"
                    ).generate(positive_combined)
                    plt.figure(figsize=(10, 5))
                    plt.imshow(wordcloud_positive, interpolation="bilinear")
                    plt.axis("off")
                    plt.title('Positive Word Cloud', fontsize=18)
                    st.pyplot(plt)
                    plt.close()
                else:
                    st.write("Aucun texte positif trouvé dans le dataset.")
            
                
            # Remettre le nuage de mots général dans la colonne restante si nécessaire
            with col4:
                
                negative_texts = dataset[dataset['Sentiment_vader'] == 'Negative']['cleaned_text_vader']
                if not negative_texts.empty:
                    negative_combined = " ".join(negative_texts.dropna().astype(str))
                    wordcloud_negative = WordCloud(
                        background_color="white",
                        width=800,
                        height=400,
                        max_words=200,
                        colormap="viridis"
                    ).generate(negative_combined)
                    plt.figure(figsize=(10, 5))
                    plt.imshow(wordcloud_negative, interpolation="bilinear")
                    plt.axis("off")
                    plt.title('Negative Word Cloud', fontsize=18)
                    st.pyplot(plt)
                    plt.close()
                else:
                    st.write("Aucun texte négatif trouvé dans le dataset.")
            
        else:
            st.error("La colonne 'Sentiment' est manquante dans le dataset.")
    else:
        st.error("La colonne 'cleaned_text' est manquante dans le dataset.")
  








def show_temporal_analysis():
    st.header("Analyse Temporelle des Commentaires")

    if 'date' in dataset.columns:
        # Convertir la colonne en format datetime
        dataset['date'] = pd.to_datetime(dataset['date'], errors='coerce')

        # Fonction pour tracer les graphes temporels avec couleur personnalisée
        def plot_temporal_trends(data, title, ylabel, color):
            monthly_counts = data.groupby(pd.Grouper(key='date', freq='M')).size()
            
            plt.figure(figsize=(6, 4))  # Ajuster la taille pour avoir deux graphiques par ligne
            sns.barplot(x=monthly_counts.index, y=monthly_counts.values, color=color, edgecolor='black')
            plt.title(title, fontsize=12)
            plt.xlabel('Date', fontsize=10)
            plt.ylabel(ylabel, fontsize=10)
            plt.xticks(rotation=45)
            plt.grid(axis='y')
            st.pyplot(plt)
            plt.close()

        # 1. Graphique pour l'évolution globale des commentaires
        col1, col2 = st.columns(2)
        with col1:
            plot_temporal_trends(
                dataset, 
                "Evolution of the Number of Comments Over Time", 
                "Number of Comments",
                color='purple'  # Couleur pour l'évolution globale
            )
        with col2:
            # 2. Graphique pour les commentaires positifs
            positive_comments = dataset[dataset['Sentiment_vader'] == 'Positive']
            plot_temporal_trends(
                positive_comments, 
               "Evolution of the Number of Positive Comments Over Time", 
                "Number of Positive Comments",
                color='green'  # Couleur pour les commentaires positifs
            )

        # Nouvelle ligne pour les graphiques suivants
        col1, col2 = st.columns(2)
        with col1:
            # 3. Graphique pour les commentaires négatifs
            negative_comments = dataset[dataset['Sentiment_vader'] == 'Negative']
            plot_temporal_trends(
                negative_comments, 
                "Evolution of the Number of negative Comments Over Time", 
                "Number of negative Comments",
                color='red'  # Couleur pour les commentaires négatifs
            )
        with col2:
            # 4. Graphique pour les commentaires neutres
            neutral_comments = dataset[dataset['Sentiment_vader'] == 'Neutral']
            plot_temporal_trends(
                neutral_comments, 
                "Evolution of the Number of neutral Comments Over Time", 
                "Number of neutral Comments",
                color='blue'  # Couleur pour les commentaires neutres
            )

    else:
        st.error("La colonne 'date_column' est manquante dans le dataset.")


# def show_geographical_distribution():
#     st.title("Répartition Géographique des Sentiments")

#     # Vérifiez que les colonnes nécessaires sont présentes
#     if 'user_location' in dataset.columns and 'Sentiment_vader' in dataset.columns:
#         # Normaliser les valeurs de la colonne 'Sentiment'
#         dataset['Sentiment_vader'] = dataset['Sentiment_vader'].str.strip().str.capitalize()

#         # Comptez le nombre de sentiments par pays
#         sentiment_counts_by_country = dataset.groupby(['user_location', 'Sentiment_vader']).size().reset_index(name='count')

#         # Créez la carte avec Plotly
#         fig = px.choropleth(
#             sentiment_counts_by_country,
#             locations='user_location',
#             locationmode='country names',
#             color='Sentiment_vader',
#             hover_name='user_location',
#             hover_data={'count': True},
#             color_discrete_map={'Positif': 'blue', 'Neutre': 'green', 'Négatif': 'red'},
#             title='Répartition Géographique des Sentiments'
#         )

#         # Mise à jour de la mise en page
#         fig.update_layout(geo=dict(scope='world'))

#         # Afficher la carte dans Streamlit
#         st.plotly_chart(fig)
#     else:
#         st.error("Les colonnes 'user_location' ou 'Sentiment' sont manquantes dans le dataset.")



# # Fonction pour enregistrer les résultats dans un fichier CSV
# def save_monthly_counts(data, filename):
#     monthly_counts = data.groupby(pd.Grouper(key='date', freq='M')).size()
#     monthly_counts.to_csv(filename, header=['count'])

# # Fonction pour tracer les graphes temporels avec couleur personnalisée
# def plot_temporal_trends(data, title, ylabel, color):
#     monthly_counts = data.groupby(pd.Grouper(key='date', freq='M')).size()
    
#     plt.figure(figsize=(6, 4))  # Ajuster la taille pour avoir deux graphiques par ligne
#     sns.barplot(x=monthly_counts.index, y=monthly_counts.values, color=color, edgecolor='black')
#     plt.title(title, fontsize=12)
#     plt.xlabel('Date', fontsize=10)
#     plt.ylabel(ylabel, fontsize=10)
#     plt.xticks(rotation=45)
#     plt.grid(axis='y')
#     st.pyplot(plt)
#     plt.close()
#     return monthly_counts

# def show_temporal_analysis():
#     st.header("Analyse Temporelle des Commentaires")

#     if 'date' in dataset.columns:
#         # Convertir la colonne en format datetime
#         dataset['date'] = pd.to_datetime(dataset['date'], errors='coerce')

#         # 1. Graphique pour l'évolution globale des commentaires
#         global_data = dataset
#         global_counts = plot_temporal_trends(
#             global_data, 
#             "Évolution du Nombre de Commentaires au Cours du Temps", 
#             "Nombre de Commentaires",
#             color='purple'  # Couleur pour l'évolution globale
#         )
#         save_monthly_counts(global_data, 'global_comments.csv')  # Enregistrer les résultats

#         # 2. Graphique pour les commentaires positifs
#         positive_comments = dataset[dataset['Sentiment_vader'] == 'Positive']
#         positive_counts = plot_temporal_trends(
#             positive_comments, 
#             "Évolution du Nombre de Commentaires Positifs au Cours du Temps", 
#             "Nombre de Commentaires Positifs",
#             color='green'  # Couleur pour les commentaires positifs
#         )
#         save_monthly_counts(positive_comments, 'positive_comments.csv')  # Enregistrer les résultats

#         # 3. Graphique pour les commentaires négatifs
#         negative_comments = dataset[dataset['Sentiment_vader'] == 'Negative']
#         negative_counts = plot_temporal_trends(
#             negative_comments, 
#             "Évolution du Nombre de Commentaires Négatifs au Cours du Temps", 
#             "Nombre de Commentaires Négatifs",
#             color='red'  # Couleur pour les commentaires négatifs
#         )
#         save_monthly_counts(negative_comments, 'negative_comments.csv')  # Enregistrer les résultats

#         # 4. Graphique pour les commentaires neutres
#         neutral_comments = dataset[dataset['Sentiment_vader'] == 'Neutral']
#         neutral_counts = plot_temporal_trends(
#             neutral_comments, 
#             "Évolution du Nombre de Commentaires Neutres au Cours du Temps", 
#             "Nombre de Commentaires Neutres",
#             color='blue'  # Couleur pour les commentaires neutres
#         )
#         save_monthly_counts(neutral_comments, 'neutral_comments.csv')  # Enregistrer les résultats

#     else:
#         st.error("La colonne 'date' est manquante dans le dataset.")







def show_positive_tweet_locations():
    st.header("Top 12 pays avec le Plus Grand Nombre de Tweets Positifs")

    # Filtrer les tweets positifs
    positive_tweets1 = dataset[dataset['Sentiment_vader'] == 'Positive']

    # Compter le nombre de tweets positifs pour chaque emplacement
    positive_distribution = positive_tweets1['user_location'].value_counts().reset_index()
    positive_distribution.columns = ['user_location', 'count']

    # Mapper les noms d'emplacements vers des noms de pays reconnus
    location_mapping = {
    'Bengaluru, India': 'India',
    'India': 'India',
    'New Delhi, India': 'India',
    'Estados Unidos': 'United States',
    'United States': 'United States',
    'Mumbai, India': 'India',
    'London, England': 'United Kingdom',
    'California, USA': 'United States',
    'New Delhi': 'India',
    'Canada': 'Canada',
    'London': 'United Kingdom',
    'Hyderabad, India': 'India',
    'Mumbai': 'India',
    'Toronto, Ontario': 'Canada',
    'Los Angeles, CA': 'United States',
    'Chennai, India': 'India',
    'United Kingdom': 'United Kingdom',
    'Sri Lanka': 'Sri Lanka',
    'Beijing, China': 'China',
    'Washington, DC': 'United States',
    'USA': 'United States',
    'Moscow, Russia': 'Russia',
    'Malaysia': 'Malaysia',
    'Pakistan': 'Pakistan',
    'California, United States': 'United States',
    'New York, NY': 'United States',
    'Delhi': 'India',
    'Pune, India': 'India',
    'Hyderabad': 'India',
    'Beijing': 'China',
    'São Paulo, Brasil': 'Brazil',
    'Earth': 'Unknown',
    'Houston, TX': 'United States',
    'Chennai': 'India',
    'New York, USA': 'United States',
    'Bangalore': 'India',
    'Michigan, USA': 'United States',
    'Singapore': 'Singapore',
    'Bhubaneshwar, India': 'India',
    'Kolkata, India': 'India',
    'New York': 'United States',
    'Chicago, IL': 'United States',
    'Australia': 'Australia',
    'भारत': 'India',  # 'Bharat' is another name for India in Hindi
    'Karachi, Pakistan': 'Pakistan',
    'San Francisco, CA': 'United States',
    'United Arab Emirates': 'United Arab Emirates',
    'Ontario, Canada': 'Canada'
}

    # Appliquer le mappage pour corriger les noms des pays
    positive_distribution['user_location'] = positive_distribution['user_location'].replace(location_mapping)

    # Filtrer pour les emplacements mappés uniquement
    positive_distribution = positive_distribution[positive_distribution['user_location'].isin(location_mapping.values())]

    # Sélectionner les 20 emplacements avec le plus grand nombre de tweets positifs
    top_20_positive = positive_distribution.head(50)

    # Créer la carte avec Plotly
    fig = px.choropleth(
        top_20_positive,
        locations='user_location',
        locationmode='country names',
        color='user_location',  # Utiliser l'emplacement pour la couleur discrète
        hover_name='user_location',
        hover_data={'count': True},
        #title='Top 12 pays avec le Plus Grand Nombre de Tweets Positifs',
        color_discrete_sequence=px.colors.qualitative.Safe  # Utiliser une palette de couleurs discrètes
    )
    # Mise à jour de la mise en page de la carte
    fig.update_layout(geo=dict(scope='world'))

    # Afficher la carte dans Streamlit
    st.plotly_chart(fig)





def show_negative_tweet_locations():
    st.header("Top 12 pays avec le Plus Grand Nombre de Tweets Négatifs")

    # Filtrer les tweets négatifs
    negative_tweets1 = dataset[dataset['Sentiment_vader'] == 'Negative']

    # Compter le nombre de tweets négatifs pour chaque emplacement
    negative_distribution = negative_tweets1['user_location'].value_counts().reset_index()
    negative_distribution.columns = ['user_location', 'count']

    # Mapper les noms d'emplacements vers des noms de pays reconnus
    location_mapping = {
        'Toronto, Canada and Worldwide': 'Canada',
        'India': 'India',
        'United States': 'United States',
        'New Delhi, India': 'India',
        'USA': 'United States',
        'London': 'United Kingdom',
        'Mumbai, India': 'India',
        'Hong Kong': 'Hong Kong',
        'Canada': 'Canada',
        'New Delhi': 'India',
        'Los Angeles, CA': 'United States',
        'Toronto, Ontario': 'Canada',
        'London, England': 'United Kingdom',
        'Mumbai': 'India',
        'United Kingdom': 'United Kingdom',
        'Bengaluru, India': 'India',
        'Sri Lanka': 'Sri Lanka',
        'California, USA': 'United States',
        'California, United States': 'United States',
        'New York, NY': 'United States',
        'Washington, DC': 'United States',
        'New York, USA': 'United States',
        'Pune, India': 'India',
        'Malaysia': 'Malaysia',
        'Chennai, India': 'India',
        'Bharat': 'India',
        'Earth': 'Unknown',
        'Colombo, Sri Lanka': 'Sri Lanka',
        'Austin, TX': 'United States',
        'Bengaluru': 'India',
        'Australia': 'Australia',
        'Planet Earth': 'Unknown',
        'Pakistan': 'Pakistan',
        'Chicago, IL': 'United States',
        'Hyderabad, India': 'India',
        'England, United Kingdom': 'United Kingdom',
        'Nairobi, Kenya': 'Kenya',
        'Bangalore': 'India',
        'South Africa': 'South Africa',
        'Texas, USA': 'United States',
        'Ireland': 'Ireland',
        'Singapore': 'Singapore',
        'Jaipur, India': 'India',
        'Hovedstaden, Danmark': 'Denmark',
        'Kolkata, India': 'India',
        'Los Angeles': 'United States',
        'Florida, USA': 'United States',
        'Ontario, Canada': 'Canada',
        'Chennai': 'India'
    }

    # Appliquer le mappage pour corriger les noms des pays
    negative_distribution['user_location'] = negative_distribution['user_location'].replace(location_mapping)

    # Filtrer pour les emplacements mappés uniquement
    negative_distribution = negative_distribution[negative_distribution['user_location'].isin(location_mapping.values())]

    # Sélectionner les 50 emplacements avec le plus grand nombre de tweets négatifs
    top_50_negative = negative_distribution.head(50)

    # Créer la carte avec Plotly
    fig = px.choropleth(
        top_50_negative,
        locations='user_location',
        locationmode='country names',
        color='user_location',  # Utiliser l'emplacement pour la couleur discrète
        hover_name='user_location',
        hover_data={'count': True},
        #title='Top 12 pays avec le Plus Grand Nombre de Tweets Négatifs',
        color_discrete_sequence=px.colors.sequential.Reds  # Utiliser une palette de couleurs pour les sentiments négatifs
    )
    # Mise à jour de la mise en page de la carte
    fig.update_layout(geo=dict(scope='world'))

    # Afficher la carte dans Streamlit
    st.plotly_chart(fig)











def top_20_country():
    st.title("Distribution of the 20 Most Frequent Locations in the Dataset")

    # Vérifiez si la colonne 'user_location' existe dans le dataset
    if 'user_location' in dataset.columns:
        # Compter le nombre d'emplacements par utilisateur
        user_location_counts = dataset['user_location'].value_counts().head(20)
        
        # Calculer le pourcentage
        total_locations = user_location_counts.sum()
        percentages = (user_location_counts / total_locations * 100).round(2)
        
        # Créer une palette de couleurs
        palette = sns.color_palette('husl', len(user_location_counts))
        
        # Créer le graphique
        plt.figure(figsize=(12, 6))
        bars = sns.barplot(x=user_location_counts.values, y=user_location_counts.index, palette=palette)
        
        # Annoter chaque barre avec le pourcentage
        for index, value in enumerate(user_location_counts.values):
            plt.text(value, index, f'{percentages[index]}%', va='center', ha='left')

        plt.xlabel('Number and Percentage of User Locations')
        plt.ylabel('User Location')
        plt.title('Distribution of User Locations')

        # Afficher le graphique dans Streamlit
        st.pyplot(plt)
        plt.close()  # Fermer la figure pour éviter des problèmes d'affichage
    else:
        st.error("La colonne 'user_location' est manquante dans le dataset.")







# def top_20_country():
#     st.title("Distribution des 20 emplacements les plus fréquents dans le dataset")

#     # Vérifiez si la colonne 'user_location' existe dans le dataset
#     if 'user_location' in dataset.columns:
#         # Compter le nombre d'emplacements par utilisateur
#         user_location_counts = dataset['user_location'].value_counts().head(20)
        
#         # Calculer le pourcentage
#         total_locations = user_location_counts.sum()
#         percentages = (user_location_counts / total_locations * 100).round(2)
        
#         # Créer une DataFrame pour sauvegarder les données
#         data_to_save = pd.DataFrame({
#             'Location': user_location_counts.index,
#             'Count': user_location_counts.values,
#             'Percentage': percentages.values
#         })

#         # Enregistrer les données dans un fichier CSV
#         data_to_save.to_csv('top_20_user_locations.csv', index=False)

#         # Créer une palette de couleurs
#         palette = sns.color_palette('husl', len(user_location_counts))
        
#         # Créer le graphique
#         plt.figure(figsize=(12, 6))
#         bars = sns.barplot(x=user_location_counts.values, y=user_location_counts.index, palette=palette)
        
#         # Annoter chaque barre avec le pourcentage
#         for index, value in enumerate(user_location_counts.values):
#             plt.text(value, index, f'{percentages[index]}%', va='center', ha='left')

#         plt.xlabel('Nombre et pourcentage d\'emplacements des utilisateurs')
#         plt.ylabel('Emplacement des utilisateurs')
#         plt.title('Distribution des emplacements des utilisateurs')

#         # Afficher le graphique dans Streamlit
#         st.pyplot(plt)
#         plt.close()  # Fermer la figure pour éviter des problèmes d'affichage
#     else:
#         st.error("La colonne 'user_location' est manquante dans le dataset.")










# def show_tweet_hourly_distribution():
#     st.header("Évolution des Tweets au Fil des Heures")

#     # Assurez-vous que la colonne de date est au format datetime
#     dataset['date_column'] = pd.to_datetime(dataset['date'])

#     # Extraire l'heure de chaque tweet
#     dataset['hour'] = dataset['date_column'].dt.hour
#     # Convertir l'heure en chaîne de caractères avec format '00', '01', '02', etc.
#     dataset['hour_formatted'] = dataset['hour'].apply(lambda x: f'{x:02d}')

#     # Compter le nombre de tweets par heure
#     hourly_counts = dataset['hour_formatted'].value_counts().sort_index()

#     # Convertir en DataFrame pour Plotly
#     df_hourly_counts = hourly_counts.reset_index()
#     df_hourly_counts.columns = ['hour', 'tweet_count']

#     # Créer le graphique de courbe avec Plotly
#     fig_curve = go.Figure()

#     fig_curve.add_trace(go.Scatter(
#         x=df_hourly_counts['hour'],
#         y=df_hourly_counts['tweet_count'],
#         mode='lines+markers',
#         line=dict(color='blue'),
#         marker=dict(size=8),
#         name='Nombre de Tweets'
#     ))

#     fig_curve.update_layout(
#         title='Évolution des Tweets au Fil des Heures',
#         xaxis_title='Heure de la Journée',
#         yaxis_title='Nombre de Tweets',
#         xaxis=dict(
#             tickmode='linear',
#             dtick=1
#         ),
#         yaxis=dict(
#             title='Nombre de Tweets',
#             autorange=True
#         ),
#         template='plotly_white'
#     )

#     # Afficher le graphique de courbe dans Streamlit
#     st.plotly_chart(fig_curve)



def show_tweet_hourly_distribution():
    st.header("Évolution des Tweets au Fil des Heures")

    # Assurez-vous que la colonne de date est au format datetime
    dataset['date_column'] = pd.to_datetime(dataset['date'])

    # Extraire l'heure de chaque tweet
    dataset['hour'] = dataset['date_column'].dt.hour

    # Convertir l'heure en chaîne de caractères avec format '00', '01', '02', etc.
    dataset['hour_formatted'] = dataset['hour'].apply(lambda x: f'{x:02d}')

    # Compter le nombre de tweets par heure
    hourly_counts = dataset['hour_formatted'].value_counts().sort_index()

    # Convertir en DataFrame pour enregistrer les données
    df_hourly_counts = hourly_counts.reset_index()
    df_hourly_counts.columns = ['hour', 'tweet_count']

    # Sauvegarder les données dans un fichier CSV
    df_hourly_counts.to_csv('tweet_hourly_distribution.csv', index=False)

    # Créer le graphique de courbe avec Plotly
    fig_curve = go.Figure()

    fig_curve.add_trace(go.Scatter(
        x=df_hourly_counts['hour'],
        y=df_hourly_counts['tweet_count'],
        mode='lines+markers',
        line=dict(color='blue'),
        marker=dict(size=8),
        name='Nombre de Tweets'
    ))

    fig_curve.update_layout(
        title='Evolution of Tweets Over the Hours',
        xaxis_title='Time of Day',
        yaxis_title='Number of Comments',
        xaxis=dict(
            tickmode='linear',
            dtick=1
        ),
        yaxis=dict(
            title='Number of Comments',
            autorange=True
        ),
        template='plotly_white'
    )

    # Afficher le graphique de courbe dans Streamlit
    st.plotly_chart(fig_curve)















# Fonction pour afficher les sujets des tweets avec LDA selon le sentiment
def show_subject_tweet(sentiment_type):  
    st.header(f"Détection des Sujets pour les Tweets {sentiment_type}")

    # Filtrer les tweets en fonction du sentiment
    if sentiment_type == 'Negative':
        filtered_tweets = dataset[dataset['Sentiment_vader'] == 'Negative']['cleaned_text_vader']
    elif sentiment_type == 'Positive':
        filtered_tweets = dataset[dataset['Sentiment_vader'] == 'Positive']['cleaned_text_vader']
    else:
        st.error("Sentiment inconnu.")
        return

    # Vectorisation des mots avec CountVectorizer
    vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
    dtm = vectorizer.fit_transform(filtered_tweets)
        
    # Appliquer LDA pour extraire les sujets
    lda = LatentDirichletAllocation(n_components=10, random_state=42)
    lda.fit(dtm)
    
    # Afficher les mots-clés pour chaque sujet
    no_top_words = 10
    topics = show_subject(lda, vectorizer.get_feature_names_out(), no_top_words)
        
    # Affichage dans Streamlit
    st.subheader(f'Mots-clés pour Chaque Sujet LDA - {sentiment_type}')
    for topic in topics:
        st.write(topic)



# Fonction pour afficher les mots-clés des sujets
def show_subject(model, feature_names, no_top_words):
    topics = []
    for topic_idx, topic in enumerate(model.components_):
        words = [feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]
        topics.append(f"Sujet {topic_idx+1}: " + ", ".join(words))
    return topics






def  show_show(texts):
    
                sentiment = predict_sentiment(texts, vectorizer, model)
                if sentiment:
                    st.write(f"Sentiment détecté est : **{sentiment}**")
                else:
                    st.error("Erreur lors de la détection du sentiment.")

# Fonction de détection de mentions de vaccins
def refer(tweet, refs):
    flag = 0
    for ref in refs:
        if ref.lower() in tweet.lower():
            flag = 1
            break
    return flag

# Fonction pour afficher l'analyse des vaccins
def show_vaccin_analysis():
    # Création des tags de référence pour 5 vaccins
    pfizer_refs = ["Pfizer", "pfizer", "Pfizer–BioNTech", "pfizer-bioNtech", "BioNTech", "biontech"]
    bbiotech_refs = ["covax", "covaxin", "Covax", "Covaxin", "Bharat Biotech", "bharat biotech", "BharatBiotech", "bharatbiotech"]
    sputnik_refs = ["russia", "sputnik", "Sputnik", "V"]
    astra_refs = ['sii', 'SII', 'adar poonawalla', 'Covishield', 'covishield', 'astra', 'zenca', 'Oxford–AstraZeneca', 'astrazenca', 'oxford-astrazenca', 'serum institute']
    moderna_refs = ['moderna', 'Moderna', 'mRNA-1273', 'Spikevax']

    # Application des références pour détecter les mentions des vaccins
    dataset['pfizer'] = dataset['cleaned_text_vader'].apply(lambda x: refer(x, pfizer_refs))
    dataset['bbiotech'] = dataset['cleaned_text_vader'].apply(lambda x: refer(x, bbiotech_refs))
    dataset['sputnik'] = dataset['cleaned_text_vader'].apply(lambda x: refer(x, sputnik_refs))
    dataset['astra'] = dataset['cleaned_text_vader'].apply(lambda x: refer(x, astra_refs))
    dataset['moderna'] = dataset['cleaned_text_vader'].apply(lambda x: refer(x, moderna_refs))

    # Liste des noms de vaccins
    vaccine_names = ['pfizer', 'moderna', 'astra', 'bbiotech', 'sputnik']

    # Calculer la distribution de chaque vaccin en nombre
    vaccine_counts = dataset[vaccine_names].sum()

    # Créer un graphique à barres pour la distribution de chaque vaccin
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(x=vaccine_counts.index, y=vaccine_counts.values, ax=ax)
    ax.set_title('Distribution des Vaccins en Nombre')
    ax.set_xlabel('Vaccin')
    ax.set_ylabel('Nombre de Tweets')

    st.pyplot(fig)



# Charger le modèle et le vectorizer
@st.cache_resource
def load_model():
    try:
        vectorizer = joblib.load('./svm_model/vectorizer2.joblib')  # Assurez-vous que le chemin du fichier est correct
        model = joblib.load('./svm_model/svm_modelFin.joblib')  # Assurez-vous que le chemin du fichier est correct
        return vectorizer, model
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle : {e}")
        return None, None

# Charger le modèle et le vectorizer
vectorizer, model = load_model()




def predict_sentiment(text):
    try:
        # # Transformation du texte en caractéristiques pour le modèle
        # text_features = vectorizer.transform([text])  # 'text' est une chaîne de caractères    
        # # Faire la prédiction avec le modèle pré-entraîné
        # prediction = model.predict(text_features)[0]  # Récupérer la prédiction pour le texte


        # Charger le modèle BERT de base
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)
        # Charger les poids du modèle en mappant vers le CPU si vous n'avez pas de GPU disponible
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        # Mettre le modèle en mode évaluation
        model.eval()
       # Nettoyer le texte
        cleaned_text = clean_text(text)

        # Tokenisation du texte avec le tokenizer BERT
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        inputs = tokenizer(cleaned_text, return_tensors='pt', padding=True, truncation=True)
        # Effectuer la prédiction avec le modèle
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Les logits de sortie sont les scores non normalisés pour chaque classe
        logits = outputs.logits
        
        # Identifier la classe prédite
        predicted_class_id = torch.argmax(logits, dim=-1).item()
        
        # Définir les noms des classes
        class_names = ["Negative", "Neutral", "Positive"]
        predicted_class_name = class_names[predicted_class_id]
        
        # Afficher le texte saisi
        st.write('Le texte saisi est :', text)
        
      # Afficher le sentiment prédicté en fonction de la classe
        if predicted_class_name == "Positive":
            st.success(f'Le sentiment prédit est : {predicted_class_name}')
        elif predicted_class_name == "Negative":
            st.error(f'Le sentiment prédit est : {predicted_class_name}')
        else:  # Neutre
            st.info(f'Le sentiment prédit est : {predicted_class_name}')
        return predicted_class_name
    
    except Exception as e:
        st.error(f"Erreur lors de la prédiction : {e}")
        return None








def formb():
        
        # Créer un conteneur pour le formulaire et les résultats
        with st.container() as container:
            # Formulaire de soumission de texte
            input_text = st.text_area("Entrez le texte à analyser", "")
            
            # Créer un conteneur pour les résultats de prédiction
            result_container = st.empty()
            
            # Bouton pour lancer la prédiction
            if st.button("Analyser"):
                if input_text:  # Vérifiez que le texte n'est pas vide
                    # Appeler la fonction de prédiction
                    prediction = predict_sentiment(input_text)
                    
                    # Réinitialiser le champ de texte dans le conteneur
                    input_text = ""
                    
                    # Afficher le texte saisi et le sentiment prédit dans le conteneur des résultats
                   # result_container.write('Le texte saisi est :', input_text)
                    #result_container.write(f'Le texte saisi est : {input_text}')

                    #result_container.success(f'Le sentiment prédit est : {prediction}')
                else:
                    st.warning("Veuillez entrer du texte avant de soumettre.")
















def show_details_project():
        show_dashboard()









# Fonction pour afficher un aperçu du dataset
def show_data():
    st.title("Aperçu du Dataset")
    
    # Vérifiez si le dataset est chargé
    if dataset is not None:
        st.write(dataset.head(20))
    else:
        st.write("Aucun dataset n'a été chargé. Veuillez charger un fichier.")














# # Variable globale pour stocker le dataset
# dataset = None


# def load_dataset(filepath):
#     global dataset
#     dataset = pd.read_csv(filepath)
#     return dataset

# def show_data():
#     st.title("Aperçu du Dataset")
#     # Vérifiez si le dataset est chargé
#     if dataset is not None:
#         st.write(dataset.head(20))
#     else:
#         st.write("Aucun dataset n'a été chargé. Veuillez charger un fichier.")

# # Charger le dataset (exécuter cette fonction au début ou dans une autre partie du code)
# dataset = load_dataset('C:/Users/Lenovo/Downloads/PFA/code/data_balanced2.csv')




# Fonction principale pour Streamlit
def display_sentiment_analysis():
    # DataFrame user_location_counts
    user_location_counts = pd.DataFrame({
        'user_location': ['unknown', 'Bengaluru, India', 'India', 
                         'Toronto, Canada  and Worldwide', 'New Delhi, India',
                         'United States', 'Mumbai, India', 'New Delhi', 
                         'London, England', 'Los Angeles, CA', 
                         'Canada', 'London', 'Toronto, Ontario', 
                         'Mumbai', 'Sri Lanka', 'California, USA', 
                         'Hyderabad, India', 'USA', 
                         'Estados Unidos', 'Beijing, China'],
        'count': [31391, 10265, 3667, 1850, 1433,
                  1222, 877, 562, 473, 460,
                  454, 446, 440, 439, 423,
                  388, 373, 358, 354, 349]
    })
    
    # Simulation de 'final_data' pour cet exemple
    # Utilisez votre vrai dataset ici
    final_data =dataset
    
    # Créer un DataFrame pour stocker les résultats
    results = []
    
    # Analyser chaque emplacement dans user_location_counts
    for location in user_location_counts['user_location']:
        # Filtrer les tweets par emplacement
        filtered_tweets = final_data[final_data['user_location'].str.contains(location, na=False)]
        
        # Compter les commentaires positifs et négatifs
        positive_comments = filtered_tweets[filtered_tweets['Sentiment_vader'] == 'Positive'].shape[0]
        negative_comments = filtered_tweets[filtered_tweets['Sentiment_vader'] == 'Negative'].shape[0]
        
        # Ajouter les résultats à la liste
        results.append({
            'user_location': location,
            'positive_comments': positive_comments,
            'negative_comments': negative_comments
        })
    
    # Convertir les résultats en DataFrame
    results_df = pd.DataFrame(results)
    
    # Afficher les résultats dans le tableau de bord Streamlit
    st.write("Sentiment Analysis Results by Location")
    st.dataframe(results_df)
    
    # Tracer le graphique à barres
    fig, ax = plt.subplots(figsize=(12, 8))
    
    bar_width = 0.35
    index = range(len(results_df))

    # Barres pour les commentaires positifs
    ax.bar(index, results_df['positive_comments'], bar_width, label='positive comments', color='blue')

    # Barres pour les commentaires négatifs
    ax.bar([i + bar_width for i in index], results_df['negative_comments'], bar_width, label='negative comments', color='red')

    # Ajouter des labels et un titre
    ax.set_xlabel('Location', fontsize=14)
    ax.set_ylabel('Number of Comments', fontsize=14)
    ax.set_title('Analysis of Positive and Negative Comments by Location', fontsize=16)
    ax.set_xticks([i + bar_width / 2 for i in index])
    ax.set_xticklabels(results_df['user_location'], rotation=45, ha='right')
    ax.legend()

    # Afficher le graphique
    st.pyplot(fig)











    


# Variable globale pour stocker le dataset
dataset = None


def load_uploaded_dataset():
    st.markdown("## 📊 Détecter le sentiment de votre dataset")

    global dataset
    uploaded_file = st.file_uploader("Choisissez un fichier CSV", type="csv")

    if uploaded_file is not None:
        # Lire le fichier téléchargé
        dataset = pd.read_csv(uploaded_file)
        st.success("Dataset chargé avec succès !")

        # Sauvegarder le dataset dans le répertoire courant
        return dataset
    else:
        st.warning("Veuillez télécharger un fichier CSV pour continuer.")
        return None



def load_dataset(filepath):
    global dataset
    dataset = pd.read_csv(filepath)
    return dataset

def show_data():
    st.title("Aperçu du Dataset")
    # Vérifiez si le dataset est chargé
    if dataset is not None:
        st.write(dataset)
    else:
        st.write("Aucun dataset n'a été chargé. Veuillez charger un fichier.")

        
# Charger le dataset (exécuter cette fonction au début ou dans une autre partie du code)
dataset = load_uploaded_dataset()















        


# Loop through the menu and create clickable buttons
selection ="Détecter le sentiment 🧠💬"

for item in menu_items:
    if st.sidebar.button(item):
        selection = item


        


if selection == "Aperçue de dataset 🔍":
   show_data()
   #show_overview()



elif selection =="Détecter le sentiment 🧠💬":  
    st.markdown("## 📊 Détecter le sentiment de votre texte")
    formb()
    


elif selection == "Détéction des sujets 🧐":
    show_subject_tweet('Negative')  # Pour les commentaires négatifs
    show_subject_tweet('Positive')  # Pour les commentaires positifs    

  

elif selection == "nuages des mots 📊":
   show_sentiment_visualization()


elif selection == "A propos du projet 📋":
   show_details_project()


elif selection == "Analyse Temporelle 🕒":
    show_temporal_analysis()
    show_tweet_hourly_distribution()

elif selection == "Répartition Géographique 🗺️":
        top_20_country()
        show_positive_tweet_locations()
        show_negative_tweet_locations()
        display_sentiment_analysis()



   # uploaded_file=st.file_uploader("import a dataset")
   # if uploaded_file is None:
   #    st.info("Upload a dataset through config", icon="📁")       
   #    st.stop()
       
   # df=load_dataset(uploaded_file)
   # st.dataframe(df)


elif selection == "Analyse par type de vaccin 💉🔬":
     show_vaccin_analysis()





