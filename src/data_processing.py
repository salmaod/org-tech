import pandas as pd
import numpy as np
import os

def load_data(file_path):
    """
    Charger les données à partir du fichier CSV.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Le fichier {file_path} n'existe pas.")
    
    data = pd.read_csv(file_path)
    print(f"Données chargées depuis {file_path}.")
    return data

def clean_data(data):
    """
    Nettoyer les données en gérant les valeurs manquantes et les colonnes inutiles.
    """
    # Suppression des lignes avec des revenus manquants pour les modèles
    data_cleaned = data.dropna(subset=['jPhone_Pro_revenue', 'Kaggle_Pixel_5_revenue', 'Planet_SX_revenue']).copy()

    # Remplissage des valeurs manquantes pour les colonnes numériques
    numeric_cols = data_cleaned.select_dtypes(include=['number']).columns
    data_cleaned[numeric_cols] = data_cleaned[numeric_cols].fillna(data_cleaned[numeric_cols].mean())

    # Remplissage des valeurs manquantes pour les colonnes catégoriques
    categorical_cols = data_cleaned.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        data_cleaned[col] = data_cleaned[col].fillna(data_cleaned[col].mode()[0])

    # Conversion de la colonne 'date' en type datetime
    if 'date' in data_cleaned.columns:
        data_cleaned['date'] = pd.to_datetime(data_cleaned['date'], errors='coerce')

    print("Nettoyage des données terminé.")
    return data_cleaned

def process_features(data):
    """
    Préparer les caractéristiques pour la modélisation.
    """
    # Ajouter une colonne pour le revenu total
    data['Total_Revenue'] = data[['jPhone_Pro_revenue', 'Kaggle_Pixel_5_revenue', 'Planet_SX_revenue']].sum(axis=1)

    # Supprimer les colonnes inutiles
    features = data.drop(columns=['Total_Revenue', 'jPhone_Pro_revenue', 'Kaggle_Pixel_5_revenue', 'Planet_SX_revenue'])
    target = data['Total_Revenue']

    # Encodage des variables catégoriques
    features = pd.get_dummies(features, drop_first=True)

    print("Préparation des caractéristiques terminée.")
    return features, target

def describe_data(data):
    """
    Générer des statistiques descriptives pour les colonnes numériques et catégoriques.
    """
    numeric_cols = data.select_dtypes(include=['number']).columns
    categorical_cols = data.select_dtypes(include=['object']).columns

    numeric_desc = data[numeric_cols].describe().transpose().reset_index()
    numeric_desc.rename(columns={"index": "Colonne"}, inplace=True)

    categorical_desc = data[categorical_cols].describe().transpose().reset_index()
    categorical_desc.rename(columns={"index": "Colonne"}, inplace=True)

    return numeric_desc, categorical_desc

def summarize_data(data_before, data_after):
    """
    Résumer les statistiques descriptives avant et après nettoyage.
    """
    numeric_summary_before = data_before.describe(include=['number']).T.reset_index()
    numeric_summary_before = numeric_summary_before[['index', 'count', 'mean', 'std', 'min', '50%', 'max']]
    numeric_summary_before.columns = ['Colonne', 'Nombre de valeurs', 'Moyenne', 'Écart-type', 'Min', 'Médiane', 'Max']
    numeric_summary_before = numeric_summary_before.round(2)

    numeric_summary_after = data_after.describe(include=['number']).T.reset_index()
    numeric_summary_after = numeric_summary_after[['index', 'count', 'mean', 'std', 'min', '50%', 'max']]
    numeric_summary_after.columns = ['Colonne', 'Nombre de valeurs', 'Moyenne', 'Écart-type', 'Min', 'Médiane', 'Max']
    numeric_summary_after = numeric_summary_after.round(2)

    return numeric_summary_before, numeric_summary_after

def transform_to_long(data):
    """
    Transformer les données en format long pour la visualisation.
    """
    data_long = data.melt(
        id_vars=['marketing_score', 'competition_index', 'customer_satisfaction',
                 'purchasing_power_index', 'weather_condition', 'tech_event', '5g_phase',
                 'store_traffic', 'public_transport', 'city'],
        value_vars=['jPhone_Pro_revenue', 'Kaggle_Pixel_5_revenue', 'Planet_SX_revenue'],
        var_name='smartphone_model',
        value_name='revenue'
    )
    return data_long

def calculate_city_revenue(data):
    """
    Calculer les revenus moyens par ville pour chaque modèle.
    """
    city_revenue = data.groupby('city')[['jPhone_Pro_revenue', 'Kaggle_Pixel_5_revenue', 'Planet_SX_revenue']].mean()
    city_revenue['Total_Revenue'] = city_revenue.sum(axis=1)
    city_revenue = city_revenue.sort_values(by='Total_Revenue', ascending=False)
    return city_revenue

def compute_correlation_matrix(data):
    """
    Calculer la matrice de corrélation des colonnes numériques.
    """
    numeric_cols = data.select_dtypes(include=['number']).columns
    correlation_matrix = data[numeric_cols].corr()
    return correlation_matrix

def harmonize_columns(X_train, X_test):
    """
    Harmoniser les colonnes entre X_train et X_test pour éviter les problèmes de dimensions.
    Supprimer les colonnes inutiles avant harmonisation.
    """
    # Identifier et supprimer les colonnes inutiles
    irrelevant_cols_train = [col for col in X_train.columns if col.startswith("Unnamed: 0_")]
    irrelevant_cols_test = [col for col in X_test.columns if col.startswith("Unnamed: 0_")]
    X_train = X_train.drop(columns=irrelevant_cols_train, errors="ignore")
    X_test = X_test.drop(columns=irrelevant_cols_test, errors="ignore")

    # Identifier les colonnes supplémentaires
    train_extra_cols = set(X_train.columns) - set(X_test.columns)
    test_extra_cols = set(X_test.columns) - set(X_train.columns)
    
    if train_extra_cols or test_extra_cols:
        print(f"Colonnes supplémentaires dans X_train : {train_extra_cols}")
        print(f"Colonnes supplémentaires dans X_test : {test_extra_cols}")
    
    # Garder uniquement les colonnes communes
    common_cols = list(set(X_train.columns) & set(X_test.columns))
    X_train = X_train[common_cols]
    X_test = X_test[common_cols]

    print("Colonnes harmonisées entre X_train et X_test.")
    return X_train, X_test

def save_cleaned_data(data, output_dir="data/processed", file_name="cleaned_data.csv"):
    """
    Sauvegarder les données nettoyées dans un fichier CSV.
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, file_name)
    data.to_csv(output_path, index=False)
    print(f"Données nettoyées sauvegardées dans : {output_path}")