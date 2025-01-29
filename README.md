# Projet de Prévision des Revenus de Ventes de Smartphones

## Description du Projet
Ce projet vise à explorer et modéliser les données de ventes de smartphones pour prévoir les revenus futurs à l'aide de divers modèles de machine learning. Les données proviennent d'un fichier brut et sont traitées, transformées, et utilisées dans des modèles de base et avancés.

### Structure des Fichiers

#### Dossiers principaux :
- **data** : Contient les données utilisées dans le projet.
  - `raw/telecom_sales_data.csv` : Données brutes.
  - `processed/cleaned_data.csv` : Données nettoyées après le prétraitement.
  - `exports/` : Contient les modèles entraînés et les prédictions finales.
    - `lightgbm_model.pkl` : Modèle LightGBM sauvegardé.
    - `random_forest_model.pkl` : Modèle Random Forest sauvegardé.
    - `xgboost_model.pkl` : Modèle XGBoost sauvegardé.
    - `predictions_finales.csv` : Fichier CSV contenant les prédictions finales pour la période future.

- **notebooks HTML** : Versions HTML des notebooks.
  - `01_exploration.html` : Exploration des données avec fonctions centralisées.
  - `02_modeling.html` : Modélisation des données avec fonctions centralisées.
  - `01.html` et `02.html` : Versions HTML des notebooks indépendants.

- **presentation** : Contient le rapport final.
  - `results.pdf` : Rapport décrivant le projet et ses résultats.

- **src** : Contient les fichiers Python centralisés pour les fonctions utilisées dans les notebooks.
  - `data_processing.py` : Gestion des données, nettoyage, et prétraitement.
  - `feature_engineering.py` : Création de nouvelles variables et transformation des données.
  - `models.py` : Fonctions pour l'entraînement et l'évaluation des modèles.

#### Notebooks principaux :
- **01.ipynb** : Exploration des données (version sans centralisation des fonctions).
- **02.ipynb** : Modélisation des données (version sans centralisation des fonctions).
- **01_exploration.ipynb** : Exploration des données en utilisant les fonctions centralisées dans `src/`.
- **02_modeling.ipynb** : Modélisation des données en utilisant les fonctions centralisées dans `src/`.

### Instructions pour Exécuter le Projet

#### Étape 1 : Configurer un environnement virtuel
##### Sur macOS/Linux :
```bash
python3 -m venv orange
source orange/bin/activate
```
##### Sur Windows :
```bash
python -m venv orange
orange\Scripts\activate
```

#### Étape 2 : Installer les dépendances
```bash
pip install -r requirements.txt
```

#### Étape 3 : Installer Jupyter Notebook
```bash
pip install jupyter ipykernel
python -m ipykernel install --user --name=orange --display-name "Python (orange)"
```

#### Étape 4 : Lancer le notebook
```bash
jupyter notebook
```

Ouvrez le notebook souhaité (`01_exploration.ipynb` ou `02_modeling.ipynb`) pour explorer ou modéliser les données.

### Remarques
- Les fichiers `01.ipynb` et `02.ipynb` sont indépendants et ne nécessitent pas les fichiers centralisés dans `src/`.
- Les fichiers `01_exploration.ipynb` et `02_modeling.ipynb` utilisent les fonctions centralisées dans `src/` pour une structure de code plus propre et réutilisable.

### Auteur
Ce projet a été développé pour démontrer une approche méthodique d'exploration et de modélisation des données. Le rapport final (`presentation/results.pdf`) détaille les résultats et conclusions principales.