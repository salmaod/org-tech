import numpy as np

def log_transform_targets(y_train, y_test):
    """
    Appliquer une transformation logarithmique sur les cibles.
    """
    y_train_log = np.log1p(y_train)
    y_test_log = np.log1p(y_test)
    return y_train_log, y_test_log

def create_features(X):
    """
    Créer de nouvelles variables basées sur des interactions ou des transformations.
    """
    X_filtered = X.copy()
    X_filtered['interaction_marketing_satisfaction'] = X_filtered['marketing_score'] * X_filtered['customer_satisfaction']
    X_filtered['log_competition_index'] = np.log1p(X_filtered['competition_index'])
    X_filtered['power_per_traffic'] = X_filtered['purchasing_power_index'] / (X_filtered['store_traffic'] + 1e-6)
    X_filtered['city_is_paris'] = (X_filtered.get('city_Paris', 0) == 1).astype(int)
    X_filtered['city_is_lyon'] = (X_filtered.get('city_Lyon', 0) == 1).astype(int)
    return X_filtered

def remove_highly_correlated_features_original(X, threshold=0.85):
    """
    Identifier et supprimer les colonnes fortement corrélées (approche originale).
    """
    correlation_matrix = X.corr()

    highly_correlated = set()
    for i in range(len(correlation_matrix.columns)):
        for j in range(i):
            if abs(correlation_matrix.iloc[i, j]) > threshold:
                colname = correlation_matrix.columns[i]
                highly_correlated.add(colname)

    X_filtered = X.drop(columns=highly_correlated)
    return X_filtered, highly_correlated