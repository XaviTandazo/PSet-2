# feature_engineering.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV
from scipy.stats import pearsonr

# Función para cargar los datos
def load_data(file_path):
    return pd.read_csv(file_path)

# Función para calcular camas por persona y otras características
def add_feature_columns(df):
    # Cantidad de camas por persona
    df['beds_per_person'] = df['beds'] / df['accommodates']
    # Precio por cama
    df['price_per_bed'] = df['log_price'] / df['beds']
    # Precio por habitaciones
    df['price_per_bedroom'] = df['log_price'] / df['bedrooms']
    
    return df

# Función para realizar el one-hot encoding
def one_hot_encode(df):
    df = pd.get_dummies(df, columns=['property_type', 'room_type', 'bed_type', 
                                      'cancellation_policy', 'host_identity_verified', 
                                      'instant_bookable'], drop_first=True)
    return df

# Función para calcular la distancia usando la fórmula de Haversine
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Radio de la Tierra en kilómetros
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c

# Función para calcular la distancia a un punto de referencia
def add_distance_column(df):
    latitude_mean = df['latitude'].mean()
    longitude_mean = df['longitude'].mean()
    df['distance_to_center'] = df.apply(lambda row: haversine(latitude_mean, longitude_mean, row['latitude'], row['longitude']), axis=1)
    return df

# Función para manejar valores infinitos y NaN
def handle_infinity_and_missing(df):
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(df.mean(), inplace=True)
    return df

# Función para normalizar columnas
def normalize_columns(df):
    columns_to_normalize = ['accommodates', 'bathrooms', 'review_scores_rating', 
                            'bedrooms', 'beds', 'beds_per_person', 'price_per_bed', 'price_per_bedroom', 'distance_to_center']
    scaler_minmax = MinMaxScaler()
    df[columns_to_normalize] = scaler_minmax.fit_transform(df[columns_to_normalize])
    return df

# Función para calcular la importancia de las características usando Random Forest
def feature_importance(df):
    X = df.drop(columns=['log_price'])  # Suponiendo que 'log_price' es la variable objetivo
    y = df['log_price']
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    feature_importances = model.feature_importances_
    importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importances})
    importance_df = importance_df.sort_values(by='Importance', ascending=False)
    return importance_df

# Función para obtener los coeficientes de Lasso
def lasso_coef(df):
    X = df.drop(columns=['log_price'])
    y = df['log_price']
    lasso = LassoCV(alphas=[0.01, 0.1, 1, 10], cv=5, random_state=42)
    lasso.fit(X, y)
    lasso_coef = pd.DataFrame({'Feature': X.columns, 'Coefficient': lasso.coef_})
    return lasso_coef.sort_values(by='Coefficient', ascending=False)

# Función para calcular la correlación de Pearson
def pearson_correlation(df):
    correlation_results = {}
    for col in df.columns:
        if col != 'log_price':
            corr, _ = pearsonr(df['log_price'], df[col])
            correlation_results[col] = corr
    return correlation_results

# Función para eliminar columnas menos relevantes
def drop_irrelevant_columns(df):
    columns_to_drop = ['cleaning_fee', 'review_scores_rating', 'beds_per_person']
    df = df.drop(columns=columns_to_drop)
    return df
