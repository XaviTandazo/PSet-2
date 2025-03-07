import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_data(filepath):
    """Carga los datos desde un archivo CSV"""
    df = pd.read_csv(filepath)
    return df

def filter_columns(df):
    """Filtra el DataFrame para mantener solo las columnas seleccionadas"""
    columnas_utilizadas = [
        'log_price', 'property_type', 'room_type', 'accommodates', 'bathrooms', 'bed_type', 
        'cancellation_policy', 'cleaning_fee', 'latitude', 'longitude', 'review_scores_rating', 
        'bedrooms', 'beds', 'host_identity_verified', 'instant_bookable'
    ]
    df = df[columnas_utilizadas]
    return df

def map_boolean_values(df):
    """Convierte las columnas de texto t/f en valores booleanos y llena valores nulos"""
    df['host_identity_verified'] = df['host_identity_verified'].map({'t': True, 'f': False})
    df['instant_bookable'] = df['instant_bookable'].map({'t': True, 'f': False})
    df['host_identity_verified'].fillna(False, inplace=True)
    return df

def fill_missing_values(df):
    """Rellena los valores nulos en el DataFrame"""
    df['review_scores_rating'].fillna(df['review_scores_rating'].mean(), inplace=True)
    df['bathrooms'].fillna(1, inplace=True)
    df['beds'].fillna(1, inplace=True)
    df['bedrooms'].fillna(1, inplace=True)
    df.dropna(subset=['bedrooms', 'beds'], inplace=True)
    return df

def encode_labels(df):
    """Codifica las columnas categóricas usando LabelEncoder"""
    label_encoder = LabelEncoder()
    df['property_type'] = label_encoder.fit_transform(df['property_type'])
    df['room_type'] = label_encoder.fit_transform(df['room_type'])
    df['bed_type'] = label_encoder.fit_transform(df['bed_type'])
    df['cancellation_policy'] = label_encoder.fit_transform(df['cancellation_policy'])
    return df

def convert_column_types(df):
    """Convierte las columnas a los tipos de datos apropiados"""
    df['bedrooms'] = df['bedrooms'].astype('int32')
    df['bathrooms'] = df['bathrooms'].astype('int32')
    df['beds'] = df['beds'].astype('int32')
    df['cleaning_fee'] = df['cleaning_fee'].astype('int32')
    df['host_identity_verified'] = df['host_identity_verified'].astype('int32')
    df['instant_bookable'] = df['instant_bookable'].astype('int32')
    return df

def save_cleaned_data(df, output_path):
    """Guarda el DataFrame limpio en un archivo CSV"""
    df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"CSV file saved successfully at: {output_path}")
