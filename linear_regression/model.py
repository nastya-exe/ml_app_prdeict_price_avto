import pandas as pd
import numpy as np
import joblib
import  re
from category_encoders import TargetEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge


def extract_torque_and_rpm(value):
    if pd.isna(value):
        return None, None
    line = str(value).lower().replace(',', '')
    numbers = re.findall(r'(\d+\.?\d*)', line)
    if not numbers:
        return None, None

    torque = float(numbers[0])
    rpms = [float(x) for x in numbers[1:] if float(x) > 500]
    max_rpm = max(rpms) if rpms else None

    if 'kgm' in line or (torque < 50 and 'nm' not in line):
        torque = torque * 9.81
    return torque, max_rpm


def train_and_export():
    CARS_TRAIN = 'https://github.com/evgpat/datasets/raw/refs/heads/main/cars_train.csv'
    df_train = pd.read_csv(CARS_TRAIN)

    for col in ['engine', 'max_power', 'torque', 'mileage']:
        if col in df_train.columns:
            df_train[col] = df_train[col].astype(str).str.split().str[0]
            df_train[col] = pd.to_numeric(df_train[col], errors='coerce')

    if 'torque' in df_train.columns:
        parsed_torque = df_train['torque'].apply(lambda x: pd.Series(extract_torque_and_rpm(x)))
        df_train['torque'] = parsed_torque[0]
        df_train['max_torque_rpm'] = parsed_torque[1]

    for col in ['engine', 'max_power', 'torque', 'mileage']:
        if col in df_train.columns:
            df_train[col] = df_train[col].fillna(df_train[col].median())

    df_train['power_per_liter'] = df_train['max_power'] / (df_train['engine'] / 1000)

    log_list = ['selling_price', 'km_driven', 'torque', 'max_power', 'power_per_liter']
    for col in log_list:
        df_train[f'{col}_log'] = np.log1p(df_train[col])
    df_train.drop(columns=log_list + ['max_torque_rpm', 'engine'], inplace=True, errors='ignore')

# Выбросы
    df_train_clean = df_train[
        (df_train['max_power_log'] > 2.0) &
        (df_train['power_per_liter_log'] > 2.0) &
        (df_train['km_driven_log'] > 2.0)
        ].copy()

    X_train = df_train_clean.drop('selling_price_log', axis=1)
    y_train = df_train_clean['selling_price_log']

# Encoding
    X_train['name'] = X_train['name'].str.split().str[0]

    joblib.dump(sorted(X_train['name'].unique().tolist()), 'brands.pkl')

    encoder = TargetEncoder(cols=['name'])
    X_train['name'] = encoder.fit_transform(X_train['name'], y_train)
    joblib.dump(encoder, 'target_encoder.pkl')

# Категоризация
    seat_bins = [1, 2, 5, 8, 14]
    seat_labels = ['2', '4-5', '6-8', '9-14']
    X_train['seats_cat'] = pd.cut(X_train['seats'], bins=seat_bins, labels=seat_labels)
    X_train.drop(columns=['seats'], inplace=True)

    X_train = pd.get_dummies(X_train, columns=['fuel', 'transmission', 'seller_type', 'owner', 'seats_cat'], dtype=int,
                             drop_first=True)

    model_columns = X_train.columns.tolist()
    joblib.dump(model_columns, 'model_columns.pkl')

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    joblib.dump(scaler, 'scaler.pkl')

# Обучение
    model = Ridge(alpha=10.0)
    model.fit(X_train_scaled, y_train)
    joblib.dump(model, 'ridge_model.pkl')

# Экспорт весов
    weights_df = pd.DataFrame({
        'Feature': model_columns,
        'Weight': model.coef_
    }).sort_values(by='Weight', ascending=False)
    weights_df.to_csv('model_weights.csv', index=False)


if __name__ == '__main__':
    train_and_export()