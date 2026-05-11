import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.title("Прогноз стоимости автомобиля")
st.subheader("Введите характеристики")
available_brands = joblib.load('brands.pkl')

with st.form("car_features_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        name = st.selectbox("Марка", available_brands)
        year = st.number_input("Год выпуска", 1980, 2026, 2020)
        km_driven = st.number_input("Пробег (км)", 0, 1000000, 50000)
        fuel = st.selectbox("Тип топлива", ["Diesel", "Petrol", "LPG", "CNG"])

    with col2:
        seller_type = st.selectbox("Продавец", ["Individual", "Dealer", "Trustmark Dealer"])
        transmission = st.selectbox("Коробка передач", ["Manual", "Automatic"])
        owner = st.selectbox("Владелец", ["First Owner", "Second Owner", "Third Owner", "Fourth & Above Owner"])
        engine = st.number_input("Объем двигателя (куб.см)", 500, 8000, 1248)

    with col3:
        max_power = st.number_input("Мощность (л.с.)", 10, 1000, 74)
        torque = st.number_input("Крутящий момент", 10, 1000, 190)
        max_torque_rpm = st.number_input("Обороты крутящего момента (RPM)", 500, 6000, 2000)
        seats = st.number_input("Количество мест", 1, 14, 5)

    submitted = st.form_submit_button("Рассчитать стоимость", type="primary")

# После нажатия кнопки
if submitted:

    input_df = pd.DataFrame([{
        'name': name, 'year': year, 'km_driven': km_driven, 'fuel': fuel,
        'seller_type': seller_type, 'transmission': transmission, 'owner': owner,
        'engine': engine, 'max_power': max_power, 'torque': torque,
        'max_torque_rpm': max_torque_rpm, 'seats': seats
    }])

    encoder = joblib.load('target_encoder.pkl')
    model_columns = joblib.load('model_columns.pkl')
    scaler = joblib.load('scaler.pkl')
    model = joblib.load('ridge_model.pkl')


    input_df['power_per_liter'] = input_df['max_power'] / (input_df['engine'] / 1000)

# Логарифмирование
    log_list = ['km_driven', 'torque', 'max_power', 'power_per_liter']
    for col in log_list:
        input_df[f'{col}_log'] = np.log1p(input_df[col])

    input_df.drop(columns=log_list + ['max_torque_rpm', 'engine'], inplace=True, errors='ignore')

# категории
    input_df['name'] = input_df['name'].str.split().str[0]
    input_df['name'] = encoder.transform(input_df['name'])

    seat_bins = [1, 2, 5, 8, 14]
    seat_labels = ['2', '4-5', '6-8', '9-14']
    input_df['seats_cat'] = pd.cut(input_df['seats'], bins=seat_bins, labels=seat_labels)
    input_df.drop(columns=['seats'], inplace=True)

#OHE
    input_df = pd.get_dummies(input_df, columns=['fuel', 'transmission', 'seller_type', 'owner', 'seats_cat'], dtype=int)

    final_df = pd.DataFrame(0, index=[0], columns=model_columns)
    for col in model_columns:
        if col in input_df.columns:
            final_df[col] = input_df[col].values

    X_scaled = scaler.transform(final_df)

    pred_log = model.predict(X_scaled)[0]

    final_price = np.expm1(pred_log)

    st.write("---")
    if final_price > 0:
        st.metric(label="Предсказанная стоимость автомобиля", value=f"{int(final_price):,} ₽")
    else:
        st.error("Проверьте вводимые параметры")