import streamlit as st
import pandas as pd
import plotly.express as px
import os
import matplotlib.pyplot as plt
import seaborn as sns

st.title("Аналитика данных и Интерпретация модели")

tab_eda, tab_weights = st.tabs(["Анализ", "Веса признаков"])

@st.cache_data
def clean_df():
    CARS_TRAIN = 'https://github.com/evgpat/datasets/raw/refs/heads/main/cars_train.csv'
    CARS_TEST = 'https://github.com/evgpat/datasets/raw/refs/heads/main/cars_test.csv'

    df_train = pd.read_csv(CARS_TRAIN)
    df_test = pd.read_csv(CARS_TEST)

    df = pd.concat([df_train, df_test])

    for col in ['engine', 'max_power', 'mileage']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.split().str[0]
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].fillna(df[col].median())

    df['name'] = df['name'].str.split().str[0]
    return df


df = clean_df()

# Вкладка с анализом 1
with tab_eda:
    st.header("Информация о датасете")

    m1, m2, m3 = st.columns(3)
    m1.metric("Всего машин в дф", f"{len(df):}")
    m2.metric("Средняя цена", f"{int(df['selling_price'].mean()): } ₽")
    m3.metric("Макс. мощность", f"{int(df['max_power'].max())} л.с.")

    st.write("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Распределение цен на автомобили")
        # Отсекаем топ 5% дорогих машин, чтобы график не растягивался
        limit_price = df['selling_price'].quantile(0.95)
        fig_hist = px.histogram(
            df[df['selling_price'] < limit_price],
            x="selling_price",
            nbins=40,
            labels={'selling_price': 'Стоимость'}
        )
        st.plotly_chart(fig_hist)


    with col2:
        st.subheader("Зависимость цены от года")
        fig_box = px.box(
            df[df['year'] >= 2000],
            x="year",
            y="selling_price",
            labels={'year': 'Год', 'selling_price': 'Цена'}
        )
        st.plotly_chart(fig_box, use_container_width=True)

    # хитмапа
    st.write("---")
    st.subheader("Матрица корреляции")

    import matplotlib.pyplot as plt
    import seaborn as sns

    corr_matrix = df.corr(numeric_only=True)

    fig, ax = plt.subplots(figsize=(12, 7))

    sns.heatmap(
        corr_matrix,
        linewidths=0.5,
        annot=True,
        cmap='viridis',
        linecolor="white",
        annot_kws={'size': 11}
    )

    plt.xticks(rotation=30, fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()

    st.pyplot(fig)

# Вкладка с весами 2
with tab_weights:
    st.header("Веса")

    weights_path = 'model_weights.csv'

    weights_df = pd.read_csv(weights_path)

    weights_df['Влияние'] = weights_df['Weight'].apply(
        lambda x: 'Увеличивает цену' if x > 0 else 'Снижает цену'
    )

    fig_weights = px.bar(
        weights_df,
        x='Weight',
        y='Feature',
        color='Влияние',
        orientation='h',
        labels={'Weight': 'Вес признака в модели', 'Feature': 'Признак'},
        color_discrete_map={'Увеличивает цену': '#2E7D32', 'Снижает цену': '#C62828'},
        height=max(400, len(weights_df) * 25)
    )

    fig_weights.update_layout(yaxis={'categoryorder': 'total ascending'}, margin=dict(l=150))

    st.plotly_chart(fig_weights, use_container_width=True)
