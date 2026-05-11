import streamlit as st

pg = st.navigation([
    st.Page("predict_page.py", title="Предсказание стоимости"),
    st.Page("eda_page.py", title="EDA")
])

pg.run()