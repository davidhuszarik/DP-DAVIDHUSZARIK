import plotly.express as px
import streamlit as st


def render():
    st.header("4. Porovnanie Zložilostí a Dopad na Kryptografiu")
    alg_data = {
        "Algoritmus": [
            "Levieil-Fouque",
            "Bernstein-Lange",
            "Nový algoritmus (Krycie kódy)",
        ],
        "Časová zložitosť (log2)": [87.5, 85.7, 79.9],
        "Počet dopytov (log2)": [75.7, 68.6, 66.3],
        "Pamäťová zložitosť (log2)": [84.8, 77.6, 75.3],
    }
    df_alg = px.pd.DataFrame(alg_data)
    st.table(df_alg)