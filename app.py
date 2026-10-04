import streamlit as st

from views import bkw_simulation, covering_codes, crypto_impact, lpn_intro

st.set_page_config(
    page_title="LPN & BKW Visualizer",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main {
        background-color: #0e1117;
    }
    div[data-testid="stSidebar"] {
        background-color: #161b22;
        border-right: 1px solid #30363d;
    }
    .css-17eq0hr, .stRadio > label {
        color: #c9d1d9 !important;
    }
    div[data-testid="stSidebar"] .stRadio > div {
        background-color: transparent;
        gap: 8px;
    }
    div[data-testid="stSidebar"] .stRadio > div > label {
        background-color: #21262d;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 10px 14px;
        margin-bottom: 4px;
        transition: all 0.2s ease-in-out;
        cursor: pointer;
        width: 100%;
    }
    div[data-testid="stSidebar"] .stRadio > div > label:hover {
        border-color: #58a6ff;
        background-color: #30363d;
    }
    div[data-testid="stSidebar"] .stRadio > div > label[data-baseweb="radio"] > div:first-child {
        display: none;
    }
    .nav-card {
        background: linear-gradient(135deg, #1f242d 0%, #0d1117 100%);
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 12px;
        margin-bottom: 15px;
    }
    .nav-header {
        font-size: 1.1rem;
        font-weight: 700;
        color: #58a6ff;
        margin-bottom: 5px;
    }
    </style>
""",
    unsafe_allow_html=True,
)

menu_options = [
    "Úvod do LPN",
    "Simulácia BKW"
]

with st.sidebar:

    selected_index = st.radio(
        "NAVIGÁCIA",
        range(len(menu_options)),
        format_func=lambda x: menu_options[x],
        label_visibility="visible",
    )

st.title("Edukačná aplikácia: LPN a BKW pomocou")

if selected_index == 0:
    lpn_intro.render()
elif selected_index == 1:
    bkw_simulation.render()
elif selected_index == 2:
    covering_codes.render()
elif selected_index == 3:
    crypto_impact.render()