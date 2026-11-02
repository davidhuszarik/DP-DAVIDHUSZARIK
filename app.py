import streamlit as st

from views import bkw_simulation, covering_codes, crypto_impact, lpn_intro

st.set_page_config(
    page_title="learnBKW | LPN & BKW Educational App",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    /* Bočný panel - tmavé pozadie a jemný okraj */
    section[data-testid="stSidebar"] {
        background-color: #0d1117 !important;
        border-right: 1px solid #30363d;
    }

    /* Hlavička aplikácie v bočnom paneli */
    .sidebar-brand {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 8px 4px 18px 4px;
        border-bottom: 1px solid #21262d;
        margin-bottom: 20px;
    }
    .brand-title {
        font-size: 1.25rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        color: #ffffff;
        margin: 0;
    }
    .brand-tag {
        font-size: 0.72rem;
        color: #58a6ff;
        font-family: monospace;
        background: rgba(88, 166, 255, 0.12);
        padding: 2px 7px;
        border-radius: 6px;
        border: 1px solid rgba(88, 166, 255, 0.3);
    }
    .menu-label {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #8b949e;
        font-weight: 700;
        margin-bottom: 10px;
        padding-left: 2px;
    }

    /* Skrytie predvoleného krúžku (radio circle) */
    div[data-testid="stSidebar"] div[data-testid="stRadio"] label > div:first-child {
        display: none !important;
    }

    /* Kartový dizajn tlačidiel menu */
    div[data-testid="stSidebar"] div[data-testid="stRadio"] label {
        display: flex !important;
        align-items: center !important;
        background: #161b22 !important;
        border: 1px solid #30363d !important;
        border-radius: 10px !important;
        padding: 12px 14px !important;
        margin-bottom: 8px !important;
        cursor: pointer !important;
        transition: all 0.22s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.15) !important;
        width: 100% !important;
    }

    /* Text vnútri karty */
    div[data-testid="stSidebar"] div[data-testid="stRadio"] label p {
        font-size: 0.93rem !important;
        font-weight: 500 !important;
        color: #c9d1d9 !important;
        margin: 0 !important;
    }

    /* Hover efekt (keď prejdete myšou) */
    div[data-testid="stSidebar"] div[data-testid="stRadio"] label:hover {
        background: #21262d !important;
        border-color: #58a6ff !important;
        transform: translateX(4px) !important;
    }
    div[data-testid="stSidebar"] div[data-testid="stRadio"] label:hover p {
        color: #ffffff !important;
    }

    /* AKTÍVNY STAV (vybraná karta) */
    div[data-testid="stSidebar"] div[data-testid="stRadio"] label:has(input:checked) {
        background: linear-gradient(135deg, rgba(88, 166, 255, 0.22) 0%, rgba(22, 27, 34, 0.95) 100%) !important;
        border-color: #58a6ff !important;
        border-left: 4px solid #58a6ff !important;
        box-shadow: 0 0 14px rgba(88, 166, 255, 0.22) !important;
    }
    div[data-testid="stSidebar"] div[data-testid="stRadio"] label:has(input:checked) p {
        color: #58a6ff !important;
        font-weight: 700 !important;
    }

    /* Pätička v sidebar */
    .sidebar-footer {
        margin-top: 35px;
        padding: 12px;
        border-radius: 8px;
        background: rgba(33, 38, 45, 0.6);
        border: 1px solid #21262d;
        color: #8b949e;
        font-size: 0.76rem;
        line-height: 1.4;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

menu_options = [
    "1. LPN problém",
    "2. BKW algoritmus",
]

with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-brand">
            <div>
                <div class="brand-title">🔐 learnBKW</div>
                <div class="brand-tag">v2.0 • Dávid Huszárik</div>
            </div>
        </div>
        <div class="menu-label">Navigácia modulov</div>
        """,
        unsafe_allow_html=True,
    )

    selected_index = st.radio(
        label="Vyberte modul:",
        options=range(len(menu_options)),
        format_func=lambda i: menu_options[i],
        label_visibility="collapsed",
    )

    st.markdown(
        """
        <div class="sidebar-footer">
            💡 <b>Tip pre výučbu:</b><br>
            Odporúčame prejsť moduly postupne.
        </div>
        """,
        unsafe_allow_html=True,
    )

st.title("learnBKW: Ako BKW krok za krokom poráža šum v LPN")

if selected_index == 0:
    lpn_intro.render()
elif selected_index == 1:
    bkw_simulation.render()
elif selected_index == 2:
    covering_codes.render()
elif selected_index == 3:
    crypto_impact.render()