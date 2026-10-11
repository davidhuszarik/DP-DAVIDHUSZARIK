import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

np.set_printoptions(threshold=np.inf, linewidth=np.inf)


@st.cache_data
def generate_lpn_instance(k_val, n_val, eta_val, seed=42):
    np.random.seed(seed)
    secret_x = np.random.randint(0, 2, size=k_val)
    matrix_G = np.random.randint(0, 2, size=(k_val, n_val))
    noise_e = (np.random.rand(n_val) < eta_val).astype(int)
    clean_z = (secret_x @ matrix_G) % 2
    noisy_z = (clean_z + noise_e) % 2
    return secret_x, matrix_G, noise_e, noisy_z, clean_z


def format_array(arr):
    return "".join(map(str, arr))


def gaussian_elimination_f2(A_in, b_in):
    A = A_in.copy() % 2
    b = b_in.copy() % 2
    rows, cols = A.shape

    steps = []
    pivot_cols = []
    r = 0

    for c in range(cols):
        if r >= rows:
            break

        pivot = -1
        for i in range(r, rows):
            if A[i, c] == 1:
                pivot = i
                break

        if pivot == -1:
            continue

        if pivot != r:
            A[[r, pivot]] = A[[pivot, r]]
            b[r], b[pivot] = b[pivot], b[r]
            steps.append(f"Zmena riadkov R_{r + 1} ↔ R_{pivot + 1}")

        pivot_cols.append((r, c))

        for i in range(rows):
            if i != r and A[i, c] == 1:
                A[i] = (A[i] ^ A[r])
                b[i] = (b[i] ^ b[r])
                steps.append(f"Riadková operácia: R_{i + 1} = R_{i + 1} ⊕ R_{r + 1}")

        r += 1

    x = np.zeros(cols, dtype=int)
    for row, col in reversed(pivot_cols):
        x[col] = b[row]
        for c in range(col + 1, cols):
            x[col] ^= (A[row, c] & x[c])

    return x, A, b, steps


def render():
    st.markdown(
        """
        <style>
        div[data-testid="stMetricValue"] {
            font-size: 1.1rem !important;
            word-break: break-all !important;
            white-space: normal !important;
        }
        div[data-testid="stCodeBlock"] pre {
            word-break: break-all !important;
            white-space: pre-wrap !important;
        }
        </style>
    """,
        unsafe_allow_html=True,
    )

    st.header("1. Základy problému LPN (Learning Parity with Noise)")

    st.markdown(
        """
    Problém **LPN** spája lineárnu algebru nad telesom $\mathbb{F}_2$ s pravdepodobnostným šumom.
    """
    )

    st.divider()
    st.subheader("Interaktívny generátor LPN inštancie")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        k_val = st.number_input(
            "Dĺžka kľúča (k)",
            min_value=4,
            max_value=32,
            value=8,
            help="Počet bitov tajného kľúča x, ktorý sa pokúšame zistiť.",
        )
    with col2:
        n_val = st.number_input(
            "Počet vzoriek (n)",
            min_value=5,
            max_value=100,
            value=16,
            help="Počet stĺpcov v matici G (počet dostupných zašumených rovníc).",
        )
    with col3:
        eta_val = st.slider(
            "Chybovosť (η)",
            0.0,
            0.49,
            0.125,
            step=0.005,
            help="Pravdepodobnosť, že konkrétny bit šumu e bude 1 (nastane chyba).",
        )
    with col4:
        seed_val = st.number_input(
            "Seed pre náhodnosť",
            min_value=1,
            max_value=999,
            value=42,
            help="Semienko pre generátor náhodných čísel na reprodukovateľnosť výsledkov.",
        )

    secret_x, matrix_G, noise_e, noisy_z, clean_z = generate_lpn_instance(
        k_val, n_val, eta_val, seed=seed_val
    )

    st.divider()
    st.subheader("Anatómia LPN rovnice: z = xG + e")

    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    with m_col1:
        st.metric(
            "Tajný kľúč x",
            format_array(secret_x),
            help="Tajný vektor x dĺžky k, ktorý je v reálnom scenári neznámy.",
        )
    with m_col2:
        st.metric(
            "Čistý súčin xG",
            format_array(clean_z),
            help="Výsledok skalárneho súčinu kľúča x s maticou G bez pridaného šumu.",
        )
    with m_col3:
        st.metric(
            "Šum e (podiel 1)",
            f"{np.mean(noise_e):.3f}",
            help="Skutočne nameraný podiel chybových bitov 1 vo vygenerovanom vektore e.",
        )
    with m_col4:
        st.metric(
            "Pozorované z",
            format_array(noisy_z),
            help="Výsledný vektor pozorovaní z = xG + e, ktorý získava útočník.",
        )

    st.markdown(
        "#### Detail konkrétnej vzorky $z_i = \\langle x, g_i \\rangle + e_i \\pmod 2$"
    )
    selected_col = st.slider(
        "Vyberte stĺpec (vzorku i):",
        0,
        n_val - 1,
        0,
        help="Posuvník na výber konkrétneho stĺpca z matice G pre zobrazenie detailného výpočtu a jeho zvýraznenie vo vizualizáciách.",
    )

    g_i = matrix_G[:, selected_col]
    dot_prod = int((secret_x @ g_i) % 2)
    e_i = int(noise_e[selected_col])
    z_i = int(noisy_z[selected_col])

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.code(f"x = {format_array(secret_x)}")
        st.caption("Tajný kľúč")
    with c2:
        st.code(f"g_{selected_col} = {format_array(g_i)}")
        st.caption(f"{selected_col}. stĺpec matice G")
    with c3:
        st.code(f"x · g_{selected_col} = {dot_prod}")
        st.caption("Skalárny súčin mod 2")
    with c4:
        st.code(f"e_{selected_col} = {e_i}")
        st.caption("Chybový bit e")
    with c5:
        st.code(f"z_{selected_col} = {z_i}")
        st.caption("Výsledný bit z")

    products = [f"({x_b}·{g_b})" for x_b, g_b in zip(secret_x, g_i)]
    calc_steps_str = " ⊕ ".join(products)
    eval_steps_str = " ⊕ ".join([str(x_b * g_b) for x_b, g_b in zip(secret_x, g_i)])

    status_e = (
        "pôvodný bit sa PREVRÁTIL (nastala chyba)"
        if e_i == 1
        else "pôvodný bit zostal NEZMENENÝ (bez chyby)"
    )

    st.info(
        f"### Kompletný príklad výpočtu pre vzorku $i = {selected_col}$:\n\n"
        f"**1. Krok: Výpočet čistého skalárneho súčinu $\\langle x, g_{{{selected_col}}} \\rangle$**\n"
        f"$$\\langle x, g_{{{selected_col}}} \\rangle = {calc_steps_str}$$\n"
        f"$$\\langle x, g_{{{selected_col}}} \\rangle = {eval_steps_str} = {dot_prod} \\pmod 2$$\n\n"
        f"**2. Krok: Pripočítanie šumu $e_{{{selected_col}}}$ mod 2**\n"
        f"$$z_{{{selected_col}}} = (x \\cdot g_{{{selected_col}}}) \\oplus e_{{{selected_col}}} = {dot_prod} \\oplus {e_i} = \\mathbf{{{z_i}}}$$\n\n"
        f"*Výsledok:* Šum pre túto vzorku je $e_{{{selected_col}}} = {e_i}$, takže {status_e}."
    )

    col_names = [
        f"({c})" if c == selected_col else f"g_{c}" for c in range(n_val)
    ]
    df_G = pd.DataFrame(
        matrix_G,
        index=[f"x_{r}" for r in range(k_val)],
        columns=col_names,
    )

    target_col_name = f"({selected_col})"

    def highlight_selected_column(s):
        if s.name == target_col_name:
            return ["background-color: #1f6feb; color: white; font-weight: bold"] * len(s)
        return [""] * len(s)

    styled_df = df_G.style.apply(highlight_selected_column, axis=0)

    st.markdown("#### Celá matica $G$ s vyznačením vybranej vzorky")
    st.caption(
        f"Vybraný stĺpec vzorky {selected_col} je zvýraznený modrou farbou a v hlavičke označený ako **({selected_col})**."
    )
    st.dataframe(styled_df, use_container_width=True)

    st.divider()
    st.subheader("Pokročilá vizualizácia matice $G$ a šumu $e$")

    text_matrix = matrix_G.astype(str) if n_val <= 30 and k_val <= 16 else None

    fig_g = px.imshow(
        matrix_G,
        text_auto=True if text_matrix is not None else False,
        color_continuous_scale=["#161b22", "#388bfd"],
        labels=dict(x="Stĺpec / Vzorka (i)", y="Riadok / Bit kľúča (k)"),
        aspect="auto",
    )
    fig_g.add_shape(
        type="rect",
        x0=selected_col - 0.5,
        x1=selected_col + 0.5,
        y0=-0.5,
        y1=k_val - 0.5,
        line=dict(color="#f0883e", width=3),
    )
    fig_g.update_layout(
        coloraxis_showscale=False,
        height=320,
        margin=dict(l=10, r=10, t=30, b=10),
    )

    bar_colors = ["#f85149" if val == 1 else "#238636" for val in noise_e]
    bar_colors[selected_col] = "#f0883e" if noise_e[selected_col] == 1 else "#d29922"

    fig_e = go.Figure()
    fig_e.add_trace(
        go.Bar(
            x=list(range(n_val)),
            y=noise_e,
            marker_color=bar_colors,
            name="Šum e_i",
        )
    )
    fig_e.add_hline(
        y=eta_val,
        line_dash="dash",
        line_color="#a5d6ff",
        annotation_text=f"Očakávaná chybovosť η = {eta_val:.3f}",
        annotation_position="top right",
    )
    fig_e.update_layout(
        xaxis_title="Index vzorky (i)",
        yaxis_title="Chyba (0 = NIE, 1 = ÁNO)",
        height=320,
        yaxis=dict(tickvals=[0, 1]),
        margin=dict(l=10, r=10, t=30, b=10),
    )

    tab1, tab2, tab3 = st.tabs(
        ["⚖Vedľa seba", "Matica G (Detail)", "Rozloženie šumu (Detail)"]
    )

    with tab1:
        col_fig1, col_fig2 = st.columns(2)
        with col_fig1:
            st.markdown(
                f"**Matica $G$ ($k \\times n$) — vybraná vzorka {selected_col} je v oranžovom ráme**",
                help="Svetlomodrá = 1, Čierna = 0. Oranžový obdĺžnik vyznačuje aktuálne vybranú vzorku z posuvníka.",
            )
            st.plotly_chart(fig_g, use_container_width=True, key="fig_g_tab1")
        with col_fig2:
            st.markdown(
                f"**Rozloženie šumu $e$ — vybraná vzorka {selected_col} je podsvietená**",
                help="Červená = chyba e_i=1, Zelená = bez chyby e_i=0. Žltá/Oranžová = vybraná vzorka z posuvníka.",
            )
            st.plotly_chart(fig_e, use_container_width=True, key="fig_e_tab1")

    with tab2:
        st.markdown("**Detailná štruktúra generujúcej matice $G$**")
        st.plotly_chart(fig_g, use_container_width=True, key="fig_g_tab2")

    with tab3:
        st.markdown("**Detailné rozloženie chýb vo vektore $e$**")
        st.plotly_chart(fig_e, use_container_width=True, key="fig_e_tab3")

    st.divider()
    st.subheader("Prečo je LPN ťažký problém?")

    col_easy, col_hard = st.columns(2)
    with col_easy:
        st.markdown("**1. Bez šumu ($\eta = 0$): Ľahké**")
        st.write(
            "Rovnica $z = xG$ sa dá vyriešiť v polynomiálnom čase pomocou Gaussovej eliminácie nad $\mathbb{F}_2$."
        )
        if st.button(
                "⚡ Vyriešiť $xG = z$ bez šumu",
                help="Spustí algebraické riešenie sústavy rovníc bez šumu krok za krokom.",
        ):
            x_rec, A_red, b_red, steps = gaussian_elimination_f2(matrix_G.T, clean_z)
            st.success(f"Vypočítaný kľúč: **{format_array(x_rec)}** (Zhoduje sa s tajným x!)")

            with st.expander("📝 Zobraziť kompletný postup Gaussovej eliminácie nad F_2", expanded=True):
                st.write("**A. Počiatočná rozšírená matica sústavy $[G^T | z]$:**")
                init_df = pd.DataFrame(
                    np.column_stack([matrix_G.T, clean_z]),
                    columns=[f"x_{r}" for r in range(k_val)] + ["z_clean"]
                )
                st.dataframe(init_df, use_container_width=True)

                st.write("**B. Vykonané riadkové XOR operácie:**")
                if steps:
                    for idx, step in enumerate(steps[:10], 1):
                        st.caption(f"{idx}. {step}")
                    if len(steps) > 10:
                        st.caption(f"... a ďalších {len(steps) - 10} operácií.")
                else:
                    st.caption("Matica bola už v úvodnom tvare.")

                st.write("**C. Výsledný odstupňovaný tvar $[G_{red}^T | z_{red}]$:**")
                red_df = pd.DataFrame(
                    np.column_stack([A_red, b_red]),
                    columns=[f"x_{r}" for r in range(k_val)] + ["z_red"]
                )
                st.dataframe(red_df, use_container_width=True)

    with col_hard:
        st.markdown("**2. So šumom ($\eta > 0$): NP-ťažké**")
        st.write(
            "Prítomnosť šumu robí z LPN náročný problém dekódovania náhodných lineárnych kódov."
        )
        if st.button(
                "Pokus o vyriešenie šumového $z = xG + e$",
                help="Ukáže zlyhanie Gaussovej eliminácie pri prítomnosti šumu.",
        ):
            x_rec_noisy, A_red, b_red, steps = gaussian_elimination_f2(matrix_G.T, noisy_z)
            st.error(
                f"Zlyhanie! Vypočítal nesprávny kľúč: **{format_array(x_rec_noisy)}** (Nezhoduje sa s tajným x!)"
            )

            with st.expander("📝 Postup eliminácie so zašumeným vektorom z", expanded=True):
                st.write("**Rozšírená matica so šumom $[G^T | z_{noisy}]$:**")
                noisy_df = pd.DataFrame(
                    np.column_stack([matrix_G.T, noisy_z]),
                    columns=[f"x_{r}" for r in range(k_val)] + ["z_noisy"]
                )
                st.dataframe(noisy_df, use_container_width=True)
                st.warning(
                    "Prítomnosť chybových bitov e_i spôsobila propagáciu chýb počas XOR operácií, čím eliminovaný tvar vrátil zlé hodnoty."
                )

    st.divider()
    st.subheader("Vyskúšajte si uhádnuť tajný kľúč")
    user_guess_str = st.text_input(
        f"Zadajte {k_val} bitov (napr. '1010...'):",
        value="0" * k_val,
        help="Zadajte váš tip pre tajný kľúč x bez medzier.",
    )

    try:
        user_guess = np.array([int(b) for b in user_guess_str.strip()])
        if len(user_guess) == k_val:
            calc_z = (user_guess @ matrix_G) % 2
            diff_e = noisy_z ^ calc_z
            observed_eta = np.mean(diff_e)

            if np.array_equal(user_guess, secret_x):
                st.success(
                    f"Správne! Uhádli ste tajný kľúč {format_array(secret_x)}. Nameraný šum: {observed_eta:.3f}"
                )
            else:
                st.warning(
                    f"Nesprávny kľúč. Nameraná chybovosť pre váš tip: {observed_eta:.3f} (Očakávaná $\eta \\approx {eta_val}$)."
                )
        else:
            st.error(f"Zadajte presne {k_val} bitov.")
    except ValueError:
        st.error("Zadávajte iba bity 0 a 1.")