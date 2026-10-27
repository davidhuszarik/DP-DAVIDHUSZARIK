import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# ==============================================================================
# 1. TEORETICKÉ VÝPOČTY PODĽA PRÁCE (LEMMA 3 A HOEFFDINGOVA NEROVNOSŤ)
# ==============================================================================
def compute_bkw_theory(a: int, eta: float, delta: float = 0.05):
    """Vypočíta teoretické parametre BKW a minimálny potrebný počet behov N_min:

    - bias epsilon = 1 - 2*eta
    - akumulovaný počet rovníc: 2^(a-1)
    - odchýlka gamma = 1/2 * (1 - 2*eta)^(2^(a-1))
    - N_min = ln(1 / delta) / (2 * gamma^2)
    """
    initial_bias = 1.0 - 2.0 * eta
    accumulated_eqs = 2 ** (a - 1)
    theoretical_bias = initial_bias ** accumulated_eqs
    gamma = 0.5 * theoretical_bias
    p_success_single = 0.5 + gamma

    if gamma > 1e-12:
        n_min = int(np.ceil(np.log(1.0 / delta) / (2.0 * (gamma ** 2))))
    else:
        n_min = float("inf")

    return {
        "initial_bias": initial_bias,
        "accumulated_eqs": accumulated_eqs,
        "theoretical_bias": theoretical_bias,
        "gamma": gamma,
        "p_success_single": p_success_single,
        "n_min": n_min,
    }


# ==============================================================================
# 2. RÝCHLE ŠTATISTICKÉ JADRO PRE VEĽKÉ MNOŽSTVÁ BEHOV (N až do 10 000)
# ==============================================================================
def execute_fast_single_run(c_true, a, b, n_samples, eta, run_seed):
    """Rýchla realizácia 1 behu bez ukladania ťažkých matíc do pamäte."""
    np.random.seed(run_seed)
    k = a * b

    X = np.random.randint(0, 2, size=(n_samples, k))
    e = (np.random.rand(n_samples) < eta).astype(int)
    ell = (X @ c_true + e) % 2

    for step in range(1, a):
        target_block = a - step
        start_col = target_block * b
        end_col = (target_block + 1) * b

        partitions = {}
        for row_idx in range(len(X)):
            pattern = tuple(X[row_idx, start_col:end_col].tolist())
            partitions.setdefault(pattern, []).append(row_idx)

        X_next, ell_next, e_next = [], [], []
        for _, indices in partitions.items():
            if len(indices) >= 2:
                pivot_idx = indices[0]
                for other_idx in indices[1:]:
                    X_next.append(X[other_idx] ^ X[pivot_idx])
                    ell_next.append(ell[other_idx] ^ ell[pivot_idx])
                    e_next.append(e[other_idx] ^ e[pivot_idx])

        if not X_next:
            return False, None, None, 0

        X = np.array(X_next, dtype=int)
        ell = np.array(ell_next, dtype=int)
        e = np.array(e_next, dtype=int)

    e_1 = np.zeros(k, dtype=int)
    e_1[0] = 1

    matches = np.where((X == e_1).all(axis=1))[0]
    if len(matches) > 0:
        match_idx = matches[0]
        return True, int(ell[match_idx]), int(e[match_idx]), len(matches)
    return False, None, None, 0


# ==============================================================================
# 3. HĹBKOVÝ GENERÁTOR PRE 1 VYBRANÝ BEH (NA POŽIADANIE)
# ==============================================================================
def generate_detailed_run_data(c_true, a, b, n_samples, eta, run_seed):
    """Vygeneruje kompletnú históriu, matice, triedy a XOR log pre zvolený beh."""
    np.random.seed(run_seed)
    k = a * b

    X_0 = np.random.randint(0, 2, size=(n_samples, k))
    e_0 = (np.random.rand(n_samples) < eta).astype(int)
    ell_0 = (X_0 @ c_true + e_0) % 2
    lineage_0 = [[i] for i in range(n_samples)]

    history = [
        {
            "step": 0,
            "sample_type": "0-vzorka (Východiskový stav)",
            "X": X_0.copy(),
            "ell": ell_0.copy(),
            "e": e_0.copy(),
            "lineage": lineage_0,
            "elim_block_idx": None,
            "elim_bit_range": None,
            "classes_info": [],
            "operations_log": [],
            "pivots_discarded": 0,
        }
    ]

    X_curr, ell_curr, e_curr, lineage_curr = (
        X_0.copy(),
        ell_0.copy(),
        e_0.copy(),
        lineage_0,
    )

    for step in range(1, a):
        target_block = a - step
        start_col = target_block * b
        end_col = (target_block + 1) * b

        partitions = {}
        for row_idx in range(len(X_curr)):
            pattern = tuple(X_curr[row_idx, start_col:end_col].tolist())
            partitions.setdefault(pattern, []).append(row_idx)

        X_next, ell_next, e_next, lineage_next = [], [], [], []
        operations_log = []
        classes_info = []
        pivots_discarded = 0

        for pattern, indices in partitions.items():
            classes_info.append(
                {
                    "Vzor bloku": "".join(map(str, pattern)),
                    "Počet vektorov": len(indices),
                    "Vzniknuté vzorky": max(0, len(indices) - 1),
                }
            )

            if len(indices) >= 2:
                pivot_idx = indices[0]
                pivots_discarded += 1

                for other_idx in indices[1:]:
                    res_vec = (X_curr[other_idx] ^ X_curr[pivot_idx]).tolist()
                    res_ell = int(ell_curr[other_idx] ^ ell_curr[pivot_idx])
                    res_e = int(e_curr[other_idx] ^ e_curr[pivot_idx])
                    res_lin = lineage_curr[other_idx] + lineage_curr[pivot_idx]

                    X_next.append(res_vec)
                    ell_next.append(res_ell)
                    e_next.append(res_e)
                    lineage_next.append(res_lin)

                    if len(operations_log) < 50:
                        operations_log.append(
                            {
                                "Vzor bloku": "".join(map(str, pattern)),
                                "Pivot (#)": pivot_idx,
                                "Cieľ (#)": other_idx,
                                "e_pivot": int(e_curr[pivot_idx]),
                                "e_cieľ": int(e_curr[other_idx]),
                                "e_výsledok (XOR)": res_e,
                                "Výsledný vektor": "".join(map(str, res_vec)),
                            }
                        )
            elif len(indices) == 1:
                pivots_discarded += 1

        if not X_next:
            break

        X_curr = np.array(X_next, dtype=int)
        ell_curr = np.array(ell_next, dtype=int)
        e_curr = np.array(e_next, dtype=int)
        lineage_curr = lineage_next

        history.append(
            {
                "step": step,
                "sample_type": f"{step}-vzorka",
                "X": X_curr.copy(),
                "ell": ell_curr.copy(),
                "e": e_curr.copy(),
                "lineage": lineage_curr,
                "elim_block_idx": target_block,
                "elim_bit_range": (start_col, end_col),
                "classes_info": classes_info,
                "operations_log": operations_log,
                "pivots_discarded": pivots_discarded,
            }
        )

    e_1 = np.zeros(k, dtype=int)
    e_1[0] = 1
    matches = (
        np.where((history[-1]["X"] == e_1).all(axis=1))[0]
        if len(history[-1]["X"]) > 0
        else []
    )

    found = len(matches) > 0
    vote = int(history[-1]["ell"][matches[0]]) if found else None
    err = int(history[-1]["e"][matches[0]]) if found else None

    return {
        "history": history,
        "found": found,
        "vote": vote,
        "err": err,
        "matches_count": len(matches),
        "matches_indices": matches.tolist() if found else [],
    }


# ==============================================================================
# 4. HROMADNÝ SIMULÁTOR S VYROVNÁVACOU PAMÄŤOU
# ==============================================================================
@st.cache_data
def run_large_scale_simulation(a, b, n_samples, eta, n_runs, base_seed):
    k = a * b
    np.random.seed(base_seed)
    c_true = np.random.randint(0, 2, size=k)

    votes_0 = 0
    votes_1 = 0
    successful_runs = 0

    runs_summary = []
    cum_votes_0 = []
    cum_votes_1 = []
    cum_ratio_true = []

    for i in range(1, n_runs + 1):
        run_seed = base_seed * 10000 + i
        found, vote, err_val, count_e1 = execute_fast_single_run(
            c_true, a, b, n_samples, eta, run_seed
        )

        if found:
            successful_runs += 1
            if vote == 0:
                votes_0 += 1
            else:
                votes_1 += 1

        total_valid = votes_0 + votes_1
        ratio_true = (
            (votes_0 if c_true[0] == 0 else votes_1) / total_valid
            if total_valid > 0
            else 0.5
        )

        cum_votes_0.append(votes_0)
        cum_votes_1.append(votes_1)
        cum_ratio_true.append(ratio_true)

        runs_summary.append(
            {
                "run_idx": i,
                "found": found,
                "vote": vote,
                "err": err_val,
                "matches_count": count_e1,
                "is_correct": (found and vote == c_true[0]),
                "cum_0": votes_0,
                "cum_1": votes_1,
            }
        )

    return c_true, runs_summary, cum_votes_0, cum_votes_1, cum_ratio_true


# ==============================================================================
# 5. HLAVNÉ POUŽÍVATEĽSKÉ ROZHRANIE
# ==============================================================================
def render():
    st.header("🛡️ BKW Algoritmus: Väčšinové Hlasovanie & Teoretické Medze")

    # Konfigurácia parametrov
    col_c1, col_c2, col_c3, col_c4 = st.columns(4)
    with col_c1:
        a = st.number_input("Počet blokov (a)", min_value=2, max_value=5, value=3)
        b = st.number_input("Šírka bloku (b)", min_value=1, max_value=3, value=2)
        k = int(a * b)
        st.caption(f"Dimenzia kľúča: **k = {k} bitov**")

    rec_s = int(a * (2 ** b))
    with col_c2:
        n_samples = st.number_input(
            "Vzorky na 1 beh (n)",
            min_value=rec_s,
            max_value=2000,
            value=max(rec_s * 4, 60),
            help=f"Teoretické minimum Lemmy 4: a · 2^b = {rec_s}",
        )
        st.caption(f"Odporúčané minimum na beh: $s = {rec_s}$")

    with col_c3:
        eta = st.slider(
            "Chybovosť šumu (η)",
            min_value=0.01,
            max_value=0.45,
            value=0.10,
            step=0.01,
        )
        desired_confidence = st.selectbox(
            "Požadovaná istota (1 - δ)",
            [0.95, 0.99, 0.999],
            index=0,
            help="Pravdepodobnosť, že väčšinové hlasovanie určí bit správne.",
        )
        delta = 1.0 - desired_confidence

    theory = compute_bkw_theory(a, eta, delta)
    n_min_theor = theory["n_min"]

    with col_c4:
        seed = st.number_input(
            "Základný Seed generátora", min_value=1, max_value=9999, value=42
        )
        st.caption(f"Teoretické $N_{{\\min}}$ (istota {desired_confidence * 100:.1f} %):")
        if n_min_theor != float("inf"):
            st.markdown(f"### **{n_min_theor} behov**")
        else:
            st.warning("Príliš nízky bias")

    st.markdown("#### ⚙️ Rozsah simulácie väčšinového hlasovania")
    col_n1, col_n2 = st.columns([3, 1])
    with col_n1:
        default_runs = (
            min(n_min_theor, 300)
            if (n_min_theor != float("inf") and n_min_theor > 0)
            else 100
        )
        n_runs = st.number_input(
            "Celkový počet opakovaní / behov (N):",
            min_value=5,
            max_value=5000,
            value=int(default_runs),
            step=10,
            help="V každom behu orákulum vygeneruje úplne nové vzorky pre ten istý kľúč c.",
        )
    with col_n2:
        st.write("")
        st.write("")
        if n_min_theor != float("inf") and n_min_theor <= 5000:
            st.info(f"Odporúčané $N \\ge {n_min_theor}$")

    c_true, runs_summary, cum_0, cum_1, cum_ratio = run_large_scale_simulation(
        a, b, n_samples, eta, n_runs, seed
    )
    true_c1 = int(c_true[0])

    votes_0_total = cum_0[-1]
    votes_1_total = cum_1[-1]
    total_valid = votes_0_total + votes_1_total
    winner = (
        0
        if votes_0_total > votes_1_total
        else (1 if votes_1_total > votes_0_total else -1)
    )
    is_winner_correct = winner == true_c1
    hoeffding_error_bound = (
        np.exp(-2 * total_valid * (theory["gamma"] ** 2))
        if total_valid > 0
        else 1.0
    )

    st.divider()
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Skutočný kľúč c", f"{list(c_true)}", f"1. bit c₁ = {true_c1}")
    with m2:
        st.metric(
            "Výsledok väčšiny (ĉ₁)",
            f"{winner}" if winner != -1 else "Remíza",
            "Správne odhalený ✅" if is_winner_correct else "Chyba ❌",
        )
    with m3:
        st.metric(
            "Pomer platných hlasov [0 : 1]",
            f"{votes_0_total} : {votes_1_total}",
            f"Z {total_valid} úspešných behov",
        )
    with m4:
        st.metric(
            "Hoeffding P(chyba väčšiny)",
            f"{hoeffding_error_bound:.4e}",
            f"Cieľová δ = {delta:.3f}",
        )

    with st.expander(
            "📐 Teoretický rozbor a vzorce z práce (kliknite pre zobrazenie)",
            expanded=False,
    ):
        st.markdown(
            f"""
        1. **Piling-up Lemma (Lemma 3):**
           Každý beh vyjadrí $e_1$ ako súčet $s = 2^{{a-1}} = 2^{{{a - 1}}} = {theory['accumulated_eqs']}$ pôvodných rovníc.
           Akumulovaný bias je:
           $$\\epsilon_{{\\text{{final}}}} = (1 - 2\\eta)^{{2^{{a-1}}}} = ({theory['initial_bias']:.2f})^{{{theory['accumulated_eqs']}}} = {theory['theoretical_bias']:.4e}$$

        2. **Pravdepodobnosť úspechu v jednom behu:**
           $$P(\\text{{úspech}}) = \\frac{{1}}{{2}} + \\gamma = \\frac{{1}}{{2}} + \\frac{{1}}{{2}}\\epsilon_{{\\text{{final}}}} = {theory['p_success_single'] * 100:.2f} \\%$$

        3. **Hoeffdingova nerovnosť a minimálny počet behov ($N_{{\\min}}$):**
           Pravdepodobnosť, že väčšinové hlasovanie zlyhá po $N$ behoch, je zhora ohraničená:
           $$P(\\text{{chyba}}) \\le e^{{-2N\\gamma^2}} \\le \\delta \\implies N \\ge \\frac{{\\ln(1/\\delta)}}{{2\\gamma^2}} = \\mathbf{{{n_min_theor}\\text{{ behov}}}}$$
        """
        )

    st.divider()

    # SEKCIA 1: GRAFY HLASOVANIA
    st.subheader(
        f"📊 1. Priebeh Väčšinového Hlasovania v Čase (N = {n_runs} behov)"
    )

    g_col1, g_col2 = st.columns(2)
    with g_col1:
        fig_votes = go.Figure()
        fig_votes.add_trace(
            go.Scatter(
                y=cum_0,
                mode="lines",
                name="Hlasy pre 0",
                line=dict(color="#1f77b4", width=2),
            )
        )
        fig_votes.add_trace(
            go.Scatter(
                y=cum_1,
                mode="lines",
                name="Hlasy pre 1",
                line=dict(color="#ff7f0e", width=2),
            )
        )
        fig_votes.update_layout(
            title="Kumulatívny súčet hlasov",
            xaxis_title="Číslo behu (nezávislé orákulum)",
            yaxis_title="Celkový počet hlasov",
            height=320,
            margin=dict(l=20, r=20, t=35, b=20),
        )
        st.plotly_chart(fig_votes, use_container_width=True)

    with g_col2:
        fig_ratio = go.Figure()
        fig_ratio.add_trace(
            go.Scatter(
                y=cum_ratio,
                mode="lines",
                name="Podiel správneho bitu",
                line=dict(color="#2ca02c", width=2),
            )
        )
        fig_ratio.add_hline(
            y=0.5,
            line_dash="dash",
            line_color="red",
            annotation_text="50 % (Hranica náhody)",
        )
        fig_ratio.add_hline(
            y=theory["p_success_single"],
            line_dash="dot",
            line_color="blue",
            annotation_text=f"Teoretická P = {theory['p_success_single']:.3f}",
        )
        fig_ratio.update_layout(
            title=f"Podiel správneho bitu (c₁ = {true_c1})",
            xaxis_title="Číslo behu",
            yaxis_title="Podiel hlasov",
            yaxis_range=[0, 1],
            height=320,
            margin=dict(l=20, r=20, t=35, b=20),
        )
        st.plotly_chart(fig_ratio, use_container_width=True)

    df_summary = pd.DataFrame(runs_summary)
    with st.expander(
            f"📋 Tabuľka výsledkov všetkých {n_runs} behov (kliknite pre zobrazenie)"
    ):
        st.dataframe(
            df_summary[
                [
                    "run_idx",
                    "found",
                    "matches_count",
                    "vote",
                    "err",
                    "is_correct",
                    "cum_0",
                    "cum_1",
                ]
            ],
            height=250,
            use_container_width=True,
        )

    st.divider()

    # SEKCIA 2: DETAILNÝ INŠPEKTOR VYBRANÉHO BEHU
    st.subheader("🔍 2. Hĺbkový Inšpektor Ľubovoľného Behu")

    selected_run_idx = st.number_input(
        f"Zadajte číslo behu na preskúmanie (1 až {n_runs}):",
        min_value=1,
        max_value=n_runs,
        value=1,
        step=1,
    )

    run_seed = seed * 10000 + selected_run_idx
    chosen_run_data = generate_detailed_run_data(
        c_true, a, b, n_samples, eta, run_seed
    )
    run_history = chosen_run_data["history"]

    c_run1, c_run2, c_run3, c_run4 = st.columns(4)
    with c_run1:
        st.info(f"**Vybraný: Beh #{selected_run_idx}**")
    with c_run2:
        st.metric(
            "Nájdený bázový vektor e₁",
            "Áno ✅" if chosen_run_data["found"] else "Nie ❌",
            f"Vyskytol sa {chosen_run_data['matches_count']}×",
        )
    with c_run3:
        st.metric(
            "Odovzdaný hlas (ℓ)",
            (
                f"{chosen_run_data['vote']}"
                if chosen_run_data["found"]
                else "Žiadny"
            ),
            f"Skutočný bit c₁ = {true_c1}",
        )
    with c_run4:
        st.metric(
            "Kumulovaná chyba (e)",
            f"{chosen_run_data['err']}" if chosen_run_data["found"] else "—",
            "0 = Správny | 1 = Prevrátený",
        )

    # SEKCIA 3: DETAIL VŠETKÝCH KROKOV VO VYBRANOM BEHU
    st.markdown("---")
    st.markdown(
        f"#### 🪜 Kroky eliminácie v rámci Behu #{selected_run_idx} (Lemma 4)"
    )

    max_steps_in_run = len(run_history) - 1
    selected_step = st.slider(
        f"Vyberte krok eliminácie v rámci Behu #{selected_run_idx}:",
        min_value=0,
        max_value=max_steps_in_run,
        value=max_steps_in_run,
        format="Krok %d",
    )

    step_data = run_history[selected_step]
    X_step = step_data["X"]

    st.markdown(
        f"**Aktuálny stav:** `{step_data['sample_type']}` | Počet prežívajúcich vektorov: **{len(X_step)}**"
    )

    col_mat, col_cls = st.columns([3, 2])
    with col_mat:
        st.markdown(
            f"##### Matica $X$ v Kroku {selected_step} (Zobrazených max 35 riadkov)"
        )
        disp_rows = min(len(X_step), 35)
        fig_mat = px.imshow(
            X_step[:disp_rows, :],
            color_continuous_scale=[[0, "#ffffff"], [1, "#1f77b4"]],
            labels=dict(x="Stĺpec (Bit)", y="Riadok (Vektor)", color="Bit"),
        )
        fig_mat.update_layout(
            coloraxis_showscale=False,
            height=320,
            margin=dict(l=10, r=10, t=20, b=10),
        )

        for blk in range(1, a):
            fig_mat.add_vline(
                x=blk * b - 0.5,
                line_width=2,
                line_dash="dash",
                line_color="red",
            )
        st.plotly_chart(fig_mat, use_container_width=True)

    with col_cls:
        st.markdown(f"##### Triedy ekvivalencie v Kroku {selected_step}")
        if step_data["classes_info"]:
            st.dataframe(
                pd.DataFrame(step_data["classes_info"]),
                height=250,
                use_container_width=True,
            )
            st.caption(
                f"Zahodených pivotov v tomto kroku: **{step_data['pivots_discarded']}**"
            )
        else:
            st.info("Krok 0: Východiskový stav pred elimináciou.")

    if step_data["operations_log"]:
        st.markdown(
            f"##### 🔬 Presný Log XOR Operácií v Kroku {selected_step}"
        )
        st.caption(
            "Ukážka sčítania s pivotom a XOR-ovania chybových bitov:"
        )
        st.dataframe(
            pd.DataFrame(step_data["operations_log"]),
            height=200,
            use_container_width=True,
        )

    if selected_step == max_steps_in_run and chosen_run_data["found"]:
        st.markdown(
            f"##### 🧬 Rodokmeň Bázového Vektora $e_1$ v Behu #{selected_run_idx}"
        )
        first_match_idx = chosen_run_data["matches_indices"][0]
        src_row_ids = step_data["lineage"][first_match_idx]

        orig_X = run_history[0]["X"][src_row_ids]
        orig_ell = run_history[0]["ell"][src_row_ids]
        orig_e = run_history[0]["e"][src_row_ids]

        df_lineage = pd.DataFrame(orig_X)
        df_lineage["Označenie (ℓ)"] = orig_ell
        df_lineage["Pôvodná chyba (e)"] = orig_e
        df_lineage.index = [f"Vzorka #{idx}" for idx in src_row_ids]

        st.dataframe(df_lineage, use_container_width=True)
        st.success(
            f"Tento vektor $e_1$ vznikol ako XOR súčet presne **{len(src_row_ids)} pôvodných vzoriek** "
            f"(teoreticky $2^{{a-1}} = {2 ** (a - 1)}$). "
            f"Výsledná kumulovaná chyba: $\\sum e_i \\pmod 2 = {sum(orig_e) % 2}$."
        )