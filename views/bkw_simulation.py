import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


def compute_bkw_theory(k: int, a: int, b: int, tau: float, theta: float = 0.05):
    delta_init = 1.0 - 2.0 * tau
    w_accumulated = 2 ** (a - 1)
    delta_final = delta_init ** w_accumulated
    p_success_single = 0.5 + 0.5 * delta_final

    if delta_final > 2e-12:
        n_min_votes = int(
            np.ceil((2.0 * np.log(1.0 / theta)) / (delta_final ** 2))
        )
    else:
        n_min_votes = float("inf")

    # Teoretický počet vzoriek z orákula na 1 beh (BKW Lemma 4):
    m_single_run = a * (2 ** b)
    # Pravdepodobnosť nájdenia e_1 v jednom behu s veľkosťou M = a * 2^b:
    p_find_e1 = 1.0 - 1.0 / np.e  # cca 63.2 %

    # Celkový teoretický počet vzoriek z orákula na zistenie 1 bitu:
    if n_min_votes != float("inf"):
        oracle_samples_1bit = int(
            np.ceil(n_min_votes * (m_single_run / p_find_e1))
        )
        oracle_samples_full_k = int(oracle_samples_1bit * k)
    else:
        oracle_samples_1bit = float("inf")
        oracle_samples_full_k = float("inf")

    return {
        "delta_init": delta_init,
        "w_accumulated": w_accumulated,
        "delta_final": delta_final,
        "p_success_single": p_success_single,
        "n_min_votes": n_min_votes,
        "m_single_run": m_single_run,
        "p_find_e1": p_find_e1,
        "oracle_samples_1bit": oracle_samples_1bit,
        "oracle_samples_full_k": oracle_samples_full_k,
    }


def execute_fast_single_run(s_true, a, b, m_samples, tau, run_seed):
    np.random.seed(run_seed)
    k = a * b

    A = np.random.randint(0, 2, size=(m_samples, k))
    d = (np.random.rand(m_samples) < tau).astype(int)
    c = (A @ s_true + d) % 2

    for step in range(1, a):
        target_block = a - step
        start_col = target_block * b
        end_col = (target_block + 1) * b

        partitions = {}
        for row_idx in range(len(A)):
            pattern = tuple(A[row_idx, start_col:end_col].tolist())
            partitions.setdefault(pattern, []).append(row_idx)

        A_next, c_next, d_next = [], [], []
        for _, indices in partitions.items():
            if len(indices) >= 2:
                pivot_idx = indices[0]
                for other_idx in indices[1:]:
                    A_next.append(A[other_idx] ^ A[pivot_idx])
                    c_next.append(c[other_idx] ^ c[pivot_idx])
                    d_next.append(d[other_idx] ^ d[pivot_idx])

        if not A_next:
            return False, None, None, 0

        A = np.array(A_next, dtype=int)
        c = np.array(c_next, dtype=int)
        d = np.array(d_next, dtype=int)

    e_1 = np.zeros(k, dtype=int)
    e_1[0] = 1

    matches = np.where((A == e_1).all(axis=1))[0]
    if len(matches) > 0:
        match_idx = matches[0]
        return True, int(c[match_idx]), int(d[match_idx]), len(matches)
    return False, None, None, 0


def generate_detailed_run_data(s_true, a, b, m_samples, tau, run_seed):
    np.random.seed(run_seed)
    k = a * b

    A_0 = np.random.randint(0, 2, size=(m_samples, k))
    d_0 = (np.random.rand(m_samples) < tau).astype(int)
    c_0 = (A_0 @ s_true + d_0) % 2
    lineage_0 = [[i] for i in range(m_samples)]

    history = [
        {
            "step": 0,
            "sample_type": "0-vzorka (východiskový stav)",
            "A": A_0.copy(),
            "c": c_0.copy(),
            "d": d_0.copy(),
            "lineage": lineage_0,
            "elim_block_idx": None,
            "elim_bit_range": None,
            "classes_info": [],
            "operations_log": [],
            "pivots_discarded": 0,
        }
    ]

    A_curr, c_curr, d_curr, lineage_curr = (
        A_0.copy(),
        c_0.copy(),
        d_0.copy(),
        lineage_0,
    )

    for step in range(1, a):
        target_block = a - step
        start_col = target_block * b
        end_col = (target_block + 1) * b

        partitions = {}
        for row_idx in range(len(A_curr)):
            pattern = tuple(A_curr[row_idx, start_col:end_col].tolist())
            partitions.setdefault(pattern, []).append(row_idx)

        A_next, c_next, d_next, lineage_next = [], [], [], []
        operations_log = []
        classes_info = []
        pivots_discarded = 0

        for pattern, indices in partitions.items():
            classes_info.append(
                {
                    "Vzor v bloku": "".join(map(str, pattern)),
                    "Počet riadkov": len(indices),
                    "Vzniknuté nové riadky": max(0, len(indices) - 1),
                }
            )

            if len(indices) >= 2:
                pivot_idx = indices[0]
                pivots_discarded += 1

                for other_idx in indices[1:]:
                    res_vec = (A_curr[other_idx] ^ A_curr[pivot_idx]).tolist()
                    res_c = int(c_curr[other_idx] ^ c_curr[pivot_idx])
                    res_d = int(d_curr[other_idx] ^ d_curr[pivot_idx])
                    res_lin = lineage_curr[other_idx] + lineage_curr[pivot_idx]

                    A_next.append(res_vec)
                    c_next.append(res_c)
                    d_next.append(res_d)
                    lineage_next.append(res_lin)

                    if len(operations_log) < 50:
                        operations_log.append(
                            {
                                "Vzor v bloku": "".join(map(str, pattern)),
                                "Pivot (#)": pivot_idx,
                                "Cieľ (#)": other_idx,
                                "Chyba pivota (d)": int(d_curr[pivot_idx]),
                                "Chyba cieľa (d)": int(d_curr[other_idx]),
                                "Výsledná chyba (d')": res_d,
                                "Výsledný vektor (v')": "".join(
                                    map(str, res_vec)
                                ),
                            }
                        )
            elif len(indices) == 1:
                pivots_discarded += 1

        if not A_next:
            break

        A_curr = np.array(A_next, dtype=int)
        c_curr = np.array(c_next, dtype=int)
        d_curr = np.array(d_next, dtype=int)
        lineage_curr = lineage_next

        history.append(
            {
                "step": step,
                "sample_type": f"{step}-vzorka",
                "A": A_curr.copy(),
                "c": c_curr.copy(),
                "d": d_curr.copy(),
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

    # Kontrola dokončenia všetkých (a - 1) krokov eliminácie
    completed = len(history) == a
    matches = (
        np.where((history[-1]["A"] == e_1).all(axis=1))[0]
        if completed and len(history[-1]["A"]) > 0
        else np.array([], dtype=int)
    )

    found = len(matches) > 0
    vote = int(history[-1]["c"][matches[0]]) if found else None
    err = int(history[-1]["d"][matches[0]]) if found else None

    return {
        "history": history,
        "completed": completed,
        "found": found,
        "vote": vote,
        "err": err,
        "matches_count": len(matches),
        "matches_indices": matches.tolist() if found else [],
    }


@st.cache_data
def run_large_scale_simulation(a, b, m_samples, tau, n_runs, base_seed):
    k = a * b
    np.random.seed(base_seed)
    s_true = np.random.randint(0, 2, size=k)

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
            s_true, a, b, m_samples, tau, run_seed
        )

        if found:
            successful_runs += 1
            if vote == 0:
                votes_0 += 1
            else:
                votes_1 += 1

        total_valid = votes_0 + votes_1
        ratio_true = (
            (votes_0 if s_true[0] == 0 else votes_1) / total_valid
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
                "is_correct": (found and vote == s_true[0]),
                "cum_0": votes_0,
                "cum_1": votes_1,
            }
        )

    return s_true, runs_summary, cum_votes_0, cum_votes_1, cum_ratio_true


def render():
    st.header("2. Pôvodný BKW algoritmus")

    # CSS štýl pre čistý biely border okolo rámčeka
    st.markdown(
        """
        <style>
        div[data-testid="stVerticalBlockBorderWrapper"] {
            border: 1.5px solid #ffffff !important;
            border-radius: 10px !important;
            background-color: transparent !important;
            padding: 10px !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    col_c1, col_c2, col_c3, col_c4 = st.columns(4)
    with col_c1:
        a = st.number_input(
            "Počet blokov (a)",
            min_value=2,
            max_value=5,
            value=3,
            help=(
                "Počet blokov, na ktoré rozdelíme tajný vektor s. Algoritmus eliminuje bloky "
                "v (a - 1) krokoch. Vyššie 'a' umožňuje menšiu šírku 'b', ale exponenciálne "
                "akumuluje šum: konečná rovnica je súčtom w = 2^(a-1) pôvodných rovníc."
            ),
        )
        b = st.number_input(
            "Šírka bloku (b)",
            min_value=1,
            max_value=3,
            value=2,
            help=(
                "Šírka jedného bloku v bitoch. Určuje počet tried ekvivalencie (2^b), "
                "do ktorých triedime riadky matice A pri XOR eliminácii. "
                "Celková dĺžka tajného kľúča s je k = a · b bitov."
            ),
        )
        k = int(a * b)
        st.caption(f"Dimenzia kľúča: **k = {k} bitov**")

    rec_m = int(a * (2 ** b))
    with col_c2:
        m_samples = st.number_input(
            "Vzorky na 1 beh (M)",
            min_value=rec_m,
            max_value=2000,
            value=max(rec_m * 4, 60),
            help=(
                f"Počet počiatočných LPN vzoriek (v_i, c_i) v jednom behu (veľkosť 0-vzorky). "
                f"Podľa Lemmy 4 je teoretické minimum na prežitie eliminácie M = a · 2^b = {rec_m}. "
                "Vyššia hodnota zabezpečí, že v každom kroku nájdeme dostatok dvojíc na elimináciu."
            ),
        )
        st.caption(f"Odporúčané minimum na beh: **$M = {rec_m}$**")

        m_detail_box = (
            st.popover("Ako sa to vyrátalo?", use_container_width=True)
            if hasattr(st, "popover")
            else st.expander("Ako sa to vyrátalo?", expanded=False)
        )

        with m_detail_box:
            st.markdown(
                f"""
                **Odvodenie minima $M = a \\cdot 2^b$ (BKW Lemma 4):**

                1. **Triedy ekvivalencie ($2^b$):**
                   Každý blok má šírku $b = {b}$ bitov, čo predstavuje najviac $2^{{{b}}} = {2**b}$ možných vzorov (tried).

                2. **Strata reprezentantov v každom kroku:**
                   Z každej neprázdnej triedy sa vyberie 1 reprezentant na odčítanie od ostatných a zahodí sa, čím v každom kroku eliminácie stratíme najviac $2^b = {2**b}$ vzoriek.

                3. **Eliminácia $a - 1$ blokov:**
                   Eliminujeme postupne $a - 1 = {a - 1}$ blokov. Celková strata vzoriek počas eliminácie je najviac:
                   $$(a - 1) \\cdot 2^b = {a - 1} \\cdot {2**b} = {(a - 1) * (2**b)}\\text{{ vzoriek}}$$

                4. **Zostávajúce vzorky pre cieľ $e_1$:**
                   Aby v podpriestore $V_{{a-1}}$ zostalo aspoň $2^b = {2**b}$ vektorov na úspešné nájdenie bázového vektora $e_1$, počiatočná veľkosť $M$ musí spĺňať:
                   $$M - (a - 1) \\cdot 2^b \\ge 2^b \\implies M \\ge a \\cdot 2^b = {a} \\cdot {2**b} = \\mathbf{{{rec_m}\\text{{ vzoriek}}}}$$
                """
            )

    with col_c3:
        tau = st.slider(
            "Chybovosť šumu (τ)",
            min_value=0.01,
            max_value=0.45,
            value=0.10,
            step=0.01,
            help=(
                "Pravdepodobnosť výskytu Bernoulliho chyby d = 1 vo výstupe (c = ⟨v, s⟩ ⊕ d). "
                "Pri τ = 0 ide o bezšumovú sústavu lineárnych rovníc. "
                "Pri τ = 0,5 je výstup čisto náhodný a neobsahuje žiadnu informáciu o kľúči s."
            ),
        )
        desired_confidence = st.selectbox(
            "Požadovaná istota (1 - θ)",
            [0.95, 0.99, 0.999],
            index=0,
            help=(
                "Pravdepodobnosť, že celkové väčšinové hlasovanie naprieč N_min platnými hlasmi správne určí bit s₁. "
                "Parameter θ predstavuje maximálnu tolerovanú pravdepodobnosť chyby väčšiny (pre 1 bit)."
            ),
        )
        theta = 1.0 - desired_confidence

    theory = compute_bkw_theory(k, a, b, tau, theta)
    n_min_votes = theory["n_min_votes"]

    with col_c4:
        seed = st.number_input(
            "Základný seed generátora",
            min_value=1,
            max_value=9999,
            value=42,
            help="Počiatočná hodnota generátora pseudonáhodných čísel.",
        )

        if n_min_votes != float("inf"):
            with st.container(border=True):
                st.metric(
                    label="Min. počet hlasov ($N_{\\min}$)",
                    value=f"{n_min_votes} hlasov",
                    delta=f"pre istotu {desired_confidence * 100:.1f} %",
                    delta_color="normal",
                    help=(
                        "Minimálny počet úspešných behov (ktoré nájdu e₁) potrebný pre väčšinové hlasovanie "
                        "s pravdepodobnosťou chyby najviac θ."
                    ),
                )

                st.metric(
                    label="Potrebné vzorky z orákula",
                    value=f"{theory['oracle_samples_1bit']:,}",
                    delta=f"pre 1 bit ({theory['oracle_samples_full_k']:,} pre celé k)",
                    delta_color="off",
                    help=(
                        "Očakávaný počet dopytov z orákula pre pôvodný BKW na nájdenie 1 bitu s₁: "
                        "n_oracle = N_min · (a · 2^b / (1 - 1/e)). Orientačný odhad pre k bitov je k · n_oracle. "
                        "Na rozdiel od BKW* sa vzorky medzi behmi nesmú recyklovať."
                    ),
                )

                detail_box = (
                    st.popover(
                        "Odvodenie počtu vzoriek", use_container_width=True
                    )
                    if hasattr(st, "popover")
                    else st.expander(
                        "Odvodenie počtu vzoriek", expanded=False
                    )
                )

                with detail_box:
                    st.markdown(
                        f"""
                        **Prečo pôvodný BKW spotrebuje toľko vzoriek z orákula?**

                        1. **Požiadavka na nezávislosť:**
                           Pôvodný BKW (2000) vyžaduje, aby každý beh bežal na **úplne nových, čerstvých vzorkách** z orákula. Žiadne rovnice sa medzi hlasmi nerecyklujú.

                        2. **Vzorky na 1 platný hlas ($e_1$):**
                           Jeden minimálny beh spotrebuje $M = a \\cdot 2^b = {theory['m_single_run']}$ vzoriek.
                           Pravdepodobnosť, že v podpriestore $V_{{a-1}}$ vznikne hľadaný vektor $e_1$, je $1 - 1/e \\approx 63{{,}}2\\,\\%$.
                           Na získanie jedného hlasu teda v priemere potrebujeme:
                           $$\\frac{{a \\cdot 2^b}}{{1 - 1/e}} = \\frac{{{theory['m_single_run']}}}{{0{{,}}632}} \\approx \\mathbf{{{int(np.ceil(theory['m_single_run'] / theory['p_find_e1']))}\\text{{ vzoriek/hlas}}}}$$

                        3. **Celkový počet vzoriek na 1 bit:**
                           $$n_{{\\text{{oracle}}}} = N_{{\\min}} \\cdot \\frac{{a \\cdot 2^b}}{{1 - 1/e}} = {n_min_votes} \\cdot {int(np.ceil(theory['m_single_run'] / theory['p_find_e1']))} = \\mathbf{{{theory['oracle_samples_1bit']:,}\\text{{ vzoriek}}}}$$

                        4. **Orientačne pre celý kľúč ($k = {k}$ bitov):**
                           $$k \\cdot n_{{\\text{{oracle}}}} = {k} \\cdot {theory['oracle_samples_1bit']:,} = \\mathbf{{{theory['oracle_samples_full_k']:,}\\text{{ vzoriek}}}}$$
                        """
                    )
        else:
            st.warning("⚠️ Príliš nízky bias (δ' ≈ 0).")

    st.markdown("#### Rozsah simulácie väčšinového hlasovania")
    default_runs = (
        min(n_min_votes, 300)
        if (n_min_votes != float("inf") and n_min_votes > 0)
        else 100
    )
    n_runs = st.number_input(
        "Celkový počet simulovaných behov (N):",
        min_value=5,
        max_value=5000,
        value=int(default_runs),
        step=10,
        help="Celkový počet nezávislých behov eliminácie. Každý beh generuje novú sadu M vzoriek.",
    )

    s_true, runs_summary, cum_0, cum_1, cum_ratio = run_large_scale_simulation(
        a, b, m_samples, tau, n_runs, seed
    )
    true_s1 = int(s_true[0])

    votes_0_total = cum_0[-1]
    votes_1_total = cum_1[-1]
    total_valid = votes_0_total + votes_1_total
    winner = (
        0
        if votes_0_total > votes_1_total
        else (1 if votes_1_total > votes_0_total else -1)
    )
    is_winner_correct = winner == true_s1
    hoeffding_error_bound = (
        np.exp(-0.5 * total_valid * (theory["delta_final"] ** 2))
        if total_valid > 0
        else 1.0
    )

    # Reálne vygenerované vzorky v tejto simulácii:
    actual_samples_generated = n_runs * m_samples

    st.divider()
    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.metric(
            "Skutočný kľúč s",
            f"{s_true.tolist()}",
            f"1. bit s₁ = {true_s1}",
            help="Náhodný tajný vektor s vygenerovaný orákulom.",
        )
    with m2:
        st.metric(
            "Výsledok väčšiny (ŝ₁)",
            f"{winner}" if winner != -1 else "Remíza",
            "Správne odhalený" if is_winner_correct else "Chyba",
            help="Hodnota prvého bitu s₁, ktorá získala väčšinu hlasov z úspešných behov.",
        )
    with m3:
        st.metric(
            "Pomer hlasov [0 : 1]",
            f"{votes_0_total} : {votes_1_total}",
            f"Z {total_valid} platných hlasov",
            help="Rozdelenie hlasov medzi hodnoty 0 a 1 v úspešných behoch.",
        )
    with m4:
        st.metric(
            "Vzorky v simulácii",
            f"{actual_samples_generated:,}",
            f"{n_runs} behov × {m_samples}",
            help="Skutočný celkový počet rovníc vygenerovaných v tejto interaktívnej simulácii.",
        )
    with m5:
        st.metric(
            "Horná hranica chyby",
            f"{hoeffding_error_bound:.4e}",
            f"Cieľová θ = {theta:.3f}",
            help="Hoeffdingov horný odhad pravdepodobnosti zlyhania väčšinového hlasovania po získaní platných hlasov.",
        )

    with st.expander(
        "Teoretický rozbor a vzorce (kliknite pre zobrazenie)",
        expanded=False,
    ):
        st.markdown(
            f"""
        1. **Piling-up Lemma (Lemma 3):**
           Každý beh vyjadrí $e_1$ ako súčet $w = 2^{{a-1}} = 2^{{{a - 1}}} = {theory['w_accumulated']}$ pôvodných rovníc.
           Akumulovaný bias je:
           $$\\delta' = \\delta^{{2^{{a-1}}}} = (1 - 2\\tau)^{{2^{{a-1}}}} = ({theory['delta_init']:.2f})^{{{theory['w_accumulated']}}} = {theory['delta_final']:.4e}$$

        2. **Pravdepodobnosť úspechu v jednom platnom behu:**
           $$\\Pr[c' = s_1] = \\frac{{1 + \\delta'}}{{2}} = \\frac{{1}}{{2}} + \\frac{{\\delta'}}{{2}} = {theory['p_success_single'] * 100:.2f}\\,\\%$$

        3. **Hoeffdingova nerovnosť a minimálny počet hlasov ($N_{{\\min}}$):**
           $$\\Pr[\\text{{chyba väčšiny}}] \\le e^{{-\\frac{{N(\\delta')^2}}{{2}}}} \\le \\theta \\implies N_{{\\min}} \\ge \\frac{{2\\ln(1/\\theta)}}{{(\\delta')^2}} = \\mathbf{{{n_min_votes}\\text{{ hlasov}}}}$$

        4. **Celkový počet vzoriek z orákula na 1 bit:**
           $$n_{{\\text{{oracle}}}} = N_{{\\min}} \\cdot \\frac{{a \\cdot 2^b}}{{1 - 1/e}} = \\mathbf{{{theory['oracle_samples_1bit']:,}\\text{{ vzoriek}}}}$$
        """
        )

    st.divider()

    st.subheader(f"1. Priebeh väčšinového hlasovania (N = {n_runs} behov)")

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
            title="Vývoj počtu hlasov v čase",
            xaxis_title="Číslo behu",
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
                name="Úspešnosť správneho bitu",
                line=dict(color="#2ca02c", width=2),
            )
        )
        fig_ratio.add_hline(
            y=0.5,
            line_dash="dash",
            line_color="red",
            annotation_text="50 % (hranica náhody)",
        )
        fig_ratio.add_hline(
            y=theory["p_success_single"],
            line_dash="dot",
            line_color="blue",
            annotation_text=f"Teoretická hodnota ({theory['p_success_single'] * 100:.1f} %)",
        )
        fig_ratio.update_layout(
            title=f"Podiel hlasov pre správny bit (s₁ = {true_s1})",
            xaxis_title="Číslo behu",
            yaxis_title="Podiel hlasov",
            yaxis_range=[0, 1],
            height=320,
            margin=dict(l=20, r=20, t=35, b=20),
        )
        st.plotly_chart(fig_ratio, use_container_width=True)

    df_summary = pd.DataFrame(runs_summary)
    with st.expander(
        f"Tabuľka výsledkov všetkých {n_runs} behov (kliknite pre zobrazenie)"
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

    st.subheader("2. Detail vybraného behu")

    selected_run_idx = st.number_input(
        f"Zadajte číslo behu na zobrazenie (1 až {n_runs}):",
        min_value=1,
        max_value=n_runs,
        value=1,
        step=1,
        help="Vyberte konkrétny simulačný beh na podrobnú analýzu eliminácie krok po kroku.",
    )

    run_seed = seed * 10000 + selected_run_idx
    chosen_run_data = generate_detailed_run_data(
        s_true, a, b, m_samples, tau, run_seed
    )
    run_history = chosen_run_data["history"]

    c_run1, c_run2, c_run3, c_run4 = st.columns(4)
    with c_run1:
        st.info(f"**Zobrazený beh #{selected_run_idx}**")
    with c_run2:
        st.metric(
            "Nájdený cieľový vektor e₁",
            "Áno" if chosen_run_data["found"] else "Nie",
            f"Vyskytol sa {chosen_run_data['matches_count']}×",
            help="Indikuje, či po všetkých (a - 1) krokoch eliminácie vznikol vektor e₁ = (1, 0, ..., 0).",
        )
    with c_run3:
        st.metric(
            "Odovzdaný hlas (c')",
            (
                f"{chosen_run_data['vote']}"
                if chosen_run_data["found"]
                else "Žiadny"
            ),
            f"Skutočný bit s₁ = {true_s1}",
            help="Výsledný zašumený výstup c' prislúchajúci vektoru e₁.",
        )
    with c_run4:
        st.metric(
            "Výsledná chyba (d')",
            f"{chosen_run_data['err']}" if chosen_run_data["found"] else "—",
            "0 = bez chyby | 1 = chyba",
            help="Akumulovaná chyba d' pre vektor e₁.",
        )

    if not chosen_run_data["completed"]:
        st.warning(
            f"Redukcia v tomto behu skončila predčasne v kroku {len(run_history) - 1} z {a - 1}, "
            "pretože nezostali žiadne dvojice s rovnakým vzorom v bloku."
        )

    st.divider()
    st.markdown(f"#### Kroky eliminácie v behu #{selected_run_idx}")

    max_steps_in_run = len(run_history) - 1
    selected_step = st.slider(
        f"Vyberte krok eliminácie v behu #{selected_run_idx}:",
        min_value=0,
        max_value=max_steps_in_run,
        value=max_steps_in_run,
        format="Krok %d",
        help="Krok 0 sú východiskové vzorky. Každý ďalší krok eliminuje stĺpce jedného bloku.",
    )

    step_data = run_history[selected_step]
    A_step = step_data["A"]

    st.markdown(
        f"**Aktuálny stav:** `{step_data['sample_type']}` | Počet zostávajúcich riadkov: **{len(A_step)}**"
    )

    col_mat, col_cls = st.columns([3, 2])
    with col_mat:
        total_rows = len(A_step)

        if total_rows == 0:
            st.warning("V tomto kroku nezostali žiadne vektory.")
        else:
            if total_rows > 1:
                slider_key = f"disp_rows_s{selected_step}_r{selected_run_idx}"
                if slider_key in st.session_state:
                    if st.session_state[slider_key] > total_rows:
                        st.session_state[slider_key] = total_rows
                    elif st.session_state[slider_key] < 1:
                        st.session_state[slider_key] = 1

                disp_rows = st.slider(
                    "Počet zobrazených riadkov matice A:",
                    min_value=1,
                    max_value=total_rows,
                    value=min(total_rows, 35),
                    step=1,
                    key=slider_key,
                )
            else:
                disp_rows = 1

            st.markdown(
                f"##### Matica vzoriek A v kroku {selected_step} (zobrazených {disp_rows} z {total_rows} riadkov)"
            )

            plot_height = max(320, min(800, 50 + disp_rows * 14))

            fig_mat = px.imshow(
                A_step[:disp_rows, :],
                color_continuous_scale=[[0, "#ffffff"], [1, "#1f77b4"]],
                labels=dict(x="Stĺpec (bit)", y="Riadok (vzorka)", color="Bit"),
            )
            fig_mat.update_layout(
                coloraxis_showscale=False,
                height=plot_height,
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
        st.markdown(f"##### Triedy ekvivalencie v kroku {selected_step}")
        if step_data["classes_info"]:
            st.dataframe(
                pd.DataFrame(step_data["classes_info"]),
                height=250,
                use_container_width=True,
            )
            st.caption(
                f"Zahodených reprezentantov (vrátane osamelých riadkov) v tomto kroku: **{step_data['pivots_discarded']}**"
            )
        else:
            st.info("Krok 0: východiskový stav pred elimináciou.")

    if step_data["operations_log"]:
        st.markdown(f"##### Záznam operácií XOR v kroku {selected_step}")
        st.caption("Ukážka sčítania dvoch riadkov a prenosu chýb:")
        st.dataframe(
            pd.DataFrame(step_data["operations_log"]),
            height=200,
            use_container_width=True,
        )

    if selected_step == max_steps_in_run and chosen_run_data["found"]:
        st.markdown(f"##### Pôvod vektora e₁ v behu #{selected_run_idx}")
        first_match_idx = chosen_run_data["matches_indices"][0]
        src_row_ids = step_data["lineage"][first_match_idx]

        orig_A = run_history[0]["A"][src_row_ids]
        orig_c = run_history[0]["c"][src_row_ids]
        orig_d = run_history[0]["d"][src_row_ids]

        df_lineage = pd.DataFrame(
            orig_A, columns=[f"v_{j + 1}" for j in range(k)]
        )
        df_lineage.insert(0, "Pôvodná vzorka", [f"#{idx}" for idx in src_row_ids])
        df_lineage["Výstup (c)"] = orig_c
        df_lineage["Pôvodná chyba (d)"] = orig_d

        st.dataframe(df_lineage, use_container_width=True, hide_index=True)

        unique_count = len(set(src_row_ids))
        cancel_note = (
            f" (z toho **{unique_count} unikátnych** – zdieľané pivoty sa pri párnom počte výskytov v XOR-e vyrušia)"
            if unique_count < len(src_row_ids)
            else ""
        )
        st.success(
            f"Tento vektor $e_1$ vznikol stromom sčítania **{len(src_row_ids)} pôvodných rovníc** "
            f"($w = 2^{{a-1}} = {2 ** (a - 1)}$){cancel_note}. "
            f"Výsledná chyba po sčítaní: $\\bigoplus d_i = {sum(orig_d) % 2}$."
        )


if __name__ == "__main__":
    render()