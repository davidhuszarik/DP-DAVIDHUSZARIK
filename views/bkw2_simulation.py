import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


def compute_bkw_star_theory(
    k: int, a: int, b: int, eta: float, theta: float = 0.333
):
    delta = 1.0 - 2.0 * eta
    accumulated_eqs = 2 ** (a - 1)
    reduced_bias = delta**accumulated_eqs
    gamma = 0.5 * reduced_bias
    p_success_single = 0.5 + gamma

    # Podľa Theorem 4 (Bogos, Tramer, Vaudenay 2016):
    # n = 2^(b+1) * delta^(-2^a) * ln(b / theta) + (a - 1) * 2^b
    term_solve = (2 ** (b + 1)) * (delta ** (-(2**a))) * np.log(b / theta)
    term_discard = (a - 1) * (2**b)
    n_theor_bogos = int(np.ceil(term_solve + term_discard))

    # Starší odhad podľa Levieil & Fouque 2006 (Theorem 1):
    # n = 20 * ln(4k) * 2^b * delta^(-2^a) + (a - 1) * 2^b
    term_lf = 20.0 * np.log(4.0 * k) * (2**b) * (delta ** (-(2**a)))
    n_theor_lf = int(np.ceil(term_lf + term_discard))

    # Počet vzoriek pre celý k-bitový kľúč (rekonštrukcia a = k/b blokov):
    n_theor_full_k = int(np.ceil(n_theor_bogos * (k / b)))

    # Očakávaný počet vzoriek s váhou 1 na jeden bit:
    n_prime_expected = (
        max(0, n_theor_bogos - term_discard) / (2**b)
        if (n_theor_bogos > term_discard)
        else 0
    )

    return {
        "delta": delta,
        "accumulated_eqs": accumulated_eqs,
        "reduced_bias": reduced_bias,
        "gamma": gamma,
        "p_success_single": p_success_single,
        "term_solve": term_solve,
        "term_discard": term_discard,
        "n_theor_bogos": n_theor_bogos,
        "n_theor_lf": n_theor_lf,
        "n_theor_full_k": n_theor_full_k,
        "n_prime_expected": n_prime_expected,
    }


def execute_bkw_star_with_lineage(c_true, a, b, n_samples, eta, seed):
    np.random.seed(seed)
    k = a * b

    # 1. Odber počiatočných vzoriek z orákula
    X_0 = np.random.randint(0, 2, size=(n_samples, k))
    e_0 = (np.random.rand(n_samples) < eta).astype(int)
    c_noisy_0 = (X_0 @ c_true + e_0) % 2
    lineage_0 = [[i] for i in range(n_samples)]

    history = [
        {
            "step": 0,
            "sample_type": "0-vzorka (východiskový stav)",
            "X": X_0.copy(),
            "c_noisy": c_noisy_0.copy(),
            "e": e_0.copy(),
            "lineage": lineage_0,
            "classes_info": [],
            "operations_log": [],
            "pivots_discarded": 0,
        }
    ]

    X_curr = X_0.copy()
    c_curr = c_noisy_0.copy()
    e_curr = e_0.copy()
    lineage_curr = lineage_0

    # 2. Redukčná fáza (eliminácia a - 1 blokov pomocou pivotov)
    for step in range(1, a):
        target_block = a - step
        start_col = target_block * b
        end_col = (target_block + 1) * b

        partitions = {}
        for row_idx in range(len(X_curr)):
            pattern = tuple(X_curr[row_idx, start_col:end_col].tolist())
            partitions.setdefault(pattern, []).append(row_idx)

        X_next, c_next, e_next, lineage_next = [], [], [], []
        operations_log = []
        classes_info = []
        pivots_discarded = 0

        for pattern, indices in partitions.items():
            classes_info.append(
                {
                    "Vzor v bloku": "".join(map(str, pattern)),
                    "Počet riadkov": len(indices),
                    "Vzniknuté riadky": max(0, len(indices) - 1),
                }
            )

            # Prvý vektor je reprezentant (pivot), sčíta sa s ostatnými a zahodí sa
            if len(indices) >= 2:
                pivot_idx = indices[0]
                pivots_discarded += 1

                for other_idx in indices[1:]:
                    res_vec = (X_curr[other_idx] ^ X_curr[pivot_idx]).tolist()
                    res_c = int(c_curr[other_idx] ^ c_curr[pivot_idx])
                    res_e = int(e_curr[other_idx] ^ e_curr[pivot_idx])
                    res_lin = lineage_curr[other_idx] + lineage_curr[pivot_idx]

                    X_next.append(res_vec)
                    c_next.append(res_c)
                    e_next.append(res_e)
                    lineage_next.append(res_lin)

                    if len(operations_log) < 50:
                        operations_log.append(
                            {
                                "Vzor v bloku": "".join(map(str, pattern)),
                                "Pivot (#)": pivot_idx,
                                "Cieľ (#)": other_idx,
                                "Chyba pivota": int(e_curr[pivot_idx]),
                                "Chyba cieľa": int(e_curr[other_idx]),
                                "Výsledná chyba": res_e,
                                "Výsledný vektor": "".join(map(str, res_vec)),
                            }
                        )
            elif len(indices) == 1:
                pivots_discarded += 1

        if not X_next:
            break

        X_curr = np.array(X_next, dtype=int)
        c_curr = np.array(c_next, dtype=int)
        e_curr = np.array(e_next, dtype=int)
        lineage_curr = lineage_next

        history.append(
            {
                "step": step,
                "sample_type": f"{step}-vzorka (vynulovaný blok {target_block + 1})",
                "X": X_curr.copy(),
                "c_noisy": c_curr.copy(),
                "e": e_curr.copy(),
                "lineage": lineage_curr,
                "classes_info": classes_info,
                "operations_log": operations_log,
                "pivots_discarded": pivots_discarded,
            }
        )

    # 3. Riešiaca fáza (Väčšinové hlasovanie pre celý prvý blok s dĺžkou b)
    bit_results = []
    recovered_block = []
    target_matches_map = {}

    for bit_idx in range(b):
        target_vec = np.zeros(k, dtype=int)
        target_vec[bit_idx] = 1

        matches = (
            np.where((X_curr == target_vec).all(axis=1))[0]
            if len(X_curr) > 0
            else np.array([])
        )
        count_matches = len(matches)
        target_matches_map[bit_idx] = matches.tolist()

        if count_matches > 0:
            votes = c_curr[matches]
            votes_0 = int(np.sum(votes == 0))
            votes_1 = int(np.sum(votes == 1))
            errors_in_sample = int(np.sum(e_curr[matches] == 1))

            guessed_bit = 0 if votes_0 >= votes_1 else 1
            is_correct = guessed_bit == c_true[bit_idx]
        else:
            votes_0 = 0
            votes_1 = 0
            errors_in_sample = 0
            guessed_bit = -1  # Nedostatok vzoriek
            is_correct = False

        recovered_block.append(guessed_bit)
        bit_results.append(
            {
                "Bit pozícia": f"s_{bit_idx + 1}",
                "Skutočná hodnota": int(c_true[bit_idx]),
                "Odhad väčšiny": guessed_bit if guessed_bit != -1 else "—",
                "Správne": "Áno" if is_correct else "Nie",
                "Počet vzoriek (váha 1)": count_matches,
                "Hlasy [0 : 1]": f"{votes_0} : {votes_1}",
                "Z toho chýb (e=1)": errors_in_sample,
            }
        )

    block_correct = recovered_block == c_true[:b].tolist()

    return {
        "history": history,
        "bit_results": bit_results,
        "recovered_block": recovered_block,
        "block_correct": block_correct,
        "remaining_samples": len(X_curr),
        "target_matches_map": target_matches_map,
    }


def render():
    st.header("Algoritmus BKW* (Praktický BKW)")
    st.caption(
        "Interaktívna simulácia vylepšeného algoritmu BKW* podľa Levieila, Fouqueho a Bogos et al. (2016). "
        "Na rozdiel od pôvodného BKW sa vzorky od orákula zbierajú iba raz na začiatku a na konci "
        "sa paralelne vyhodnotí celý b-bitový blok pomocou väčšinového hlasovania."
    )

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

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        a = st.number_input(
            "Počet blokov (a)",
            min_value=2,
            max_value=5,
            value=3,
            help="Počet blokov kľúča. Redukcia eliminuje stĺpce v a - 1 krokoch.",
        )
        b = st.number_input(
            "Šírka bloku (b)",
            min_value=1,
            max_value=4,
            value=2,
            help="Veľkosť jedného bloku v bitoch. Po redukcii zostane neznámych práve b bitov.",
        )
        k = int(a * b)
        st.caption(f"Celková dĺžka kľúča: **k = {k} bitov**")

    rec_loss = int((a - 1) * (2**b))

    with col2:
        eta = st.slider(
            "Chybovosť šumu (η)",
            min_value=0.01,
            max_value=0.45,
            value=0.10,
            step=0.01,
            help=(
                "Pravdepodobnosť výskytu Bernoulliho chyby e = 1 v označení vzorky (c = ⟨v, s⟩ ⊕ e). "
                "Pri η = 0 ide o bežnú lineárnu algebru bez šumu. "
                "Pri η = 0.5 je šum úplne náhodný a kľúč sa nedá zistiť žiadnym algoritmom."
            ),
        )
        theta = st.selectbox(
            "Tolerancia zlyhania bloku (θ)",
            [0.10, 0.20, 0.333, 0.50],
            index=2,
            help=(
                "Maximálna akceptovateľná pravdepodobnosť, že aspoň jeden bit z b-bitového bloku bude uhádnutý nesprávne. "
                "Istota správnosti bloku je 1 - θ."
            ),
        )

    theory = compute_bkw_star_theory(k, a, b, eta, theta)
    n_bogos = theory["n_theor_bogos"]
    n_lf = theory["n_theor_lf"]

    with col3:
        n_input_default = min(max(n_bogos, 200), 50000)
        n_samples = st.number_input(
            "Počiatočné vzorky z orákula (n)",
            min_value=int(rec_loss + 50),
            max_value=200000,
            value=int(n_input_default),
            step=100,
            help=(
                "Počet počiatočných LPN rovníc zozbieraných od orákula na začiatku. "
                "BKW* ich spracuje naraz v jedinom behu a recykluje ich."
            ),
        )
        st.caption(f"Minimálna strata pivotov v redukcii: **${rec_loss}$ vzoriek**")

        # Edukatívne vysvetlenie straty pivotov
        loss_box = (
            st.popover("ℹ️ Ako vzniká strata pivotov?", use_container_width=True)
            if hasattr(st, "popover")
            else st.expander("ℹ️ Ako vzniká strata pivotov?", expanded=False)
        )

        with loss_box:
            st.markdown(
                f"""
                **Odvodenie straty pivotov v BKW*:**

                1. **Priehradkový princíp ($2^b$):**
                   Každý blok má šírku $b = {b}$ bitov, čo predstavuje $2^{{{b}}} = {2**b}$ možných kombinácií (tried ekvivalencie).

                2. **Strata 1 pivota na triedu v každom kroku:**
                   V každej neprázdnej triede sa vyberie 1 reprezentatívny vektor (pivot), sčíta sa s ostatnými a zahodí sa. V každom redukčnom kroku tak stratíme najviac $2^b = {2**b}$ vzoriek.

                3. **Celková strata za $a - 1$ krokov:**
                   Eliminujeme postupne $a - 1 = {a - 1}$ blokov. Celkový počet zahodených pivotov je:
                   $$(a - 1) \\cdot 2^b = {a - 1} \\cdot {2**b} = \\mathbf{{{rec_loss}\\text{{ vzoriek}}}}$$

                *Všetky ostatné vzorky redukciou prejdú a na konci priamo hlasujú o bitoch tajomstva.*
                """
            )

    with col4:
        seed = st.number_input(
            "Základný seed generátora",
            min_value=1,
            max_value=99999,
            value=42,
            help="Počiatočná hodnota generátora pseudonáhodných čísel.",
        )

        with st.container(border=True):
            st.metric(
                label="Potrebné vzorky z orákula ($n$)",
                value=f"{n_bogos:,}",
                delta=f"pre 1 blok ({theory['n_theor_full_k']:,} pre celé k)",
                delta_color="normal",
                help=(
                    "Minimálny teoretický počet počiatočných dopytov z orákula podľa Theorem 4 (Bogos et al. 2016), "
                    "aby algoritmus BKW* s pravdepodobnosťou aspoň 1 - θ správne odhalil celý b-bitový blok."
                ),
            )

            st.caption(
                f"Odhad Levieil & Fouque (2006): **{n_lf:,}** ({n_lf / n_bogos:.1f}× viac)"
            )

            detail_box = (
                st.popover("ℹ️ Ako sa to vyrátalo?", use_container_width=True)
                if hasattr(st, "popover")
                else st.expander("ℹ️ Ako sa to vyrátalo?", expanded=False)
            )

            with detail_box:
                st.markdown(
                    f"""
                    **Postup výpočtu $n = {n_bogos:,}$ vzoriek (Theorem 4):**

                    1. **Akumulácia šumu v redukcii:**
                       Každá vzorka po $a - 1 = {a - 1}$ krokoch vznikne sčítaním $w = 2^{{a-1}} = 2^{{{a - 1}}} = {theory['accumulated_eqs']}$ pôvodných rovníc.

                    2. **Konečný bias (Piling-up Lemma):**
                       Počiatočný bias $\\delta = 1 - 2\\eta = {theory['delta']:.2f}$ klesne na:
                       $$\\delta' = \\delta^{{2^{{a-1}}}} = ({theory['delta']:.2f})^{{{theory['accumulated_eqs']}}} = \\mathbf{{{theory['reduced_bias']:.4e}}}$$
                       *(Jedna vzorka trafí správny bit s pravdepodobnosťou len **{theory['p_success_single'] * 100:.2f} %**)*.

                    3. **Hoeffdingova nerovnosť a Union Bound:**
                       Pre 1 bit je potrebných $n' = 2(\\delta')^{{-2}} \\ln(b/\\theta)$ vzoriek s váhou 1. Keďže takýto tvar má v priemere len $1 / 2^b$ vzoriek, na riešenie treba:
                       $$n_{{\\text{{solve}}}} = 2^{{b+1}} \\delta^{{-2^a}} \\ln\\left(\\frac{{b}}{{\\theta}}\\right) \\approx \\mathbf{{{int(theory['term_solve']):,}\\text{{ vzoriek}}}}$$

                    4. **Započítanie straty pivotov počas redukcie:**
                       K tomu pripočítame vzorky zahodené počas eliminácie:
                       $$n = n_{{\\text{{solve}}}} + (a - 1)2^b = {int(theory['term_solve']):,} + {int(theory['term_discard'])} = \\mathbf{{{n_bogos:,}\\text{{ vzoriek}}}}$$

                    5. **Celý kľúč ($k = {k}$ bitov):**
                       Pri postupnom riešení všetkých $a = k/b = {int(k/b)}$ blokov je celková teoretická spotreba:
                       $$a \\cdot n \\approx \\mathbf{{{theory['n_theor_full_k']:,}\\text{{ vzoriek}}}}$$
                    """
                )

    # Spustenie simulácie BKW*
    np.random.seed(seed)
    c_true = np.random.randint(0, 2, size=k)
    sim_result = execute_bkw_star_with_lineage(c_true, a, b, n_samples, eta, seed)

    st.divider()

    # Prehľadové metriky
    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.metric(
            "Skutočný kľúč s",
            f"{c_true.tolist()}",
            f"1. blok s₁..s_{b} = {c_true[:b].tolist()}",
            help="Náhodný tajný kľúč s vygenerovaný orákulom.",
        )
    with m2:
        st.metric(
            "Odhadnutý 1. blok",
            f"{sim_result['recovered_block']}",
            (
                "Celý blok správne"
                if sim_result["block_correct"]
                else "Chyba v bloku"
            ),
            help="Výsledok paralelného väčšinového hlasovania pre všetkých b bitov prvého bloku.",
        )
    with m3:
        st.metric(
            "Zostávajúce po redukcii",
            f"{sim_result['remaining_samples']:,}",
            f"Stratených pivotov: {n_samples - sim_result['remaining_samples']:,}",
            help="Počet rovníc, ktoré prežili eliminačné XOR operácie.",
        )
    with m4:
        total_weight1 = sum(
            r["Počet vzoriek (váha 1)"] for r in sim_result["bit_results"]
        )
        st.metric(
            "Vzorky s váhou 1",
            f"{total_weight1:,}",
            f"Priemer: {total_weight1 / b:.1f} na bit",
            help="Celkový počet vzoriek s Hammingovou váhou 1, ktoré reálne hlasovali o bitoch kľúča.",
        )
    with m5:
        # Empirická horná hranica chyby Hoeffdinga pre blok
        min_samples_per_bit = min(
            r["Počet vzoriek (váha 1)"] for r in sim_result["bit_results"]
        )
        empirical_error_bound = (
            b * np.exp(-min_samples_per_bit * (theory["reduced_bias"] ** 2) / 2)
            if min_samples_per_bit > 0
            else 1.0
        )
        st.metric(
            "Horná hranica chyby",
            f"{empirical_error_bound:.4e}",
            f"Cieľová θ = {theta:.3f}",
            help="Pravdepodobnosť zlyhania aspoň jedného bitu v bloku podľa Hoeffdingovej nerovnosti a union bound.",
        )

    with st.expander(
        "Teoretický rozbor a vzorce (kliknite pre zobrazenie)",
        expanded=False,
    ):
        st.markdown(
            f"""
        1. **Piling-up Lemma:**
           Každá vzorka po redukcii vznikne sčítaním $w = 2^{{a-1}} = 2^{{{a - 1}}} = {theory['accumulated_eqs']}$ pôvodných rovníc.
           Akumulovaný bias je:
           $$\\delta' = \\delta^{{2^{{a-1}}}} = ({theory['delta']:.2f})^{{{theory['accumulated_eqs']}}} = {theory['reduced_bias']:.4e}$$

        2. **Pravdepodobnosť správnosti jednej vzorky:**
           $$P(\\text{{úspech}}) = \\frac{{1}}{{2}} + \\frac{{\\delta'}}{{2}} = {theory['p_success_single'] * 100:.2f} \\%$$

        3. **Hoeffdingova nerovnosť na 1 bit:**
           Pravdepodobnosť, že väčšinové hlasovanie pre bit $s_j$ zlyhá pri $n'$ vzorkách s váhou 1:
           $$\\Pr[\\text{{chyba pri }} s_j] \\le e^{{-\\frac{{n'(\\delta')^2}}{{2}}}}$$

        4. **Celkový počet vzoriek z orákula na 1 blok (Theorem 4):**
           $$n = 2^{{b+1}} \\delta^{{-2^a}} \\ln\\left(\\frac{{b}}{{\\theta}}\\right) + (a - 1)2^b = \\mathbf{{{n_bogos:,}\\text{{ vzoriek}}}}$$
        """
        )

    st.divider()

    st.subheader(f"1. Výsledky väčšinového hlasovania pre blok ({b} bitov)")
    df_bits = pd.DataFrame(sim_result["bit_results"])
    st.dataframe(df_bits, use_container_width=True, hide_index=True)

    # Graf hlasov pre jednotlivé bity
    fig_votes = go.Figure()
    bit_names = [r["Bit pozícia"] for r in sim_result["bit_results"]]
    v0_list = [
        int(r["Hlasy [0 : 1]"].split(" : ")[0])
        for r in sim_result["bit_results"]
    ]
    v1_list = [
        int(r["Hlasy [0 : 1]"].split(" : ")[1])
        for r in sim_result["bit_results"]
    ]

    fig_votes.add_trace(
        go.Bar(name="Hlasy pre 0", x=bit_names, y=v0_list, marker_color="#1f77b4")
    )
    fig_votes.add_trace(
        go.Bar(name="Hlasy pre 1", x=bit_names, y=v1_list, marker_color="#ff7f0e")
    )
    fig_votes.update_layout(
        barmode="group",
        title="Rozdelenie hlasov pre jednotlivé bity prvého bloku",
        xaxis_title="Bit kľúča",
        yaxis_title="Počet odovzdaných hlasov",
        height=320,
        margin=dict(l=20, r=20, t=35, b=20),
    )
    st.plotly_chart(fig_votes, use_container_width=True)

    st.divider()

    st.subheader("2. Priebeh redukčnej fázy krok po kroku")
    run_history = sim_result["history"]
    max_step = len(run_history) - 1

    selected_step = st.slider(
        "Zvoľte krok eliminácie:",
        min_value=0,
        max_value=max_step,
        value=max_step,
        format="Krok %d",
        help="Krok 0 predstavuje pôvodné vzorky. Každý ďalší krok eliminuje jeden b-bitový blok.",
    )

    step_info = run_history[selected_step]
    X_mat = step_info["X"]

    st.markdown(
        f"**Aktuálny stav:** `{step_info['sample_type']}` | Počet zostávajúcich riadkov: **{len(X_mat):,}**"
    )

    col_mat, col_cls = st.columns([3, 2])

    with col_mat:
        total_rows = len(X_mat)
        if total_rows > 0:
            disp_rows = st.slider(
                "Počet riadkov zobrazených v heatmape:",
                min_value=1,
                max_value=min(total_rows, 100),
                value=min(total_rows, 30),
                step=1,
            )

            fig_mat = px.imshow(
                X_mat[:disp_rows, :],
                color_continuous_scale=[[0, "#ffffff"], [1, "#1f77b4"]],
                labels=dict(x="Pozícia bitu", y="Riadok", color="Hodnota"),
            )
            fig_mat.update_layout(
                coloraxis_showscale=False,
                height=max(300, 40 + disp_rows * 14),
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
        else:
            st.warning("V tomto kroku nezostali žiadne vektory.")

    with col_cls:
        st.markdown("##### Rozdelenie vzoriek do tried ekvivalencie")
        if step_info["classes_info"]:
            st.dataframe(
                pd.DataFrame(step_info["classes_info"]),
                height=250,
                use_container_width=True,
            )
            st.caption(
                f"Zahodených pivotov (reprezentantov) v tomto kroku: **{step_info['pivots_discarded']}**"
            )
        else:
            st.info("Krok 0: Východiskový stav pred začatím eliminácie.")

    if step_info["operations_log"]:
        st.markdown(f"##### Záznam operácií XOR v kroku {selected_step}")
        st.caption("Ukážka sčítania dvoch riadkov a prenosu chýb:")
        st.dataframe(
            pd.DataFrame(step_info["operations_log"]),
            height=200,
            use_container_width=True,
        )

    # Zobrazenie pôvodu (lineage) pre nájdené vektory s váhou 1
    if selected_step == max_step:
        st.markdown("##### Pôvod vzoriek s váhou 1 v počiatočnej množine")
        chosen_bit_idx = st.selectbox(
            "Vyberte bit na analýzu rodokmeňa jeho vzoriek:",
            options=list(range(b)),
            format_func=lambda idx: f"Bit s_{idx + 1} ({len(sim_result['target_matches_map'][idx])} nájdených vzoriek)",
        )

        matches_for_bit = sim_result["target_matches_map"][chosen_bit_idx]
        if len(matches_for_bit) > 0:
            match_row_idx = matches_for_bit[0]
            src_row_ids = step_info["lineage"][match_row_idx]

            orig_X = run_history[0]["X"][src_row_ids]
            orig_c = run_history[0]["c_noisy"][src_row_ids]
            orig_e = run_history[0]["e"][src_row_ids]

            df_lineage = pd.DataFrame(orig_X)
            df_lineage["Označenie (c)"] = orig_c
            df_lineage["Pôvodná chyba (e)"] = orig_e
            df_lineage.index = [f"Vzorka #{idx}" for idx in src_row_ids]

            st.dataframe(df_lineage, use_container_width=True)
            st.success(
                f"Tento vektor $e_{chosen_bit_idx + 1}$ vznikol sčítaním presne **{len(src_row_ids)} pôvodných rovníc** "
                f"(teoreticky $2^{{a-1}} = {2 ** (a - 1)}$). "
                f"Výsledná chyba po sčítaní: $\\sum e_i \\pmod 2 = {sum(orig_e) % 2}$."
            )
        else:
            st.warning(
                f"Pre bit $s_{chosen_bit_idx + 1}$ nevznikla po redukcii žiadna vzorka s váhou 1. Skúste zvýšiť počet počiatočných vzoriek $n$."
            )


if __name__ == "__main__":
    render()